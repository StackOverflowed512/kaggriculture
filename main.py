import os
import sys

# In a Kaggle environment, all imports would be bundled.
# For this local workspace, we import them directly.
from rules_loader import RulesLoader
from action_emitter import ActionEmitter
from telemetry import Telemetry
from market_model import MarketModel
from care_monitor import CareMonitor
from forecaster import Forecaster
from scheduler import Scheduler, SHED_TILES


class KaggricultureAgent:
    # Where a worker starts / returns to when we have no observed position.
    # The shed's first centre tile keeps early walking short (it is where drops
    # happen) and ties the default to the single SHED_TILES source; live games
    # overwrite this from obs.
    HOME = SHED_TILES[0]

    # Coordinate deltas for dead-reckoning persistent worker positions between
    # turns (see scheduler.py for the (x, y) / NORTH-SOUTH-EAST-WEST convention).
    _MOVE_DELTA = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }

    def __init__(self):
        self.telemetry = Telemetry()
        self.rules_loader = RulesLoader()
        self.rules = None
        self.care_monitor = CareMonitor()
        self.forecaster = Forecaster()
        self.scheduler = Scheduler()
        # Persistent worker state: id -> {"pos": (x, y), "carried": int}.
        # The permanent farmer is always present; hired hands are (re)created
        # each morning and dead-reckoned between turns when the observation does
        # not report positions directly.
        self.workers = {"farmer": {"pos": self.HOME, "carried": 0, "carrying_item": None}}

    def __call__(self, obs, config):
        step = obs.step if hasattr(obs, 'step') else obs.get('step', 0)

        try:
            # Load rules if not already loaded
            if self.rules is None:
                self.rules = self.rules_loader.load_rules()

            constants = self.rules["constants"]
            hours_per_day = constants.get("hours_per_day", 24)
            season_length = constants.get("season_length", 720)

            # Initialize ActionEmitter for the current turn
            max_orders = constants["max_market_orders"]
            emitter = ActionEmitter(max_market_orders=max_orders)

            # Phase 3: run the operational core only when the observation carries
            # board state. A bare observation (no crops/weeds/workers/etc.) leaves
            # the agent in the safe PASS posture of earlier phases.
            state = self._extract_state(obs)
            if state is None:
                emitter.set_farmer_action([])
                return emitter.emit()

            # End-of-day monitoring update, driven by real observation signals
            # where the engine reports them. Missed-watering count feeds the
            # CareMonitor's acreage control; realized cash calibrates yesterday's
            # forecast. Absent fields default to no-ops so the monitors stay
            # dormant rather than acting on mocked numbers.
            if step > 0 and step % hours_per_day == 0:
                day = step // hours_per_day
                self.care_monitor.observe_day(day, state.get("missed_waterings", 0) or 0)
                realized_money = state.get("money")
                projected_yesterday = self.forecaster.last_prediction
                if realized_money is not None and projected_yesterday:
                    self.forecaster.observe(day, realized_money, projected_yesterday)

            self._run_operations(state, step, emitter)

            # Refresh the terminal-cash projection for tomorrow's calibration,
            # but only when the observation actually reports current cash.
            money = state.get("money")
            if money is not None:
                self.forecaster.project(
                    self.rules, money, state["shed"], state["crops"],
                    state["market_stocks"],
                )

            if step == season_length - 1:
                self.telemetry.flush_to_disk()
            return emitter.emit()

        except Exception as e:
            # Exception containment: never crash the season
            self.telemetry.record_exception(step, e)
            self.telemetry.flush_to_disk()

            # Degrade gracefully to PASS
            return {
                "farmer": [],
                "hands": [],
                "market": []
            }

    # ------------------------------------------------------------------ #
    # Observation handling
    # ------------------------------------------------------------------ #
    def _extract_state(self, obs):
        """Normalise the observation into a plain board-state dict, or None.

        Returns None for observations that carry no board information at all, so
        the wrapper falls back to the safe minimal PASS emitted in Phases 1-2.
        """
        if isinstance(obs, dict):
            get = lambda key, default=None: obs.get(key, default)
        else:
            get = lambda key, default=None: getattr(obs, key, default)

        board_keys = ("crops", "weeds", "empty_tiles", "workers", "shed", "market_stocks", "animals")
        if not any(get(k) is not None for k in board_keys):
            return None

        # Current cash may be reported as "money" or "cash"; keep it as None when
        # absent (distinct from a real $0) so the forecaster only calibrates on
        # genuine readings.
        money = get("money")
        if money is None:
            money = get("cash")

        return {
            "step": get("step", 0) or 0,
            "crops": get("crops") or [],
            "weeds": get("weeds") or [],
            "empty_tiles": get("empty_tiles") or [],
            "workers": get("workers"),  # None -> use persistent model
            "shed": get("shed") or {},
            "market_stocks": get("market_stocks") or {},
            "animals": get("animals") or [],  # Phase 4: animal lifecycle slots
            "money": money,  # None when the observation does not report cash
            "missed_waterings": get("missed_waterings", 0) or 0,
        }

    # ------------------------------------------------------------------ #
    # Operational core (Phase 3)
    # ------------------------------------------------------------------ #
    def _run_operations(self, state, step, emitter):
        """Drive one turn: daily routines, task scheduling and worker routing."""
        rules = self.rules
        policy = rules["policy"]
        constants = rules["constants"]

        hours_per_day = constants.get("hours_per_day", 24)
        hour = step % hours_per_day
        endgame_start = policy.get("endgame_start_turn", 670)
        target_hands = policy.get("target_hands", 10)
        drop_pressure = policy.get("drop_pressure", 0.8)
        shed_size = constants["shed_size"]

        shed = state["shed"]
        shed_usage = sum(v for v in shed.values() if v) if shed else 0
        in_endgame = step >= endgame_start

        # Drop mode governs the DROP pre-emption threshold for this turn:
        #   endgame  -> bank every carried item (threshold 0)
        #   overflow -> the last hour of the day, or a shed already near full,
        #               makes a spill likely: tighten to half the shed
        #   normal   -> the standard 80% threshold
        if in_endgame:
            self.scheduler.drop_mode = "endgame"
        elif hour == hours_per_day - 1 or shed_usage >= drop_pressure * shed_size:
            self.scheduler.drop_mode = "overflow"
        else:
            self.scheduler.drop_mode = "normal"

        # Keep the crew roster current (and re-hire at midnight roll-over).
        self._sync_workers(state, hour, target_hands)

        # Market plan for the turn (morning HIRE/SELL, input buys, endgame
        # liquidation). Shared with the compliance audit via the scheduler so
        # both emit an identical bucket; the emitter applies the 10-order cap.
        for order in self.scheduler.daily_market_orders(
            state, rules, hour, in_endgame, shed_usage=shed_usage
        ):
            emitter.add_market_order(order)

        # Generate prioritized tasks and route workers to them.
        care_capacity = self.care_monitor.capacity()
        tasks = self.scheduler.generate_tasks(
            state, rules, care_capacity, market_stocks=state["market_stocks"]
        )
        workers = self._workers_list()
        actions = self.scheduler.assign_tasks(
            workers,
            tasks,
            rules,
            shed_usage=shed_usage,
            drop_mode=self.scheduler.drop_mode,
        )

        # Feed the real idle fraction back to the CareMonitor so it can grow
        # acreage when the crew has spare hands (and shrink it on thirst above).
        if workers:
            idle = sum(1 for w in workers if not actions.get(w["id"])) / len(workers)
            self.care_monitor.note_idle(idle)

        self._emit_worker_actions(actions, emitter)
        self._advance_positions(state, actions)

    # ------------------------------------------------------------------ #
    # Worker roster helpers
    # ------------------------------------------------------------------ #
    def _sync_workers(self, state, hour, target_hands):
        """Refresh the roster from the observation, or maintain it locally."""
        if state["workers"] is not None:
            # Trust the live observation for positions and carried loads.
            roster = {}
            for w in state["workers"]:
                roster[w["id"]] = {
                    "pos": tuple(w["pos"]),
                    "carried": w.get("carried", 0),
                    "carrying_item": w.get("carrying_item"),
                }
            roster.setdefault("farmer", {"pos": self.HOME, "carried": 0, "carrying_item": None})
            self.workers = roster
            return

        # Persistent local model: the farmer is permanent; the hired hands go
        # home at midnight, so we (re)create them at hour 0 up to target_hands.
        self.workers.setdefault("farmer", {"pos": self.HOME, "carried": 0, "carrying_item": None})
        if hour == 0:
            for i in range(target_hands):
                self.workers.setdefault(f"hand_{i}", {"pos": self.HOME, "carried": 0, "carrying_item": None})

    def _workers_list(self):
        """Roster as a list of dicts, farmer first then hands in index order."""
        def order(worker_id):
            if worker_id == "farmer":
                return (0, -1)
            if worker_id.startswith("hand_"):
                try:
                    return (1, int(worker_id.split("_", 1)[1]))
                except ValueError:
                    return (2, 0)
            return (2, 0)

        result = []
        for wid in sorted(self.workers, key=order):
            w = self.workers[wid]
            result.append({
                "id": wid,
                "pos": tuple(w["pos"]),
                "carried": w.get("carried", 0),
                "carrying_item": w.get("carrying_item"),
            })
        return result

    def _advance_positions(self, state, actions):
        """Dead-reckon persistent worker positions from the actions we emitted.

        Only meaningful in the local model; when the observation reports worker
        state we rebuild the roster from it each turn and this is harmless.
        """
        if state["workers"] is not None:
            return
        for wid, action in actions.items():
            worker = self.workers.get(wid)
            if not worker or not action:
                continue
            verb = action[0]
            if verb in self._MOVE_DELTA:
                dx, dy = self._MOVE_DELTA[verb]
                x, y = worker["pos"]
                worker["pos"] = (x + dx, y + dy)
            elif verb == "DROP":
                worker["carried"] = 0
                worker["carrying_item"] = None
            elif verb == "PICKUP":
                # Fetching a specific item from the shed (wheat, an animal).
                worker["carried"] = worker.get("carried", 0) + 1
                worker["carrying_item"] = action[1] if len(action) > 1 else None
            elif verb in ("PLACE", "FEED"):
                # The carried item is consumed onto the tile (placed / fed).
                worker["carried"] = max(0, worker.get("carried", 0) - 1)
                worker["carrying_item"] = None
            elif verb == "HARVEST":
                worker["carried"] += 1

    # ------------------------------------------------------------------ #
    # Emission helpers
    # ------------------------------------------------------------------ #
    def _emit_worker_actions(self, actions, emitter):
        """Route the scheduler's per-worker actions into the emitter buckets."""
        for worker in self._workers_list():
            action = actions.get(worker["id"], [])
            if worker["id"] == "farmer":
                emitter.set_farmer_action(action)
            else:
                emitter.add_hand_action(action)


# Global instance to persist state across turns
_agent_instance = KaggricultureAgent()

def agent(obs, config):
    """
    Public Kaggle entry point. Invokes protected internal execution.
    Never raises an exception.
    """
    return _agent_instance(obs, config)
