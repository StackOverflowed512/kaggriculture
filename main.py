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
from scheduler import Scheduler


class KaggricultureAgent:
    # Where a worker starts / returns to when we have no observed position.
    # A central tile keeps early walking short; live games overwrite this from obs.
    HOME = (4, 4)

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
        self.workers = {"farmer": {"pos": self.HOME, "carried": 0}}

    def __call__(self, obs, config):
        step = obs.step if hasattr(obs, 'step') else obs.get('step', 0)

        try:
            # Load rules if not already loaded
            if self.rules is None:
                self.rules = self.rules_loader.load_rules()

            # Initialize ActionEmitter for the current turn
            max_orders = self.rules["constants"]["max_market_orders"]
            emitter = ActionEmitter(max_market_orders=max_orders)

            # Phase 2 Integration: end-of-day monitoring update.
            # Realized/missed values are still mocked here; wiring them to real
            # observation deltas is future work and does not affect emission.
            if step > 0 and step % 24 == 0:
                day = step // 24
                self.care_monitor.observe_day(day, 0)
                realized_money = 1000
                projected_yesterday = self.forecaster.last_prediction or 1000
                self.forecaster.observe(day, realized_money, projected_yesterday)

            # Phase 3: run the operational core only when the observation carries
            # board state. A bare observation (no crops/weeds/workers/etc.) leaves
            # the agent in the safe PASS posture of earlier phases.
            state = self._extract_state(obs)
            if state is None:
                emitter.set_farmer_action([])
                return emitter.emit()

            self._run_operations(state, step, emitter)
            return emitter.emit()

        except Exception as e:
            # Exception containment: never crash the season
            self.telemetry.record_exception(step, e)

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

        board_keys = ("crops", "weeds", "empty_tiles", "workers", "shed", "market_stocks")
        if not any(get(k) is not None for k in board_keys):
            return None

        return {
            "step": get("step", 0) or 0,
            "crops": get("crops") or [],
            "weeds": get("weeds") or [],
            "empty_tiles": get("empty_tiles") or [],
            "workers": get("workers"),  # None -> use persistent model
            "shed": get("shed") or {},
            "market_stocks": get("market_stocks") or {},
        }

    # ------------------------------------------------------------------ #
    # Operational core (Phase 3)
    # ------------------------------------------------------------------ #
    def _run_operations(self, state, step, emitter):
        """Drive one turn: daily routines, task scheduling and worker routing."""
        rules = self.rules
        policy = rules["policy"]
        constants = rules["constants"]

        hour = step % 24
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
        elif hour == 23 or shed_usage >= drop_pressure * shed_size:
            self.scheduler.drop_mode = "overflow"
        else:
            self.scheduler.drop_mode = "normal"

        # Keep the crew roster current (and re-hire at midnight roll-over).
        self._sync_workers(state, hour, target_hands)

        # Daily routine at hour 0: re-hire the whole crew, then sell the shed.
        # HIRE and SELL share the market bucket's 10-order cap (engine contract);
        # HIRE is emitted first, mirroring the handover's morning order.
        if hour == 0:
            self._emit_hire(emitter, target_hands)
            self._emit_sells(emitter, shed)

        # Endgame liquidation overlay: phone the broker EVERY turn (not just at
        # hour 0) to convert inventory to cash before the season ends. Buying new
        # land or animals is disabled here (no such purchases are built yet).
        if in_endgame and hour != 0:
            self._emit_sells(emitter, shed)

        # Generate prioritized tasks and route workers to them.
        care_capacity = self.care_monitor.capacity()
        tasks = self.scheduler.generate_tasks(
            state, rules, care_capacity, market_stocks=state["market_stocks"]
        )
        actions = self.scheduler.assign_tasks(
            self._workers_list(),
            tasks,
            rules,
            shed_usage=shed_usage,
            drop_mode=self.scheduler.drop_mode,
        )

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
                }
            roster.setdefault("farmer", {"pos": self.HOME, "carried": 0})
            self.workers = roster
            return

        # Persistent local model: the farmer is permanent; the hired hands go
        # home at midnight, so we (re)create them at hour 0 up to target_hands.
        self.workers.setdefault("farmer", {"pos": self.HOME, "carried": 0})
        if hour == 0:
            for i in range(target_hands):
                self.workers.setdefault(f"hand_{i}", {"pos": self.HOME, "carried": 0})

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
            result.append({"id": wid, "pos": tuple(w["pos"]), "carried": w.get("carried", 0)})
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
            elif verb == "HARVEST":
                worker["carried"] += 1

    # ------------------------------------------------------------------ #
    # Emission helpers
    # ------------------------------------------------------------------ #
    def _emit_hire(self, emitter, target_hands):
        """Emit one HIRE order per hand to bring the crew up to strength."""
        for _ in range(target_hands):
            emitter.add_market_order(["HIRE"])

    def _emit_sells(self, emitter, shed):
        """Emit a SELL order for every shed item that has stock."""
        for item, qty in shed.items():
            if qty and qty > 0:
                emitter.add_market_order(["SELL", item, int(qty)])

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
