"""compliance_audit.py -- scan simulated season actions for violations.

Release step 4 of the design doc's workflow: the compliance and legality audit.
It drives the ``Scheduler`` over a synthetic season and inspects every action it
emits, classifying problems as:

  * **illegal** -- would be rejected by the engine or zero the season: a
    non-list action, an unknown verb, DROP/PICKUP off a shed tile, a PLANT with
    no crop / PICKUP with no item / malformed SELL, or more than
    ``max_market_orders`` market orders in one turn.
  * **wasted**  -- legal but pointless, the "watering an already-watered plant"
    class: WATER on a tile that needs none, HARVEST with nothing ready, DIG with
    no weed, PLANT onto an occupied tile, FEED/CARE/PLACE with no animal home,
    or SELL of stock the shed does not hold.

A clean agent should produce zero of either (the design doc's audit baseline),
so any finding fails the audit. The per-action ``audit_*`` functions are pure
and are exercised directly by the test suite with crafted bad actions.

Usage:
    python compliance_audit.py            # audit the built-in synthetic season

Exit 0 when no findings, non-zero (and a printed report) otherwise.
"""

import argparse
import sys

from action_emitter import ActionEmitter
from rules_loader import RulesLoader
from scheduler import Scheduler, SHED_TILES

MOVE_VERBS = {"NORTH", "SOUTH", "EAST", "WEST"}
WORK_VERBS = {"WATER", "HARVEST", "PLANT", "DIG", "FEED", "CARE",
              "PLACE", "BUILD_COOP", "BUILD_PASTURE"}
SHED_VERBS = {"DROP", "PICKUP"}
MARKET_VERBS = {"HIRE", "SELL", "BUY_PRODUCT"}


def build_board(obs, rules):
    """Index an observation for O(1) tile lookups during the audit."""
    return {
        "crops": {tuple(c["pos"]): c for c in obs.get("crops") or []},
        "weeds": {tuple(w) for w in obs.get("weeds") or []},
        "empty": {tuple(t) for t in obs.get("empty_tiles") or []},
        "homes": {tuple(a["home_pos"]) for a in (obs.get("animals") or [])
                  if a.get("home_built") and a.get("home_pos") is not None},
        "ready_animals": {tuple(a["home_pos"]) for a in (obs.get("animals") or [])
                          if a.get("ready") and a.get("home_pos") is not None},
        "to_die": rules["constants"]["missed_waterings_to_die"],
    }


def _finding(turn, actor, kind, action, detail):
    return {"turn": turn, "actor": actor, "kind": kind, "action": action, "detail": detail}


def audit_worker_action(worker_id, pos, action, board, turn=0):
    """Return findings for one worker action (empty list = clean)."""
    if not isinstance(action, list):
        return [_finding(turn, worker_id, "illegal", action, "action is not a list")]
    if not action:
        return []  # PASS is always legal
    verb = action[0]
    pos = tuple(pos)

    if verb in MOVE_VERBS:
        return []  # movement is always legal on the obstacle-free board

    if verb in SHED_VERBS:
        findings = []
        if pos not in SHED_TILES:
            findings.append(_finding(turn, worker_id, "illegal", action,
                                     f"{verb} off a shed tile at {pos}"))
        if verb == "PICKUP" and len(action) < 2:
            findings.append(_finding(turn, worker_id, "illegal", action, "PICKUP missing item"))
        return findings

    if verb == "WATER":
        crop = board["crops"].get(pos)
        if crop is None:
            return [_finding(turn, worker_id, "wasted", action, f"WATER on empty tile {pos}")]
        dying = crop.get("misses", 0) >= board["to_die"] - 1
        if not dying and not crop.get("needs_water", False):
            return [_finding(turn, worker_id, "wasted", action,
                             f"WATER on a crop needing none at {pos}")]
        return []

    if verb == "HARVEST":
        crop = board["crops"].get(pos)
        if (crop and crop.get("ready")) or pos in board["ready_animals"]:
            return []
        return [_finding(turn, worker_id, "wasted", action, f"HARVEST with nothing ready at {pos}")]

    if verb == "DIG":
        if pos in board["weeds"]:
            return []
        return [_finding(turn, worker_id, "wasted", action, f"DIG with no weed at {pos}")]

    if verb == "PLANT":
        if len(action) < 2:
            return [_finding(turn, worker_id, "illegal", action, "PLANT missing crop type")]
        if pos in board["crops"]:
            return [_finding(turn, worker_id, "wasted", action, f"PLANT on occupied tile {pos}")]
        return []

    if verb in ("FEED", "CARE", "PLACE"):
        if pos in board["homes"]:
            return []
        return [_finding(turn, worker_id, "wasted", action, f"{verb} with no animal home at {pos}")]

    if verb in ("BUILD_COOP", "BUILD_PASTURE"):
        return []  # building on an empty tile is legal; nothing to over-check offline

    return [_finding(turn, worker_id, "illegal", action, f"unknown verb {verb!r}")]


def audit_market_order(order, shed, turn=0):
    """Return findings for one market order (empty list = clean)."""
    if not isinstance(order, list) or not order:
        return [_finding(turn, "market", "illegal", order, "market order is not a non-empty list")]
    verb = order[0]
    if verb not in MARKET_VERBS:
        return [_finding(turn, "market", "illegal", order, f"unknown market verb {verb!r}")]
    if verb == "SELL":
        if len(order) < 3:
            return [_finding(turn, "market", "illegal", order, "SELL missing item/qty")]
        item, qty = order[1], order[2]
        held = shed.get(item, 0) or 0
        if not isinstance(qty, int) or qty <= 0:
            return [_finding(turn, "market", "wasted", order, "SELL of non-positive qty")]
        if held <= 0:
            return [_finding(turn, "market", "wasted", order, f"SELL of {item} not in shed")]
        if qty > held:
            return [_finding(turn, "market", "wasted", order, f"SELL {qty} > {held} held of {item}")]
    if verb == "BUY_PRODUCT" and len(order) < 2:
        return [_finding(turn, "market", "illegal", order, "BUY_PRODUCT missing item")]
    return []


def default_season():
    """A small, deterministic season exercising each action class legally."""
    return [
        # Hour 0: hire + sell shed inventory; workers sitting on their tasks.
        {"step": 24, "crops": [{"pos": (1, 1), "type": "WHEAT", "misses": 1}],
         "weeds": [(5, 5)], "empty_tiles": [(6, 6)], "shed": {"WHEAT": 8, "MELON": 2},
         "market_stocks": {}, "workers": [
             {"id": "farmer", "pos": (1, 1), "carried": 0},
             {"id": "hand_0", "pos": (5, 5), "carried": 0},
             {"id": "hand_1", "pos": (6, 6), "carried": 0}]},
        # Mid-day: a ripe crop, one needing water, one satisfied (must be left alone).
        {"step": 130, "crops": [
            {"pos": (2, 2), "type": "MELON", "ready": True},
            {"pos": (3, 3), "type": "CARROT", "needs_water": True},
            {"pos": (4, 4), "type": "WHEAT", "misses": 0, "needs_water": False}],
         "weeds": [], "empty_tiles": [], "shed": {}, "market_stocks": {},
         "workers": [{"id": "farmer", "pos": (2, 2), "carried": 0},
                     {"id": "hand_0", "pos": (3, 3), "carried": 0},
                     {"id": "hand_1", "pos": (4, 4), "carried": 0}]},
        # Endgame: a carrier must bank its load, nothing else productive.
        {"step": 700, "crops": [], "weeds": [], "empty_tiles": [], "shed": {"MELON": 4},
         "market_stocks": {}, "workers": [{"id": "farmer", "pos": (4, 4), "carried": 6}]},
    ]


def _simulate_turn(sched, obs, rules):
    """Mirror main.py's per-turn decisions and return (worker rows, market orders).

    Market orders come from the *shared* ``Scheduler.daily_market_orders`` helper
    -- the same one main.py emits from -- pushed through the real
    ``ActionEmitter`` so the audit sees the identical capped bucket the engine
    would, with no re-implemented ladder to drift from the agent. worker rows are
    ``(worker_id, pos, action)``; positions come from the observation so the
    semantic audit knows exactly where each action lands.
    """
    shed = obs.get("shed") or {}
    shed_usage = sum(v for v in shed.values() if v)
    hours_per_day = rules["constants"].get("hours_per_day", 24)
    step = obs.get("step", 0)
    hour = step % hours_per_day
    in_endgame = step >= rules["policy"].get("endgame_start_turn", 670)
    drop_mode = "endgame" if in_endgame else ("overflow" if hour == hours_per_day - 1 else "normal")

    tasks = sched.generate_tasks(obs, rules, care_capacity=100,
                                 market_stocks=obs.get("market_stocks"))
    workers = obs.get("workers") or []
    actions = sched.assign_tasks(workers, tasks, rules, shed_usage=shed_usage, drop_mode=drop_mode)
    pos_by_id = {w["id"]: tuple(w["pos"]) for w in workers}
    rows = [(wid, pos_by_id.get(wid, SHED_TILES[0]), act) for wid, act in actions.items()]

    emitter = ActionEmitter(max_market_orders=rules["constants"]["max_market_orders"])
    for order in sched.daily_market_orders(obs, rules, hour, in_endgame, shed_usage=shed_usage):
        emitter.add_market_order(order)
    return rows, emitter.emit()["market"]


def audit_season(rules, season=None):
    """Audit every turn of ``season`` (default: the built-in one). Returns findings."""
    sched = Scheduler()
    season = season if season is not None else default_season()
    max_orders = rules["constants"]["max_market_orders"]
    findings = []

    for turn, obs in enumerate(season):
        board = build_board(obs, rules)
        rows, orders = _simulate_turn(sched, obs, rules)
        for worker_id, pos, action in rows:
            findings.extend(audit_worker_action(worker_id, pos, action, board, turn))
        for order in orders:
            findings.extend(audit_market_order(order, obs.get("shed") or {}, turn))
        if len(orders) > max_orders:
            findings.append(_finding(turn, "market", "illegal", orders,
                                     f"{len(orders)} market orders > cap {max_orders}"))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit simulated season actions for legality/waste.")
    parser.add_argument("--rules", default="rules_validated.json", help="rules file")
    args = parser.parse_args(argv)

    rules = RulesLoader(args.rules).load_rules()
    findings = audit_season(rules)

    if not findings:
        print("OK: audited season is clean (0 illegal, 0 wasted actions)")
        return 0

    illegal = sum(1 for f in findings if f["kind"] == "illegal")
    wasted = sum(1 for f in findings if f["kind"] == "wasted")
    for f in findings:
        print(f"[{f['kind'].upper()}] turn {f['turn']} {f['actor']}: {f['detail']} "
              f"(action={f['action']})", file=sys.stderr)
    print(f"\nFAIL: {illegal} illegal, {wasted} wasted actions", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
