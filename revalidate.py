"""revalidate.py -- offline behavioural acceptance checks for the agent.

Release step 3/5 of the design doc's workflow: run behavioural revalidation and
automated acceptance/release checks. This drives the real ``agent()`` wrapper
and the ``Scheduler`` over synthetic observations and asserts the properties a
release must hold, without needing the Kaggle engine:

  * **Output structure**   -- every turn returns the ``{farmer, hands, market}``
    envelope with list-typed actions and at most ``max_market_orders`` orders.
  * **Exception containment** -- a malformed observation degrades to PASS and is
    counted, never raised (a single crash zeroes the season).
  * **Latency**            -- each turn returns well within the 1s act timeout.
  * **Watering coverage**  -- workers standing on crops about to die actually
    WATER them (the care obligation the agent exists to meet).
  * **Archive layout**     -- if a ``submission.tar.gz`` is present, it contains
    exactly the runtime modules and rules (skipped if not built yet).

Usage:
    python revalidate.py                 # run all checks
    python revalidate.py --max-latency 0.5

Exit 0 when every check passes, non-zero otherwise.
"""

import argparse
import os
import sys
import tarfile
import time

from rules_loader import RulesLoader
from scheduler import Scheduler, Task
from build_submission import RUNTIME_MODULES, RULES_FILE, DEFAULT_OUTPUT
import main as agent_main
from main import KaggricultureAgent


# The act-timeout is 1s; we assert a comfortable margin below it.
DEFAULT_MAX_LATENCY = 0.5


def _fresh_agent():
    """A clean agent instance so checks never share turn-to-turn state."""
    return KaggricultureAgent()


def _is_list_of_lists(value):
    return isinstance(value, list) and all(isinstance(item, list) for item in value)


def check_output_structure(rules):
    """agent() always returns the engine envelope with list-typed actions."""
    agent = _fresh_agent()
    max_orders = rules["constants"]["max_market_orders"]
    observations = [
        {"step": 0, "crops": [], "weeds": [], "empty_tiles": [(3, 3)],
         "shed": {"WHEAT": 5}, "market_stocks": {},
         "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}]},
        {"step": 100, "crops": [{"pos": (2, 2), "type": "WHEAT", "misses": 1}],
         "weeds": [(7, 7)], "empty_tiles": [], "shed": {}, "market_stocks": {},
         "workers": [{"id": "farmer", "pos": (2, 2), "carried": 0}]},
        {"step": 700, "crops": [], "weeds": [], "empty_tiles": [],
         "shed": {"MELON": 4}, "market_stocks": {},
         "workers": [{"id": "farmer", "pos": (4, 4), "carried": 3}]},
    ]
    for obs in observations:
        out = agent(obs, {})
        if set(out) != {"farmer", "hands", "market"}:
            return False, f"bad envelope keys at step {obs['step']}: {sorted(out)}"
        if not isinstance(out["farmer"], list):
            return False, f"farmer action not a list at step {obs['step']}"
        if not _is_list_of_lists(out["hands"]):
            return False, f"hands not a list of lists at step {obs['step']}"
        if not _is_list_of_lists(out["market"]):
            return False, f"market not a list of lists at step {obs['step']}"
        if len(out["market"]) > max_orders:
            return False, f"{len(out['market'])} market orders > cap {max_orders}"
    return True, f"{len(observations)} turns returned a compliant envelope"


def check_exception_containment():
    """A malformed observation degrades to PASS and is counted, never raised."""
    agent = _fresh_agent()
    before = agent.telemetry.get_exception_count()
    try:
        out = agent({"step": 5, "crops": "not-a-list", "workers": 123}, {})
    except Exception as exc:  # pragma: no cover - the whole point is this cannot happen
        return False, f"wrapper raised: {exc!r}"
    if out != {"farmer": [], "hands": [], "market": []}:
        return False, f"did not degrade to PASS: {out}"
    if agent.telemetry.get_exception_count() <= before:
        return False, "exception was swallowed without being counted"
    return True, "malformed obs degraded to PASS and was counted"


def check_latency(max_latency):
    """Every turn returns within the latency budget; report the worst turn."""
    agent = _fresh_agent()
    worst = 0.0
    for step in range(0, 720, 24):  # one turn per in-game day, hour 0
        obs = {
            "step": step,
            "crops": [{"pos": (i % 10, i // 10), "type": "WHEAT",
                       "misses": step % 2, "needs_water": True} for i in range(20)],
            "weeds": [(9, 9)], "empty_tiles": [(0, 0)], "shed": {"WHEAT": 10},
            "market_stocks": {},
            "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}]
                       + [{"id": f"hand_{i}", "pos": (i, 0), "carried": 0} for i in range(10)],
        }
        start = time.perf_counter()
        agent(obs, {})
        worst = max(worst, time.perf_counter() - start)
    ok = worst < max_latency
    return ok, f"worst turn {worst * 1000:.1f}ms (budget {max_latency * 1000:.0f}ms)"


def check_watering_coverage(rules):
    """Workers standing on about-to-die crops all emit WATER this turn."""
    sched = Scheduler()
    dying = [(1, 1), (8, 2), (4, 5), (0, 9)]
    obs = {
        "step": 100,
        "crops": [{"pos": p, "type": "WHEAT", "misses": 1} for p in dying],
        "weeds": [], "empty_tiles": [], "market_stocks": {},
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    workers = [{"id": f"w{i}", "pos": p, "carried": 0} for i, p in enumerate(dying)]
    actions = sched.assign_tasks(workers, tasks, rules)
    watered = sum(1 for a in actions.values() if a == ["WATER"])
    ok = watered == len(dying)
    return ok, f"{watered}/{len(dying)} dying crops watered on-tile"


def check_archive_layout(archive=DEFAULT_OUTPUT):
    """If a submission archive exists, it holds exactly the expected members."""
    if not os.path.isfile(archive):
        return True, f"{archive} not present -- skipped (run build_submission.py)"
    expected = set(RUNTIME_MODULES) | {RULES_FILE}
    with tarfile.open(archive, "r:gz") as tar:
        members = {os.path.basename(name) for name in tar.getnames()}
    missing = expected - members
    extra = members - expected
    if missing or extra:
        return False, f"archive layout off (missing={sorted(missing)}, extra={sorted(extra)})"
    return True, f"{archive} contains the {len(expected)} expected members"


def revalidate(max_latency=DEFAULT_MAX_LATENCY):
    """Run every acceptance check. Returns ``(ok, results)``."""
    rules = RulesLoader("rules_validated.json").load_rules()
    results = [
        ("output structure", *check_output_structure(rules)),
        ("exception containment", *check_exception_containment()),
        ("latency", *check_latency(max_latency)),
        ("watering coverage", *check_watering_coverage(rules)),
        ("archive layout", *check_archive_layout()),
    ]
    ok = all(row[1] for row in results)
    return ok, results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline behavioural revalidation.")
    parser.add_argument("--max-latency", type=float, default=DEFAULT_MAX_LATENCY,
                        help=f"per-turn latency budget in seconds (default {DEFAULT_MAX_LATENCY})")
    args = parser.parse_args(argv)

    ok, results = revalidate(args.max_latency)
    for name, passed, detail in results:
        tag = "OK  " if passed else "FAIL"
        print(f"[{tag}] {name} -- {detail}", file=sys.stdout if passed else sys.stderr)
    passed_n = sum(1 for r in results if r[1])
    print(f"\n{passed_n}/{len(results)} acceptance checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
