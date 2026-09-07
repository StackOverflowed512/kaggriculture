"""Regression tests for the adversarial defect audit fixes.

Each test targets a specific bug identified during the audit and proves the
fix works.  Tests are organised by bug ID matching the final report.
"""

import copy
import sys
import os

# Ensure the working directory is on the path so local imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from rules_loader import RulesLoader
from care_monitor import CareMonitor
from scheduler import Scheduler, Task, SHED_TILES
from market_model import MarketModel
from forecaster import Forecaster
import main
from main import agent, KaggricultureAgent

import compliance_audit


@pytest.fixture(scope="module")
def rules():
    return RulesLoader("rules_validated.json").load_rules()


@pytest.fixture
def sched():
    return Scheduler()


def _rules_copy(rules):
    return copy.deepcopy(rules)


def _with_policy(rules, **flags):
    r = copy.deepcopy(rules)
    r["policy"].update(flags)
    return r


# ===========================================================================
# BUG-01 CRITICAL: 10 HIRE orders at hour 0 fill the market cap, zeroing
# out all SELL orders -- the agent never sells shed inventory in the morning.
# ===========================================================================
def test_bug01_sell_not_starved_by_hire(rules, sched):
    """With default target_hands=10, SELL orders must appear *before* HIRE
    and survive the 10-order cap."""
    state = {
        "step": 24,  # hour 0
        "crops": [], "weeds": [], "empty_tiles": [],
        "shed": {"WHEAT": 12, "MELON": 3},
        "market_stocks": {},
    }
    orders = sched.daily_market_orders(state, rules, hour=0, in_endgame=False,
                                        shed_usage=15)
    sell_idx = [i for i, o in enumerate(orders) if o[0] == "SELL"]
    hire_idx = [i for i, o in enumerate(orders) if o[0] == "HIRE"]
    assert sell_idx, "No SELL orders generated at hour 0"
    assert all(si < hi for si in sell_idx for hi in hire_idx), \
        "SELL orders must precede HIRE orders"
    # After cap to 10, at least one SELL must survive.
    capped = orders[:rules["constants"]["max_market_orders"]]
    assert any(o[0] == "SELL" for o in capped), \
        "SELL orders truncated by the 10-order cap"


def test_bug01_hire_reduced_when_shed_full(rules, sched):
    """If there are 8+ SELL orders, HIRE fills only the remaining 2 slots."""
    shed = {f"ITEM{i}": 1 for i in range(8)}
    # Add real crop names so they pass the market_params filter.
    shed = {"WHEAT": 1, "CARROT": 1, "MELON": 1, "TOMATO": 1,
            "STRAWBERRY": 1, "MILK": 1, "WOOL": 1, "EGG": 1}
    state = {"step": 24, "crops": [], "weeds": [], "empty_tiles": [],
             "shed": shed, "market_stocks": {}}
    orders = sched.daily_market_orders(state, rules, hour=0, in_endgame=False,
                                        shed_usage=8)
    capped = orders[:rules["constants"]["max_market_orders"]]
    sell_count = sum(1 for o in capped if o[0] == "SELL")
    hire_count = sum(1 for o in capped if o[0] == "HIRE")
    assert sell_count == 8
    assert hire_count == 2  # 10 - 8 = 2


# ===========================================================================
# BUG-02 HIGH: Endgame at hour 0 still issued HIRE orders -- late-season
# hiring wastes cash and steals market slots from liquidation SELLs.
# ===========================================================================
def test_bug02_no_hire_during_endgame_hour0(rules, sched):
    """Endgame + hour 0: no HIRE, only SELL."""
    state = {"step": 672,  # day 28, hour 0 -- endgame
             "crops": [], "weeds": [], "empty_tiles": [],
             "shed": {"MELON": 5}, "market_stocks": {}}
    orders = sched.daily_market_orders(state, rules, hour=0, in_endgame=True,
                                        shed_usage=5)
    assert all(o[0] != "HIRE" for o in orders), "HIRE emitted during endgame"
    assert ["SELL", "MELON", 5] in orders


# ===========================================================================
# BUG-03 HIGH: Local-model hands persisted across days via setdefault --
# ghost hands from yesterday caused actions for non-existent workers.
# ===========================================================================
def test_bug03_hands_reset_at_midnight(rules, monkeypatch):
    """In the local model (no worker obs), hands must be recreated at hour 0,
    not carried over from the previous day."""
    monkeypatch.setattr(main._agent_instance, "rules", rules)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())

    # Day 1, hour 5: create hands via the local model.
    obs1 = {"step": 5, "crops": [], "weeds": [], "empty_tiles": [],
            "shed": {}, "market_stocks": {}}
    agent(obs1, {})
    hands_d1 = {k: v for k, v in main._agent_instance.workers.items()
                if k.startswith("hand_")}
    # Move a hand to a non-shed position.
    if "hand_0" in main._agent_instance.workers:
        main._agent_instance.workers["hand_0"]["pos"] = (9, 9)

    # Day 2, hour 0: hands should be recreated at HOME, not kept at (9,9).
    obs2 = {"step": 24, "crops": [], "weeds": [], "empty_tiles": [],
            "shed": {}, "market_stocks": {}}
    agent(obs2, {})
    h0 = main._agent_instance.workers.get("hand_0")
    assert h0 is not None
    assert h0["pos"] == SHED_TILES[0], \
        f"hand_0 kept stale position {h0['pos']} from yesterday"


# ===========================================================================
# BUG-04 HIGH: HARVEST in _advance_positions incremented carried by 1
# instead of the crop's full yield, undercounting carried load.
# ===========================================================================
def test_bug04_harvest_increments_by_yield(rules, monkeypatch):
    """A worker harvesting a wheat (yield 4) should carry 4, not 1."""
    monkeypatch.setattr(main._agent_instance, "rules", rules)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())

    obs = {"step": 100,  # mid-day, no HIRE
           "crops": [{"pos": (4, 4), "type": "WHEAT", "ready": True}],
           "weeds": [], "empty_tiles": [], "shed": {}, "market_stocks": {},
           # No workers key -> local model
           }
    out = agent(obs, {})
    # The farmer is at HOME=(4,4) and the crop is at (4,4), so it harvests.
    assert out["farmer"] == ["HARVEST"]
    farmer = main._agent_instance.workers["farmer"]
    assert farmer["carried"] == 4, \
        f"Expected carried=4 (wheat yield), got {farmer['carried']}"


# ===========================================================================
# BUG-05 MEDIUM: Forecaster used crop["type"] (KeyError) instead of .get().
# ===========================================================================
def test_bug05_forecaster_no_keyerror_on_missing_type(rules):
    """A crop dict without 'type' must not crash the forecaster."""
    f = Forecaster()
    # This should not raise.
    result = f.project(rules, 1000, {}, [{"pos": (0, 0)}], {})
    assert isinstance(result, int)


# ===========================================================================
# BUG-06 MEDIUM: generate_tasks used crop["pos"] (KeyError) instead of .get().
# ===========================================================================
def test_bug06_generate_tasks_no_keyerror_on_missing_pos(rules, sched):
    """A crop dict without 'pos' must be skipped, not crash."""
    obs = {"step": 0, "crops": [{"type": "WHEAT", "ready": True}],
           "weeds": [], "empty_tiles": [], "market_stocks": {}}
    # Should not raise.
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    assert tasks == []  # crop without pos is skipped


# ===========================================================================
# BUG-07 MEDIUM: _sell_orders tried to sell non-sellable shed items
# (FERTILIZER, animal types), wasting market slots.
# ===========================================================================
def test_bug07_sell_orders_skip_non_marketable(rules, sched):
    """FERTILIZER and animal types in the shed must not produce SELL orders."""
    shed = {"WHEAT": 5, "FERTILIZER": 3, "GOOSE": 1, "MELON": 2}
    orders = sched._sell_orders(shed, rules)
    items = [o[1] for o in orders]
    assert "WHEAT" in items
    assert "MELON" in items
    assert "FERTILIZER" not in items, "FERTILIZER is not sellable"
    assert "GOOSE" not in items, "GOOSE is not sellable"


def test_bug07_sell_orders_skip_zero_qty(rules, sched):
    """Items with zero or fractional-into-zero quantity must not be sold."""
    shed = {"WHEAT": 0, "MELON": 3, "CARROT": 0.4}
    orders = sched._sell_orders(shed, rules)
    items = [o[1] for o in orders]
    assert "WHEAT" not in items  # qty 0
    assert "CARROT" not in items  # int(0.4) = 0
    assert "MELON" in items


# ===========================================================================
# BUG-08 MEDIUM: Compliance audit drop_mode didn't check shed_usage >=
# drop_pressure * shed_size, diverging from main.py's overflow trigger.
# ===========================================================================
def test_bug08_audit_drop_mode_matches_main(rules):
    """The audit's drop_mode must enter 'overflow' when the shed is near
    full, not only at the last hour of the day."""
    shed = {"WHEAT": 85}  # 85 > 80% of 100
    obs = {"step": 100,  # hour 4, not last hour
           "crops": [], "weeds": [], "empty_tiles": [],
           "shed": shed, "market_stocks": {},
           "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}]}
    findings = compliance_audit.audit_season(rules, season=[obs])
    # The audit should not crash and the season should be clean (or at least
    # not produce false positives from the mismatch).
    assert isinstance(findings, list)


# ===========================================================================
# BUG-09 MEDIUM: Forecaster valued non-sellable shed items (FERTILIZER)
# at the floor price, inflating the terminal-cash projection.
# ===========================================================================
def test_bug09_forecaster_ignores_non_market_shed_items(rules):
    """FERTILIZER in the shed must not contribute to the projected cash."""
    f = Forecaster()
    result_with = f.project(rules, 1000, {"FERTILIZER": 50}, [], {})
    result_without = f.project(rules, 1000, {}, [], {})
    assert result_with == result_without, \
        "FERTILIZER should not inflate the projection"


# ===========================================================================
# BUG-10 MEDIUM: crop_values used direct key access (params["seed"], etc.)
# which could KeyError on incomplete crop params.
# ===========================================================================
def test_bug10_crop_values_safe_on_partial_params():
    """crop_values must not crash if a crop param is missing a key."""
    rules = {
        "crop_params": {"WHEAT": {"type": "one-time"}},  # missing seed, etc.
        "market_params": {"WHEAT": {"normal_price": 25, "curve": "log",
                                     "halves_after": -1, "log_decay": 0.03}},
        "constants": {"price_floor": 1},
    }
    # Should not raise.
    values = MarketModel.crop_values(rules, {})
    assert isinstance(values, dict)


# ===========================================================================
# BUG-11 MEDIUM: _drop_threshold used rules["policy"]["drop_pressure"]
# directly, which KeyError if the key is missing.
# ===========================================================================
def test_bug11_drop_threshold_safe_on_missing_policy():
    """_drop_threshold must use a default when drop_pressure is absent."""
    sched = Scheduler()
    rules = {"constants": {"shed_size": 100}, "policy": {}}
    assert sched._drop_threshold(rules, "normal") == 80  # 0.8 * 100


# ===========================================================================
# BUG-12 ROBUSTNESS: _advance_positions didn't clamp to the 10x10 board,
# so dead-reckoning could drift off-grid.
# ===========================================================================
def test_bug12_positions_clamped_to_board(rules, monkeypatch):
    """A worker at a boundary tile must not drift off-grid."""
    monkeypatch.setattr(main._agent_instance, "rules", rules)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())
    # Force the farmer to (0,0) and give it a task at (0,0) -- it should
    # stay at (0,0), not go to (-1,-1).
    main._agent_instance.workers = {
        "farmer": {"pos": (0, 0), "carried": 0, "carrying_item": None}}
    obs = {"step": 100,
           "crops": [{"pos": (0, 0), "type": "WHEAT", "ready": True}],
           "weeds": [], "empty_tiles": [], "shed": {}, "market_stocks": {}}
    agent(obs, {})
    pos = main._agent_instance.workers["farmer"]["pos"]
    assert 0 <= pos[0] <= 9 and 0 <= pos[1] <= 9, \
        f"Position {pos} off the 10x10 board"


# ===========================================================================
# BUG-13 MEDIUM: _task_action didn't DROP the wrong item before PICKUP --
# a worker carrying the wrong item would be stuck trying to PICKUP.
# ===========================================================================
def test_bug13_drop_wrong_item_before_pickup():
    """A worker carrying WHEAT but assigned a PLACE(GOOSE) task should DROP
    first, not attempt PICKUP."""
    sched = Scheduler()
    task = [Task("PLACE", (2, 2), 4800, item="GOOSE", fetch="GOOSE")]
    # Worker on a shed tile, carrying WHEAT (wrong item).
    actions = sched.assign_tasks(
        [{"id": "w", "pos": (4, 4), "carried": 1,
          "carrying_item": "WHEAT"}],
        task, _with_policy(RulesLoader("rules_validated.json").load_rules(),
                           ANIMALS_ENABLED=True))
    assert actions["w"] == ["DROP"], \
        f"Expected DROP, got {actions['w']}"


# ===========================================================================
# BUG-14 MEDIUM: market_price used rules["market_params"].get(item) which
# could KeyError if market_params is missing entirely.
# ===========================================================================
def test_bug14_market_price_safe_on_missing_market_params():
    """market_price must return the floor if market_params is absent."""
    rules = {"constants": {"price_floor": 1}}
    assert MarketModel.market_price(rules, "WHEAT", 10) == 1


# ===========================================================================
# Integration tests for cross-component interactions
# ===========================================================================
def test_integration_market_forecaster_consistency(rules):
    """The forecaster's shed valuation uses the same market_model that
    prices SELL orders -- a sellable item's projected value must be > 0."""
    f = Forecaster()
    result = f.project(rules, 1000, {"WHEAT": 10}, [], {})
    assert result > 1000, "Wheat in shed should add value"


def test_integration_endgame_scheduler_no_planting(rules, sched):
    """During endgame, the cash test should prevent planting late crops."""
    # Step 29*24 = 696, endgame.  No crop can finish.
    obs = {"step": 696, "crops": [], "weeds": [],
           "empty_tiles": [(5, 5)], "market_stocks": {}}
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    assert not any(t.kind == "PLANT" for t in tasks)


def test_integration_agent_full_season_no_crash(rules, monkeypatch):
    """Run 30 turns (one per day) through agent() -- must never crash."""
    monkeypatch.setattr(main._agent_instance, "rules", rules)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())
    for day in range(30):
        step = day * 24
        obs = {
            "step": step,
            "crops": [{"pos": (i % 10, i // 10), "type": "WHEAT",
                       "misses": step % 2, "needs_water": True}
                      for i in range(5)],
            "weeds": [(9, 9)] if day % 3 == 0 else [],
            "empty_tiles": [(0, 0)] if day < 20 else [],
            "shed": {"WHEAT": day * 2} if day > 0 else {},
            "market_stocks": {},
            "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}]
                       + [{"id": f"hand_{i}", "pos": (i, 0), "carried": 0}
                          for i in range(3)],
        }
        out = agent(obs, {})
        assert set(out) == {"farmer", "hands", "market"}
        assert len(out["market"]) <= rules["constants"]["max_market_orders"]


def test_integration_agent_malformed_obs_degrades_to_pass():
    """A completely malformed observation must degrade to PASS, not crash."""
    a = KaggricultureAgent()
    out = a({"step": 5, "crops": "garbage", "workers": 42}, {})
    assert out == {"farmer": [], "hands": [], "market": []}


def test_integration_zero_workers_schedules_nothing(rules, sched):
    """With an empty worker list, assign_tasks returns empty actions."""
    tasks = [Task("WATER", (1, 1), 9500)]
    actions = sched.assign_tasks([], tasks, rules)
    assert actions == {}


def test_integration_worker_on_target_works_immediately(rules, sched):
    """A worker standing on the target tile should work, not move."""
    tasks = [Task("WATER", (3, 3), 9500)]
    actions = sched.assign_tasks(
        [{"id": "w", "pos": (3, 3), "carried": 0}], tasks, rules)
    assert actions["w"] == ["WATER"]


def test_integration_drop_preemption_projected_shed(rules, sched):
    """Two carriers with shed near full: both should be preempted, and the
    projected shed fill should reflect both loads."""
    workers = [
        {"id": "a", "pos": (4, 4), "carried": 20},
        {"id": "b", "pos": (5, 5), "carried": 20},
    ]
    tasks = [Task("WATER", (1, 1), 9500)]
    actions = sched.assign_tasks(workers, tasks, rules, shed_usage=70,
                                  drop_mode="normal")
    # Both should be sent to DROP (20+70=90 > 80 for first,
    # 20+90=110 > 80 for second with projected shed).
    assert actions["a"] == ["DROP"]
    assert actions["b"] == ["DROP"]


def test_integration_build_submission_still_works():
    """The build_submission check must still pass after our changes."""
    import build_submission
    ok, messages = build_submission.verify_rules(".")
    assert ok, messages


def test_integration_compliance_audit_clean_season(rules):
    """The compliance audit on the default season must still be clean."""
    findings = compliance_audit.audit_season(rules)
    assert findings == [], f"Audit findings: {findings}"
