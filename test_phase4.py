"""Phase 4 verification suite (pytest).

Covers the Phase 4 additions:
    * fertilizer market purchases (gated, capacity-clamped),
    * the animal lifecycle -- build home, buy, pickup/place transport, daily
      feed + care, harvest -- with ANIMALS_ENABLED flipped on via a rules copy,
    * end-to-end wiring of both through the real agent() entry point,
    * the release tooling: build_submission, validate_rules, compliance_audit,
      mc and revalidate.

As in the Phase 3 suite, global agent state is isolated with monkeypatch and a
deep-copied rules dict is used whenever a test flips a policy flag.
"""

import copy
import os
import tarfile

import pytest

from rules_loader import RulesLoader
from care_monitor import CareMonitor
from scheduler import Scheduler, Task, SHED_TILES
import main
from main import agent

import build_submission
import validate_rules
import compliance_audit
import mc
import revalidate


@pytest.fixture(scope="module")
def rules():
    return RulesLoader("rules_validated.json").load_rules()


@pytest.fixture
def sched():
    return Scheduler()


def _rules_copy(rules):
    """Deep copy so a test can flip policy flags without touching the shared dict."""
    return copy.deepcopy(rules)


def _with_policy(rules, **flags):
    r = copy.deepcopy(rules)
    r["policy"].update(flags)
    return r


# ===========================================================================
# Fertilizer logic
# ===========================================================================
def test_fertilizer_disabled_by_default(rules, sched):
    """FERTILIZER_ENABLED is off in the shipped rules -> never buy fertilizer."""
    assert rules["policy"]["FERTILIZER_ENABLED"] is False
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}], "shed": {}}
    orders = sched.generate_market_orders(obs, rules, shed_usage=0)
    assert all(o[:2] != ["BUY_PRODUCT", "FERTILIZER"] for o in orders)


def test_fertilizer_qty_counts_only_beneficial_crops(rules):
    """Only crops whose fertilized yield beats their base yield count.
    WHEAT (6>4) and CARROT (4>3) do; MELON (6==6) does not."""
    obs = {
        "crops": [
            {"pos": (0, 0), "type": "WHEAT"},
            {"pos": (1, 0), "type": "CARROT"},
            {"pos": (2, 0), "type": "MELON"},
        ],
        "shed": {},
    }
    assert Scheduler._fertilizer_buy_qty(obs, rules, shed_usage=0) == 2


def test_fertilizer_qty_discounts_existing_shed_stock(rules):
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}, {"pos": (1, 0), "type": "CARROT"}],
           "shed": {"FERTILIZER": 1}}
    assert Scheduler._fertilizer_buy_qty(obs, rules, shed_usage=1) == 1  # need 2, have 1


def test_fertilizer_qty_never_exceeds_shed_capacity(rules):
    """The order is clamped to free shed slots -- a buy can never push past 100."""
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}, {"pos": (1, 0), "type": "CARROT"}],
           "shed": {}}
    # 99 used: only one slot free even though two crops would benefit.
    assert Scheduler._fertilizer_buy_qty(obs, rules, shed_usage=99) == 1
    # Shed full: buying anything would push usage above 100 -> buy nothing.
    assert Scheduler._fertilizer_buy_qty(obs, rules, shed_usage=100) == 0


def test_fertilizer_qty_zero_without_beneficial_crops(rules):
    obs = {"crops": [{"pos": (0, 0), "type": "MELON"}], "shed": {}}
    assert Scheduler._fertilizer_buy_qty(obs, rules, shed_usage=0) == 0


def test_fertilizer_order_emitted_when_enabled(rules, sched):
    r = _with_policy(rules, FERTILIZER_ENABLED=True)
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}, {"pos": (1, 0), "type": "CARROT"}],
           "shed": {}}
    orders = sched.generate_market_orders(obs, r, shed_usage=0)
    assert ["BUY_PRODUCT", "FERTILIZER", 2] in orders


def test_fertilizer_not_bought_when_shed_full(rules, sched):
    r = _with_policy(rules, FERTILIZER_ENABLED=True)
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}], "shed": {"WHEAT": 100}}
    orders = sched.generate_market_orders(obs, r, shed_usage=100)
    assert all(o[:2] != ["BUY_PRODUCT", "FERTILIZER"] for o in orders)


def test_fertilizer_wires_through_agent(rules, monkeypatch):
    """End to end: with the flag on, the agent emits the FERTILIZER buy order."""
    r = _with_policy(rules, FERTILIZER_ENABLED=True)
    monkeypatch.setattr(main._agent_instance, "rules", r)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())
    obs = {
        "step": 100,  # hour 4: no HIRE crowding the market bucket
        "crops": [{"pos": (1, 1), "type": "WHEAT", "misses": 0, "needs_water": False},
                  {"pos": (2, 2), "type": "CARROT", "misses": 0, "needs_water": False}],
        "weeds": [], "empty_tiles": [], "shed": {}, "market_stocks": {},
        "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0, "carrying_item": None}],
    }
    out = agent(obs, {})
    assert ["BUY_PRODUCT", "FERTILIZER", 2] in out["market"]
    assert len(out["market"]) <= rules["constants"]["max_market_orders"]


# ===========================================================================
# Animal lifecycle (ANIMALS_ENABLED flipped on)
# ===========================================================================
@pytest.fixture
def animal_rules(rules):
    return _with_policy(rules, ANIMALS_ENABLED=True)


def test_build_action_maps_home_to_verb():
    assert Scheduler._build_action("COOP") == "BUILD_COOP"
    assert Scheduler._build_action("PASTURE") == "BUILD_PASTURE"
    assert Scheduler._build_action("BARN") is None


def test_animal_build_home_task(animal_rules, sched):
    """An animal with no home yet yields the right BUILD task at its home tile."""
    obs = {"animals": [
        {"type": "GOOSE", "home_pos": (2, 2), "home_built": False},
        {"type": "COW", "home_pos": (7, 7), "home_built": False},
    ]}
    tasks = sched.generate_tasks(obs, animal_rules, care_capacity=100)
    by_pos = {t.pos: t for t in tasks}
    assert by_pos[(2, 2)].kind == "BUILD_COOP"
    assert by_pos[(7, 7)].kind == "BUILD_PASTURE"
    assert by_pos[(2, 2)].priority == animal_rules["daily_routines"]["BUILD_HOME"]["priority"] == 4600


def test_animal_buy_order_when_home_built(animal_rules, sched):
    """Home built, animal not owned -> a BUY_PRODUCT order for that animal."""
    obs = {"animals": [{"type": "GOOSE", "home_pos": (2, 2), "home_built": True, "owned": False}],
           "shed": {}}
    orders = sched.generate_market_orders(obs, animal_rules, shed_usage=0)
    assert orders == [["BUY_PRODUCT", "GOOSE"]]


def test_animal_not_bought_when_shed_full(animal_rules, sched):
    obs = {"animals": [{"type": "GOOSE", "home_pos": (2, 2), "home_built": True, "owned": False}],
           "shed": {"WHEAT": 100}}
    orders = sched.generate_market_orders(obs, animal_rules, shed_usage=100)
    assert orders == []  # +1 would exceed the 100-slot shed


def test_animal_not_bought_before_home_or_when_owned(animal_rules, sched):
    unbuilt = {"animals": [{"type": "GOOSE", "home_pos": (2, 2), "home_built": False}], "shed": {}}
    owned = {"animals": [{"type": "GOOSE", "home_pos": (2, 2), "home_built": True, "owned": True}],
             "shed": {}}
    assert sched.generate_market_orders(unbuilt, animal_rules, shed_usage=0) == []
    assert sched.generate_market_orders(owned, animal_rules, shed_usage=0) == []


def test_animal_place_task_carries_from_shed(animal_rules, sched):
    """An owned-but-unplaced animal becomes a PLACE task that fetches the animal."""
    obs = {"animals": [{"type": "GOOSE", "home_pos": (2, 2),
                        "home_built": True, "owned": True, "placed": False}]}
    tasks = sched.generate_tasks(obs, animal_rules, care_capacity=100)
    place = [t for t in tasks if t.kind == "PLACE"]
    assert len(place) == 1
    task = place[0]
    assert task.pos == (2, 2)
    assert task.priority == animal_rules["daily_routines"]["PLACE_ANIMAL"]["priority"] == 4800
    assert task.item == "GOOSE" and task.fetch == "GOOSE"


def test_placed_animal_generates_harvest_feed_care(animal_rules, sched):
    """A placed animal that is ready, unfed and needing care yields all three
    chores in the right bands: harvest (9100) > feed (8800) > care (4000)."""
    obs = {"animals": [{"type": "GOOSE", "home_pos": (3, 3), "home_built": True,
                        "owned": True, "placed": True, "ready": True,
                        "fed_today": False, "needs_care": True}],
           "market_stocks": {}}
    tasks = sched.generate_tasks(obs, animal_rules, care_capacity=100)
    by_kind = {t.kind: t for t in tasks}
    assert by_kind["HARVEST"].priority == 9100 and by_kind["HARVEST"].pos == (3, 3)
    assert by_kind["HARVEST"].value > 0  # priced off the EGG market
    assert by_kind["FEED"].priority == 8800 and by_kind["FEED"].fetch == "WHEAT"
    assert by_kind["CARE"].priority == 4000


def test_fed_animal_has_no_feed_task(animal_rules, sched):
    obs = {"animals": [{"type": "COW", "home_pos": (5, 6), "home_built": True,
                        "owned": True, "placed": True, "fed_today": True}],
           "market_stocks": {}}
    tasks = sched.generate_tasks(obs, animal_rules, care_capacity=100)
    assert all(t.kind != "FEED" for t in tasks)


# --- Animal transport routing (PICKUP -> carry -> PLACE / FEED) ------------
def test_pickup_action_shape():
    assert Task("PICKUP", (4, 4), 4800, item="GOOSE").action() == ["PICKUP", "GOOSE"]
    assert Task("PLACE", (2, 2), 4800, item="GOOSE", fetch="GOOSE").action() == ["PLACE"]
    assert Task("BUILD_COOP", (2, 2), 4600).action() == ["BUILD_COOP"]


def test_place_routing_fetches_then_places(animal_rules, sched):
    task = [Task("PLACE", (2, 2), 4800, item="GOOSE", fetch="GOOSE")]
    # Empty-handed, away from the shed -> walk to the shed first.
    away = sched.assign_tasks(
        [{"id": "w", "pos": (0, 0), "carried": 0, "carrying_item": None}], task, animal_rules)
    assert away["w"] == ["EAST"]  # (0,0) -> nearest shed tile (4,4)
    # Empty-handed, on a shed tile -> PICKUP the animal.
    on_shed = sched.assign_tasks(
        [{"id": "w", "pos": (4, 4), "carried": 0, "carrying_item": None}], task, animal_rules)
    assert on_shed["w"] == ["PICKUP", "GOOSE"]
    # Carrying the animal, standing on the home tile -> PLACE it.
    at_home = sched.assign_tasks(
        [{"id": "w", "pos": (2, 2), "carried": 1, "carrying_item": "GOOSE"}], task, animal_rules)
    assert at_home["w"] == ["PLACE"]


def test_feed_routing_fetches_wheat_first(animal_rules, sched):
    task = [Task("FEED", (3, 3), 8800, crop="COW", fetch="WHEAT")]
    on_shed = sched.assign_tasks(
        [{"id": "w", "pos": (5, 5), "carried": 0, "carrying_item": None}], task, animal_rules)
    assert on_shed["w"] == ["PICKUP", "WHEAT"]
    carrying = sched.assign_tasks(
        [{"id": "w", "pos": (3, 3), "carried": 1, "carrying_item": "WHEAT"}], task, animal_rules)
    assert carrying["w"] == ["FEED"]


def test_carrying_worker_not_preempted_to_drop(animal_rules, sched):
    """A worker carrying a task item (the animal) must not be diverted to DROP
    -- that would dump the very thing it just fetched."""
    task = [Task("PLACE", (2, 2), 4800, item="GOOSE", fetch="GOOSE")]
    actions = sched.assign_tasks(
        [{"id": "w", "pos": (2, 2), "carried": 1, "carrying_item": "GOOSE"}],
        task, animal_rules, shed_usage=99, drop_mode="endgame")
    assert actions["w"] == ["PLACE"]  # not ["DROP"]


def test_animal_lifecycle_wires_through_agent(rules, monkeypatch):
    """End to end through agent(): build the coop, then buy the goose."""
    r = _with_policy(rules, ANIMALS_ENABLED=True)
    monkeypatch.setattr(main._agent_instance, "rules", r)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())

    base = {"step": 100, "crops": [], "weeds": [], "empty_tiles": [],
            "shed": {}, "market_stocks": {}}

    build = dict(base, workers=[{"id": "farmer", "pos": (2, 2), "carried": 0, "carrying_item": None}],
                 animals=[{"type": "GOOSE", "home_pos": (2, 2), "home_built": False}])
    assert agent(build, {})["farmer"] == ["BUILD_COOP"]

    buy = dict(base, workers=[{"id": "farmer", "pos": (4, 4), "carried": 0, "carrying_item": None}],
               animals=[{"type": "GOOSE", "home_pos": (2, 2), "home_built": True, "owned": False}])
    assert ["BUY_PRODUCT", "GOOSE"] in agent(buy, {})["market"]


def test_animals_and_fertilizer_off_emit_no_purchases(rules, sched):
    """Both flags off (shipped default): no BUY_PRODUCT orders at all."""
    obs = {"crops": [{"pos": (0, 0), "type": "WHEAT"}],
           "animals": [{"type": "GOOSE", "home_pos": (2, 2), "home_built": True, "owned": False}],
           "shed": {}}
    assert sched.generate_market_orders(obs, rules, shed_usage=0) == []


# ===========================================================================
# Build / release validation tooling
# ===========================================================================
def test_submission_members_present():
    assert build_submission.missing_files(".") == []


def test_verify_rules_passes_on_source():
    ok, messages = build_submission.verify_rules(".")
    assert ok, messages


def test_verify_rules_detects_embedded_mismatch(monkeypatch):
    """If the packaged/embedded rules ever diverge from source, refuse to ship."""
    def fake_embedded(root="."):
        drifted = copy.deepcopy(build_submission.source_rules(root))
        drifted["constants"]["shed_size"] = 999
        return drifted
    monkeypatch.setattr(build_submission, "embedded_rules", fake_embedded)
    ok, messages = build_submission.verify_rules(".")
    assert not ok
    assert any("do not match" in m for m in messages)


def test_build_submission_check_only_writes_nothing(tmp_path):
    out = tmp_path / "should_not_exist.tar.gz"
    ok, messages = build_submission.build_submission(output=str(out), check_only=True)
    assert ok, messages
    assert not out.exists()


def test_build_archive_contains_exactly_expected_members(tmp_path):
    out = tmp_path / "submission.tar.gz"
    build_submission.build_archive(output=str(out))
    with tarfile.open(out, "r:gz") as tar:
        members = {os.path.basename(n) for n in tar.getnames()}
    expected = set(build_submission.RUNTIME_MODULES) | {build_submission.RULES_FILE}
    assert members == expected


def test_validate_rules_self_consistency_all_pass(rules):
    results = validate_rules.self_consistency_checks(rules)
    failed = [name for name, ok, _ in results if not ok]
    assert failed == []


def test_validate_rules_catches_broken_price_floor(rules):
    broken = copy.deepcopy(rules)
    broken["constants"]["price_floor"] = 2
    results = validate_rules.self_consistency_checks(broken)
    assert any(name == "price_floor is $1" and not ok for name, ok, _ in results)


def test_validate_rules_catches_scrambled_bands(rules):
    broken = copy.deepcopy(rules)
    broken["daily_routines"]["HARVEST_CROP"]["priority"] = 99999  # now outranks dying-water
    results = validate_rules.self_consistency_checks(broken)
    assert any(name == "priority bands strictly descending" and not ok for name, ok, _ in results)


def test_validate_engine_skipped_without_strict():
    ok, results, status = validate_rules.validate("rules_validated.json", strict=False)
    assert ok and status == "unavailable"


def test_validate_engine_required_under_strict():
    ok, _, status = validate_rules.validate("rules_validated.json", strict=True)
    assert not ok and status == "unavailable"


# ===========================================================================
# Compliance audit
# ===========================================================================
@pytest.fixture
def board(rules):
    obs = {
        "crops": [{"pos": (1, 1), "type": "WHEAT", "needs_water": True, "misses": 0},
                  {"pos": (6, 6), "type": "MELON", "ready": True}],
        "weeds": [(2, 2)],
        "animals": [{"type": "GOOSE", "home_pos": (7, 7), "home_built": True,
                     "owned": True, "placed": True, "ready": True}],
    }
    return compliance_audit.build_board(obs, rules)


def test_audit_clean_worker_actions(board):
    assert compliance_audit.audit_worker_action("w", (1, 1), ["WATER"], board) == []
    assert compliance_audit.audit_worker_action("w", (6, 6), ["HARVEST"], board) == []
    assert compliance_audit.audit_worker_action("w", (7, 7), ["HARVEST"], board) == []  # ready animal
    assert compliance_audit.audit_worker_action("w", (2, 2), ["DIG"], board) == []
    assert compliance_audit.audit_worker_action("w", (4, 4), ["DROP"], board) == []
    assert compliance_audit.audit_worker_action("w", (0, 0), [], board) == []  # PASS
    assert compliance_audit.audit_worker_action("w", (0, 0), ["EAST"], board) == []


def test_audit_flags_wasted_worker_actions(board):
    assert _kind(compliance_audit.audit_worker_action("w", (5, 5), ["WATER"], board)) == "wasted"
    assert _kind(compliance_audit.audit_worker_action("w", (5, 5), ["HARVEST"], board)) == "wasted"
    assert _kind(compliance_audit.audit_worker_action("w", (0, 0), ["DIG"], board)) == "wasted"
    assert _kind(compliance_audit.audit_worker_action("w", (1, 1), ["PLANT", "WHEAT"], board)) == "wasted"


def test_audit_flags_illegal_worker_actions(board):
    assert _kind(compliance_audit.audit_worker_action("w", (0, 0), {"a": 1}, board)) == "illegal"
    assert _kind(compliance_audit.audit_worker_action("w", (0, 0), ["FLY"], board)) == "illegal"
    assert _kind(compliance_audit.audit_worker_action("w", (0, 0), ["DROP"], board)) == "illegal"
    assert _kind(compliance_audit.audit_worker_action("w", (0, 0), ["PLANT"], board)) == "illegal"


def test_audit_market_orders(board):
    assert compliance_audit.audit_market_order(["HIRE"], {}) == []
    assert compliance_audit.audit_market_order(["SELL", "WHEAT", 5], {"WHEAT": 5}) == []
    assert _kind(compliance_audit.audit_market_order(["SELL", "WHEAT", 5], {})) == "wasted"
    assert _kind(compliance_audit.audit_market_order(["SELL", "WHEAT", 9], {"WHEAT": 5})) == "wasted"
    assert _kind(compliance_audit.audit_market_order(["BUY_PRODUCT"], {})) == "illegal"
    assert _kind(compliance_audit.audit_market_order(["FOO"], {})) == "illegal"
    assert _kind(compliance_audit.audit_market_order("not-a-list", {})) == "illegal"


def test_audit_season_is_clean(rules):
    assert compliance_audit.audit_season(rules) == []


def _kind(findings):
    """First finding's kind, or None if the action was clean."""
    return findings[0]["kind"] if findings else None


# ===========================================================================
# Monte-Carlo runner (pure summarize + graceful engine-absent behaviour)
# ===========================================================================
def test_summarize_populated():
    s = mc.summarize([40000, 52000, 31000, 45000], 35000)
    assert s["n"] == 4
    assert s["mean"] == 42000 and s["min"] == 31000 and s["max"] == 52000
    assert s["spread"] == 21000
    assert s["below_target"] == 1
    assert abs(s["pass_rate"] - 0.75) < 1e-9


def test_summarize_empty():
    s = mc.summarize([], 35000)
    assert s["n"] == 0 and s["mean"] is None and s["pass_rate"] is None


def test_summarize_single():
    s = mc.summarize([35000], 35000)
    assert s["mean"] == 35000 and s["stdev"] == 0 and s["spread"] == 0 and s["pass_rate"] == 1.0


def test_starter_opponent_is_pass():
    starter = mc.make_starter()
    assert starter({}, {}) == {"farmer": [], "hands": [], "market": []}


def test_run_matches_returns_none_without_engine():
    """kaggle_environments is absent here, so no matches can be played."""
    assert mc.load_engine() is None
    assert mc.run_matches(3, "starter") is None


def test_format_report_handles_empty():
    assert "no completed matches" in mc.format_report("starter", mc.summarize([], 35000))


# ===========================================================================
# Behavioural revalidation checks
# ===========================================================================
def test_revalidate_output_structure(rules):
    ok, detail = revalidate.check_output_structure(rules)
    assert ok, detail


def test_revalidate_exception_containment():
    ok, detail = revalidate.check_exception_containment()
    assert ok, detail


def test_revalidate_watering_coverage(rules):
    ok, detail = revalidate.check_watering_coverage(rules)
    assert ok, detail
