"""Phase 3 verification suite (pytest).

Covers the Scheduler and its integration into the agent loop:
    * urgency priority-band task generation and sorting,
    * greedy Manhattan routing / movement step generation,
    * same-day watering for freshly planted seeds,
    * DROP threshold overrides and route pre-emption,
    * planting capacity + cash test,
    * the hour-0 HIRE/SELL routine and endgame liquidation overlay.

Global agent state is isolated with pytest's ``monkeypatch`` fixture.
"""

import copy

import pytest

from rules_loader import RulesLoader
from care_monitor import CareMonitor
from scheduler import (
    Scheduler,
    Task,
    manhattan,
    step_towards,
    nearest_shed_tile,
    SHED_TILES,
)
import main
from main import agent, KaggricultureAgent


# Movement deltas mirroring the documented (x, y) convention, used to walk a
# worker along the path a series of step actions would produce.
MOVE_DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


@pytest.fixture(scope="module")
def rules():
    return RulesLoader("rules_validated.json").load_rules()


@pytest.fixture
def sched():
    return Scheduler()


def _rules_copy(rules):
    """A deep copy so a test can mutate policy without touching the shared dict."""
    return copy.deepcopy(rules)


# ---------------------------------------------------------------------------
# Geometry / Manhattan movement
# ---------------------------------------------------------------------------
def test_manhattan_distance():
    assert manhattan((0, 0), (2, 3)) == 5
    assert manhattan((5, 5), (5, 5)) == 0
    assert manhattan((9, 0), (0, 9)) == 18


def test_step_towards_closes_x_then_y():
    # x-axis is closed first, then the y-axis.
    assert step_towards((0, 0), (2, 3)) == ["EAST"]
    assert step_towards((3, 0), (2, 3)) == ["WEST"]
    assert step_towards((2, 0), (2, 3)) == ["SOUTH"]
    assert step_towards((2, 5), (2, 3)) == ["NORTH"]
    # Already on the tile -> no movement.
    assert step_towards((2, 3), (2, 3)) == []


def test_nearest_shed_tile():
    assert nearest_shed_tile((0, 0)) == (4, 4)
    assert nearest_shed_tile((9, 9)) == (5, 5)
    assert nearest_shed_tile((9, 0)) == (5, 4)
    for tile in SHED_TILES:
        assert nearest_shed_tile(tile) == tile


def test_worker_walks_step_by_step_to_target(rules, sched):
    """A worker at (0,0) tasked at (2,3) emits one move per turn toward it,
    then performs the work action once standing on the tile."""
    pos = (0, 0)
    target = (2, 3)
    task = [Task("WATER", target, 9500)]
    path = []
    for _ in range(10):
        actions = sched.assign_tasks([{"id": "w", "pos": pos, "carried": 0}], task, rules)
        act = actions["w"]
        path.append(act)
        if act and act[0] in MOVE_DELTA:
            dx, dy = MOVE_DELTA[act[0]]
            pos = (pos[0] + dx, pos[1] + dy)
        else:
            break
    # Exactly manhattan-distance move steps, then the WATER action on-tile.
    assert path[-1] == ["WATER"]
    assert len(path) == manhattan((0, 0), target) + 1
    assert pos == target
    assert all(step[0] in MOVE_DELTA for step in path[:-1])


def test_assign_moves_toward_then_works_on_tile(rules, sched):
    tasks = [Task("HARVEST", (5, 5), 9000)]
    off = sched.assign_tasks([{"id": "w", "pos": (5, 3), "carried": 0}], tasks, rules)
    assert off["w"] == ["SOUTH"]  # (5,3) -> (5,5)
    on = sched.assign_tasks([{"id": "w", "pos": (5, 5), "carried": 0}], tasks, rules)
    assert on["w"] == ["HARVEST"]


# ---------------------------------------------------------------------------
# Priority band task generation & sorting
# ---------------------------------------------------------------------------
def test_priority_bands_read_from_rules(rules, sched):
    dr = rules["daily_routines"]
    obs = {
        "step": 48,
        "crops": [
            {"pos": (0, 0), "type": "WHEAT", "misses": 1},  # dying -> 9500
            {"pos": (1, 0), "type": "MELON", "ready": True},  # harvest -> 9000
            {"pos": (2, 0), "type": "WHEAT", "needs_water": True, "in_bonus_window": True},  # 8000
            {"pos": (3, 0), "type": "TOMATO", "needs_water": True, "repeater": True},  # 7000
            {"pos": (4, 0), "type": "CARROT", "needs_water": True},  # normal 5000
        ],
        "weeds": [(9, 9)],  # dig -> 5500
        "empty_tiles": [],
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    by_kind_pos = {(t.kind, t.pos): t for t in tasks}

    assert by_kind_pos[("WATER", (0, 0))].priority == dr["WATER"]["priority_dying"] == 9500
    assert by_kind_pos[("HARVEST", (1, 0))].priority == dr["HARVEST_CROP"]["priority"] == 9000
    assert by_kind_pos[("WATER", (2, 0))].priority == dr["WATER"]["priority_bonus"] == 8000
    assert by_kind_pos[("WATER", (3, 0))].priority == dr["WATER"]["priority_ongoing"] == 7000
    assert by_kind_pos[("DIG", (9, 9))].priority == dr["DIG"]["priority"] == 5500
    assert by_kind_pos[("WATER", (4, 0))].priority == dr["WATER"]["priority_normal"] == 5000


def test_dying_watering_sorts_to_top(rules, sched):
    obs = {
        "step": 24,
        "crops": [
            {"pos": (2, 2), "type": "MELON", "ready": True},  # 9000
            {"pos": (1, 1), "type": "WHEAT", "misses": 1},  # 9500 dying
            {"pos": (3, 3), "type": "TOMATO", "needs_water": True, "repeater": True},  # 7000
        ],
        "weeds": [(5, 5)],
        "empty_tiles": [],
    }
    tasks = sorted(sched.generate_tasks(obs, rules, care_capacity=100), key=Task.sort_key, reverse=True)
    priorities = [t.priority for t in tasks]

    assert tasks[0].kind == "WATER" and tasks[0].priority == 9500
    assert priorities == sorted(priorities, reverse=True)  # strictly non-increasing


def test_harvest_value_orders_within_band(rules, sched):
    """Two ripe crops share band 9000; the richer one sorts first (the
    '9000 + crop value' rule realized as an in-band tie-break)."""
    obs = {
        "step": 24,
        "crops": [
            {"pos": (0, 0), "type": "WHEAT", "ready": True},
            {"pos": (1, 1), "type": "MELON", "ready": True},
        ],
        "weeds": [],
        "empty_tiles": [],
        "market_stocks": {},
    }
    harvests = [t for t in sched.generate_tasks(obs, rules, care_capacity=100) if t.kind == "HARVEST"]
    harvests.sort(key=Task.sort_key, reverse=True)
    assert [t.crop for t in harvests] == ["MELON", "WHEAT"]  # 250 > 25
    assert all(t.priority == 9000 for t in harvests)


def test_no_water_task_when_not_needed(rules, sched):
    obs = {
        "step": 24,
        "crops": [{"pos": (0, 0), "type": "WHEAT", "misses": 0, "needs_water": False}],
        "weeds": [],
        "empty_tiles": [],
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    assert not any(t.kind == "WATER" for t in tasks)


def test_animal_tasks_skipped_when_disabled(rules, sched):
    assert rules["policy"]["ANIMALS_ENABLED"] is False
    obs = {
        "step": 24,
        "crops": [{"pos": (0, 0), "type": "TOMATO", "ready": True, "repeater": True}],
        "weeds": [],
        "empty_tiles": [],
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    assert all(t.kind not in ("FEED", "CARE") for t in tasks)
    assert all(t.priority not in (9100, 8800, 4000) for t in tasks)


# ---------------------------------------------------------------------------
# Same-day watering for planted seeds
# ---------------------------------------------------------------------------
def test_planting_generates_same_day_watering(rules, sched):
    """Planting a seed produces an immediate watering task for that cell."""
    obs = {"step": 0, "crops": [], "weeds": [], "empty_tiles": [(3, 3)], "market_stocks": {}}
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    by = {(t.kind, t.pos): t for t in tasks}

    assert ("PLANT", (3, 3)) in by
    assert ("WATER", (3, 3)) in by
    water = by[("WATER", (3, 3))]
    assert water.same_day is True
    assert water.priority == rules["daily_routines"]["WATER"]["priority_dying"] == 9500


def test_same_day_water_deferred_until_plant_lands(rules, sched):
    """A worker must PLANT the seed, not water bare ground, on the planting turn.
    The paired same-day WATER is deferred while the PLANT is still pending."""
    obs = {"step": 0, "crops": [], "weeds": [], "empty_tiles": [(2, 2)], "market_stocks": {}}
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    actions = sched.assign_tasks([{"id": "w", "pos": (2, 2), "carried": 0}], tasks, rules)
    assert actions["w"][0] == "PLANT"  # not WATER


def test_just_planted_crop_is_dying_priority(rules, sched):
    """Next turn the seed exists carrying one miss -> top-priority watering,
    which is how same-day watering is actually realized."""
    obs = {
        "step": 1,
        "crops": [{"pos": (2, 2), "type": "MELON", "misses": 1}],
        "weeds": [],
        "empty_tiles": [],
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    water = [t for t in tasks if t.kind == "WATER" and t.pos == (2, 2)]
    assert water and water[0].priority == 9500


# ---------------------------------------------------------------------------
# Planting capacity + cash test
# ---------------------------------------------------------------------------
def test_planting_respects_care_capacity(rules, sched):
    """With capacity 3 and 2 crops already planted, only one new tile is planted."""
    obs = {
        "step": 0,
        "crops": [{"pos": (0, 0), "type": "MELON"}, {"pos": (0, 1), "type": "MELON"}],
        "weeds": [],
        "empty_tiles": [(5, 5), (5, 6), (5, 7), (5, 8)],
        "market_stocks": {},
    }
    tasks = sched.generate_tasks(obs, rules, care_capacity=3)
    plant_tiles = [t.pos for t in tasks if t.kind == "PLANT"]
    assert len(plant_tiles) == 1  # 3 capacity - 2 standing = 1 free slot


def test_can_finish_cash_test(rules):
    day = lambda d: d * 24
    # Melon needs 10 days -> cutoff day 19 (season 30 - 10 - 1).
    assert Scheduler._can_finish("MELON", rules, day(19)) is True
    assert Scheduler._can_finish("MELON", rules, day(20)) is False
    # Wheat cutoff day 25; carrot cutoff day 26.
    assert Scheduler._can_finish("WHEAT", rules, day(25)) is True
    assert Scheduler._can_finish("WHEAT", rules, day(26)) is False
    assert Scheduler._can_finish("CARROT", rules, day(26)) is True
    assert Scheduler._can_finish("CARROT", rules, day(27)) is False


def test_no_planting_when_no_crop_can_finish(rules, sched):
    """Late in the season no crop passes the cash test, so nothing is planted."""
    obs = {"step": 29 * 24, "crops": [], "weeds": [], "empty_tiles": [(5, 5)], "market_stocks": {}}
    tasks = sched.generate_tasks(obs, rules, care_capacity=100)
    assert not any(t.kind == "PLANT" for t in tasks)


# ---------------------------------------------------------------------------
# Greedy assignment
# ---------------------------------------------------------------------------
def test_greedy_matches_nearest_worker_per_task(rules, sched):
    workers = [{"id": "a", "pos": (0, 0), "carried": 0}, {"id": "b", "pos": (9, 9), "carried": 0}]
    tasks = [Task("WATER", (1, 0), 9500), Task("WATER", (9, 8), 9500)]
    actions = sched.assign_tasks(workers, tasks, rules)
    assert actions["a"] == ["EAST"]   # a -> (1,0)
    assert actions["b"] == ["NORTH"]  # b -> (9,8)


def test_idle_worker_passes(rules, sched):
    workers = [{"id": "a", "pos": (0, 0), "carried": 0}, {"id": "b", "pos": (9, 9), "carried": 0}]
    tasks = [Task("WATER", (0, 0), 9500)]  # only one task
    actions = sched.assign_tasks(workers, tasks, rules)
    assert actions["a"] == ["WATER"]
    assert actions["b"] == []  # nothing to do -> PASS


# ---------------------------------------------------------------------------
# DROP threshold overrides & route pre-emption
# ---------------------------------------------------------------------------
def test_drop_threshold_values(rules, sched):
    assert sched._drop_threshold(rules, "normal") == 80  # 0.8 * 100
    assert sched._drop_threshold(rules, "overflow") == 50  # half the shed
    assert sched._drop_threshold(rules, "endgame") == 0


def test_drop_preemption_normal(rules, sched):
    """carried + shed usage over 80 -> diverted to the shed, off the bands."""
    tasks = [Task("WATER", (1, 1), 9500)]
    over = sched.assign_tasks(
        [{"id": "h", "pos": (9, 9), "carried": 30}], tasks, rules, shed_usage=60, drop_mode="normal"
    )
    assert over["h"] == ["WEST"]  # 30+60=90 > 80 -> step toward shed tile (5,5)

    under = sched.assign_tasks(
        [{"id": "h", "pos": (1, 1), "carried": 10}], tasks, rules, shed_usage=60, drop_mode="normal"
    )
    assert under["h"] == ["WATER"]  # 10+60=70 <= 80 -> normal band work


def test_drop_preemption_overflow_tightens(rules, sched):
    """A load that survives the 80% threshold is diverted at the 50% one."""
    worker = [{"id": "h", "pos": (4, 4), "carried": 30}]
    tasks = [Task("WATER", (1, 1), 9500)]
    normal = sched.assign_tasks(worker, tasks, rules, shed_usage=25, drop_mode="normal")
    overflow = sched.assign_tasks(worker, tasks, rules, shed_usage=25, drop_mode="overflow")
    assert normal["h"] == ["WEST"]  # 55 <= 80 -> heads to task at (1,1)
    assert overflow["h"] == ["DROP"]  # 55 > 50 -> on a shed tile, drop now


def test_drop_preemption_endgame_forces_any_carrier(rules, sched):
    tasks = [Task("WATER", (1, 1), 9500)]
    carrier = sched.assign_tasks(
        [{"id": "h", "pos": (4, 4), "carried": 1}], tasks, rules, shed_usage=0, drop_mode="endgame"
    )
    assert carrier["h"] == ["DROP"]  # threshold 0: any carried item is banked

    empty = sched.assign_tasks(
        [{"id": "h", "pos": (1, 1), "carried": 0}], tasks, rules, shed_usage=0, drop_mode="endgame"
    )
    assert empty["h"] == ["WATER"]  # empty-handed worker is not diverted


def test_preempted_worker_leaves_band_for_others(rules, sched):
    """The diverted carrier abandons the task; a free worker picks it up."""
    workers = [
        {"id": "carrier", "pos": (4, 4), "carried": 90},
        {"id": "free", "pos": (0, 0), "carried": 0},
    ]
    tasks = [Task("WATER", (0, 1), 9500)]
    actions = sched.assign_tasks(workers, tasks, rules, shed_usage=0, drop_mode="normal")
    assert actions["carrier"] == ["DROP"]  # 90 > 80, already on a shed tile
    assert actions["free"] == ["SOUTH"]  # (0,0) -> (0,1), takes the watering


# ---------------------------------------------------------------------------
# main.py integration (global state isolated with monkeypatch)
# ---------------------------------------------------------------------------
def test_bare_observation_still_passes(monkeypatch):
    """A board-less observation degrades to the safe PASS of earlier phases."""
    monkeypatch.setattr(main._agent_instance, "rules", None)
    result = agent({"step": 24}, {})
    assert result == {"farmer": [], "hands": [], "market": []}


def test_hour_zero_hires_and_sells(rules, monkeypatch):
    """At hour 0 the agent re-hires the crew and sells shed inventory."""
    r = _rules_copy(rules)
    r["policy"]["target_hands"] = 3  # keep HIRE + SELL under the 10-order cap
    monkeypatch.setattr(main._agent_instance, "rules", r)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())
    monkeypatch.setattr(main._agent_instance, "workers", {"farmer": {"pos": (4, 4), "carried": 0}})

    obs = {
        "step": 24,  # hour 0 of day 1
        "crops": [{"pos": (4, 5), "type": "WHEAT", "misses": 1}],
        "weeds": [],
        "empty_tiles": [],
        "shed": {"WHEAT": 12, "MELON": 3},
        "market_stocks": {},
        "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}],
    }
    out = agent(obs, {})
    assert out["market"].count(["HIRE"]) == 3
    assert ["SELL", "WHEAT", 12] in out["market"]
    assert ["SELL", "MELON", 3] in out["market"]
    assert len(out["market"]) <= rules["constants"]["max_market_orders"]
    assert out["farmer"] == ["SOUTH"]  # farmer at (4,4) -> dying crop at (4,5)


def test_endgame_sells_every_turn(rules, monkeypatch):
    """Past the endgame start turn the broker is phoned even outside hour 0,
    and the DROP threshold drops to endgame mode."""
    r = _rules_copy(rules)
    monkeypatch.setattr(main._agent_instance, "rules", r)
    monkeypatch.setattr(main._agent_instance, "scheduler", Scheduler())
    monkeypatch.setattr(main._agent_instance, "care_monitor", CareMonitor())
    monkeypatch.setattr(main._agent_instance, "workers", {"farmer": {"pos": (4, 4), "carried": 0}})

    step = 700  # >= endgame_start_turn (670), hour 700 % 24 = 4 (not hour 0)
    obs = {
        "step": step,
        "crops": [],
        "weeds": [],
        "empty_tiles": [],
        "shed": {"MELON": 5},
        "market_stocks": {},
        "workers": [{"id": "farmer", "pos": (4, 4), "carried": 0}],
    }
    out = agent(obs, {})
    assert ["SELL", "MELON", 5] in out["market"]
    assert ["HIRE"] not in out["market"]  # not hour 0 -> no re-hiring
    assert main._agent_instance.scheduler.drop_mode == "endgame"


def test_wrapper_never_raises(monkeypatch):
    """A malformed rules object is contained and degraded to PASS + counted."""
    monkeypatch.setattr(main._agent_instance, "rules", "broken-not-a-dict")
    before = main._agent_instance.telemetry.get_exception_count()
    result = agent({"step": 5}, {})
    assert result == {"farmer": [], "hands": [], "market": []}
    assert main._agent_instance.telemetry.get_exception_count() == before + 1
