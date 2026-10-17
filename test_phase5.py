import os
import json
from unittest import mock
import pytest
from scheduler import Scheduler, Task
from telemetry import Telemetry

def test_routing_optimization():
    sched = Scheduler()
    rules = {"policy": {"drop_pressure": 0.8}, "constants": {"shed_size": 100}}
    
    workers = [
        {"id": "w1", "pos": (0, 0), "carried": 0},
        {"id": "w2", "pos": (9, 9), "carried": 0},
    ]
    tasks = [
        Task("WATER", (1, 1), 5000),
        Task("WATER", (8, 8), 5000),
    ]
    actions = sched.assign_tasks(workers, tasks, rules)
    assert "w1" in actions
    assert "w2" in actions
    
    # w1 should go toward (1, 1), so EAST or SOUTH
    assert actions["w1"] in [["EAST"], ["SOUTH"]]
    # w2 should go toward (8, 8), so WEST or NORTH
    assert actions["w2"] in [["WEST"], ["NORTH"]]

def test_telemetry_flush():
    t = Telemetry()
    t.record_exception(10, ValueError("Test"))
    
    # Normal flush
    t.flush_to_disk()
    
    # Find the most recent file
    files = [f for f in os.listdir('.') if f.startswith('telemetry_run_')]
    assert len(files) > 0
    latest = sorted(files)[-1]
    
    with open(latest, 'r') as f:
        data = json.load(f)
    assert data["exception_count"] == 1
    assert "ValueError: Test" in data["exceptions_log"][0]["exception"]
    
    # Test permission error silently ignored
    with mock.patch("builtins.open", side_effect=PermissionError("Mocked Permission Error")):
        # Should not crash
        t.flush_to_disk()
        
    # cleanup
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass
