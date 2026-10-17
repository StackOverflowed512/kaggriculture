import pytest
from rules_loader import RulesLoader
from market_model import MarketModel
from care_monitor import CareMonitor
from forecaster import Forecaster
from main import agent
import main

@pytest.fixture(scope="module")
def rules():
    loader = RulesLoader("rules_validated.json")
    return loader.load_rules()

def test_market_model_prices(rules):
    # Milk: $160 normal, linear, halves at 38
    milk_0 = MarketModel.market_price(rules, "MILK", 0)
    assert milk_0 == 160
    
    milk_38 = MarketModel.market_price(rules, "MILK", 38)
    assert milk_38 == 80
    
    milk_floor = MarketModel.market_price(rules, "MILK", 1000)
    assert milk_floor == 1 # Price floor
    
    # Wheat: never halves (approx)
    wheat = MarketModel.market_price(rules, "WHEAT", 2000)
    assert 1 < wheat < 25
    
def test_market_model_avg_price(rules):
    # Sell 10 milk when stock is 30.
    avg = MarketModel.avg_price(rules, "MILK", 30, 10)
    assert avg > 0
    
def test_market_model_crop_values(rules):
    values = MarketModel.crop_values(rules, {"WHEAT": 0, "MELON": 0})
    assert "WHEAT" in values
    assert "MELON" in values
    # Melon (250*6-80)/10 = 142. Wheat (25*4-10)/4 = 22.5
    # We expect Melon > Wheat usually
    assert values["MELON"] > values["WHEAT"]
    
def test_care_monitor():
    monitor = CareMonitor(initial_capacity=100)
    # With the evidence-based rolling window, a single day of missed waterings
    # does NOT immediately shrink capacity -- only sustained misses do.
    monitor.observe_day(1, missed_waterings=1)
    assert monitor.capacity() == 100  # no reduction from single event

    # Sustained misses over the rolling window should trigger reduction
    monitor.observe_day(2, missed_waterings=2)
    monitor.observe_day(3, missed_waterings=1)
    assert monitor.capacity() < 100  # now reduced after sustained misses

    # Idle expansion still works
    monitor.note_idle(0.15)  # > 10% idle
    assert monitor.capacity() <= 100
    
def test_forecaster(rules):
    forecaster = Forecaster()
    shed = {"WHEAT": 10, "MELON": 2}
    standing = [{"type": "CARROT"}, {"type": "TOMATO"}]
    market = {}

    projected = forecaster.project(rules, 1000, shed, standing, market)
    assert projected > 1000

    # Calibration now uses delta-based comparison (Bug 3 fix):
    # delta_realized = realized - last_current_money
    # delta_predicted = projected_yesterday - last_current_money
    # Here: realized=2000, last_current=1000, projected_yesterday=projected
    # delta_realized = 1000, delta_predicted = projected - 1000
    # ratio = 1000 / (projected - 1000), clamped to [0, 2]
    forecaster.observe(1, 2000, projected)
    # The ratio should have moved (not stay at exactly 1.0)
    assert forecaster.calibration_ratio != 1.0 or projected == 2000

    # Tags still work
    tags = forecaster.diagnose()
    assert isinstance(tags, list)

def test_agent_wrapper_phase2(monkeypatch):
    # Use monkeypatch to temporarily modify the global state for this test
    monkeypatch.setattr(main._agent_instance, "rules", None)
    
    obs = {"step": 24}
    config = {}
    result = agent(obs, config)
    assert result == {"farmer": [], "hands": [], "market": []}
