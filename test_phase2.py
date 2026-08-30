import unittest
from rules_loader import RulesLoader
from market_model import MarketModel
from care_monitor import CareMonitor
from forecaster import Forecaster
from main import agent, _agent_instance

class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = RulesLoader("rules_validated.json")
        cls.rules = cls.loader.load_rules()

    def test_market_model_prices(self):
        # Milk: $160 normal, linear, halves at 38
        milk_0 = MarketModel.market_price(self.rules, "MILK", 0)
        self.assertEqual(milk_0, 160)
        
        milk_38 = MarketModel.market_price(self.rules, "MILK", 38)
        self.assertEqual(milk_38, 80)
        
        milk_floor = MarketModel.market_price(self.rules, "MILK", 1000)
        self.assertEqual(milk_floor, 1) # Price floor
        
        # Wheat: never halves (approx)
        wheat = MarketModel.market_price(self.rules, "WHEAT", 2000)
        self.assertTrue(1 < wheat < 25) 
        
    def test_market_model_avg_price(self):
        # Sell 10 milk when stock is 30.
        avg = MarketModel.avg_price(self.rules, "MILK", 30, 10)
        self.assertGreater(avg, 0)
        
    def test_market_model_crop_values(self):
        values = MarketModel.crop_values(self.rules, {"WHEAT": 0, "MELON": 0})
        self.assertIn("WHEAT", values)
        self.assertIn("MELON", values)
        # Melon (250*6-80)/10 = 142. Wheat (25*4-10)/4 = 22.5
        # We expect Melon > Wheat usually
        self.assertGreater(values["MELON"], values["WHEAT"])
        
    def test_care_monitor(self):
        monitor = CareMonitor(initial_capacity=100)
        monitor.observe_day(self.rules, 1, [], missed_waterings=1)
        self.assertEqual(monitor.capacity(), 98) # max(1, 100 - 2)
        
        monitor.note_idle(0.15) # > 10% idle
        self.assertEqual(monitor.capacity(), 99) 
        
    def test_forecaster(self):
        forecaster = Forecaster()
        shed = {"WHEAT": 10, "MELON": 2}
        standing = [{"type": "CARROT"}, {"type": "TOMATO"}]
        market = {}
        
        projected = forecaster.project(self.rules, 1000, shed, standing, market)
        self.assertGreater(projected, 1000)
        
        # Test calibration update
        forecaster.observe(1, 2000, 1000)
        self.assertGreater(forecaster.calibration_ratio, 1.0) # 0.8*1.0 + 0.2*(2000/1000) = 1.2
        
        # Tags
        tags = forecaster.diagnose()
        self.assertIn("UNEXPECTED_WINDFALL", tags)

    def test_agent_wrapper_phase2(self):
        _agent_instance.rules = None
        obs = {"step": 24}
        config = {}
        result = agent(obs, config)
        self.assertEqual(result, {"farmer": [], "hands": [], "market": []})
        # ensure no exceptions crashed it
        # Actually it's okay if exceptions were contained, but let's check it didn't throw

if __name__ == '__main__':
    unittest.main()
