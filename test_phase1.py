import unittest
from rules_loader import RulesLoader
from action_emitter import ActionEmitter
from main import agent, _agent_instance

class TestPhase1(unittest.TestCase):
    def test_rules_loader(self):
        loader = RulesLoader("rules_validated.json")
        rules = loader.load_rules()
        self.assertIn("constants", rules)
        self.assertEqual(rules["constants"]["season_length"], 720)
        self.assertEqual(loader.get_provenance(), "rules_validated.json")

    def test_action_emitter(self):
        emitter = ActionEmitter(max_market_orders=2)
        emitter.set_farmer_action(["WATER"])
        emitter.add_hand_action({"invalid": "dict"})  # Should be ignored
        emitter.add_hand_action(["HARVEST"])
        
        emitter.add_market_order(["SELL", "WHEAT", 5])
        emitter.add_market_order(["SELL", "CARROT", 2])
        emitter.add_market_order(["SELL", "MELON", 1]) # Should be truncated

        out = emitter.emit()
        self.assertEqual(out["farmer"], ["WATER"])
        self.assertEqual(len(out["hands"]), 1)
        self.assertEqual(out["hands"][0], ["HARVEST"])
        self.assertEqual(len(out["market"]), 2)
        self.assertEqual(out["market"][1], ["SELL", "CARROT", 2])

    def test_agent_wrapper_success(self):
        obs = {"step": 1}
        config = {}
        # Clear rules to force load
        _agent_instance.rules = None
        
        result = agent(obs, config)
        self.assertIn("farmer", result)
        self.assertIn("hands", result)
        self.assertIn("market", result)

    def test_agent_wrapper_exception_containment(self):
        # Force an error by providing an invalid obs that causes step fetch to fail if poorly handled, 
        # or we can mock an internal failure.
        obs = {"step": 2}
        config = {}
        
        # Manually break something internal to trigger exception
        original_rules = _agent_instance.rules
        _agent_instance.rules = "broken string" # This will cause a TypeError when agent tries to do rules["constants"]
        
        result = agent(obs, config)
        
        # It should degrade to PASS
        self.assertEqual(result, {"farmer": [], "hands": [], "market": []})
        self.assertGreaterEqual(_agent_instance.telemetry.get_exception_count(), 1)
        
        # Restore for other tests
        _agent_instance.rules = original_rules

if __name__ == '__main__':
    unittest.main()
