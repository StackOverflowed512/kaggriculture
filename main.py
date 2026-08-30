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

class KaggricultureAgent:
    def __init__(self):
        self.telemetry = Telemetry()
        self.rules_loader = RulesLoader()
        self.rules = None
        self.care_monitor = CareMonitor()
        self.forecaster = Forecaster()

    def __call__(self, obs, config):
        step = obs.step if hasattr(obs, 'step') else obs.get('step', 0)
        
        try:
            # Load rules if not already loaded
            if self.rules is None:
                self.rules = self.rules_loader.load_rules()
                
            # Initialize ActionEmitter for the current turn
            max_orders = self.rules["constants"]["max_market_orders"]
            emitter = ActionEmitter(max_market_orders=max_orders)
            
            # Phase 2 Integration: Update state models
            # In a real game, we'd extract standing crops, shed inventory, and market stocks from `obs`.
            # For this phase, we mock these for demonstration.
            # current_money = obs.reward if hasattr(obs, 'reward') else 0
            
            # End of day (every 24 steps) triggers observation logic
            if step > 0 and step % 24 == 0:
                day = step // 24
                # Mock missed waterings and idle fraction for now
                self.care_monitor.observe_day(self.rules, day, [], 0)
                
                # Mock realized money and yesterday's projection
                realized_money = 1000
                projected_yesterday = self.forecaster.last_prediction or 1000
                self.forecaster.observe(day, realized_money, projected_yesterday)
            
            # Project terminal cash (mocked state)
            # self.forecaster.project(self.rules, current_money, {}, [], {})
            
            # --- Phase 1: Engine Contract Sandbox ---
            emitter.set_farmer_action([]) 
            
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

# Global instance to persist state across turns
_agent_instance = KaggricultureAgent()

def agent(obs, config):
    """
    Public Kaggle entry point. Invokes protected internal execution.
    Never raises an exception.
    """
    return _agent_instance(obs, config)
