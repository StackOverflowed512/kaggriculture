import os
import sys

# In a Kaggle environment, all imports would be bundled. 
# For this local workspace, we import them directly.
from rules_loader import RulesLoader
from action_emitter import ActionEmitter
from telemetry import Telemetry

class KaggricultureAgent:
    def __init__(self):
        self.telemetry = Telemetry()
        self.rules_loader = RulesLoader()
        self.rules = None

    def __call__(self, obs, config):
        step = obs.step if hasattr(obs, 'step') else obs.get('step', 0)
        
        try:
            # Load rules if not already loaded
            if self.rules is None:
                self.rules = self.rules_loader.load_rules()
                
            # Initialize ActionEmitter for the current turn
            # max_market_orders is guaranteed by RulesLoader to exist in mandatory keys
            max_orders = self.rules["constants"]["max_market_orders"]
            emitter = ActionEmitter(max_market_orders=max_orders)
            
            # --- Phase 1: Engine Contract Sandbox ---
            # For now, we emit empty actions (PASS) for everything.
            # In Phase 2 & 3, we will integrate MarketModel, CareMonitor, Forecaster, and Scheduler here.
            
            # Example placeholder: the farmer just waits or PASSes
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
