import json
import os

class RulesLoader:
    MANDATORY_KEYS = [
        "season_length",
        "shed_size",
        "max_market_orders",
        "land_prices",
        "hiring_ladder",
        "missed_waterings_to_die",
        "price_floor",
        "MIN_TERMINAL_TARGET",
        "CORRECTION_THRESHOLD"
    ]
    
    def __init__(self, rules_path: str = "rules_validated.json"):
        self.rules_path = rules_path
        self._rules = None
        self._provenance = None
        
    def load_rules(self) -> dict:
        if self._rules is not None:
            return self._rules
            
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Rules file not found at {self.rules_path}")
            
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse rules JSON: {e}")
                
        # Validate mandatory constants
        constants = data.get("constants", {})
        missing_keys = [key for key in self.MANDATORY_KEYS if key not in constants]
        
        if missing_keys:
            raise ValueError(f"Rules file missing mandatory keys in 'constants': {missing_keys}")
            
        self._rules = data
        self._provenance = self.rules_path
        return self._rules

    def get_provenance(self) -> str:
        return self._provenance
