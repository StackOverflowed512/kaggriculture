import copy
import json
import os

# Embedded last-resort copy of the rule table. It exists so a *missing* rules
# file degrades to "play on sane defaults" rather than a whole-season PASS (the
# __call__ sandbox catches runtime errors, but not the import/first-load failure
# that a hard FileNotFoundError would cause). This mirrors rules_validated.json;
# build_submission.verify_rules guards the packaged file against drift, and a
# present-but-corrupt file is still surfaced as an error rather than masked.
EMBEDDED_RULES = {
    "constants": {
        "season_length": 720,
        "shed_size": 100,
        "max_market_orders": 10,
        "land_prices": {"NE": 1000, "SW": 2000, "SE": 4000},
        "hiring_ladder": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        "missed_waterings_to_die": 2,
        "price_floor": 1,
        "hours_per_day": 24,
        "MIN_TERMINAL_TARGET": 35000,
        "CORRECTION_THRESHOLD": 45000,
    },
    "crop_params": {
        "WHEAT": {"seed": 10, "full_harvest": 4, "watering_window": [2, 4], "yield_no_fertilizer": 4, "yield_fertilized": 6, "yield_barely_alive": 1, "type": "one-time"},
        "CARROT": {"seed": 20, "full_harvest": 3, "watering_window": [2, 3], "yield_no_fertilizer": 3, "yield_fertilized": 4, "yield_barely_alive": 1, "type": "one-time"},
        "MELON": {"seed": 80, "full_harvest": 10, "watering_window": [6, 10], "yield_no_fertilizer": 6, "yield_fertilized": 6, "yield_barely_alive": 1, "type": "one-time"},
        "TOMATO": {"seed": 50, "first_fruit": 8, "fruit_every": 1, "max_fruits": 4, "type": "repeater"},
        "STRAWBERRY": {"seed": 100, "first_fruit": 10, "fruit_every": 2, "max_fruits": 4, "type": "repeater"},
    },
    "market_params": {
        "WHEAT": {"normal_price": 25, "curve": "log", "halves_after": -1, "hits_floor_after": -1, "log_decay": 0.0315},
        "EGG": {"normal_price": 50, "curve": "log", "halves_after": -1, "hits_floor_after": -1, "log_decay": 0.0315},
        "CARROT": {"normal_price": 35, "curve": "sqrt", "halves_after": 230, "hits_floor_after": 842},
        "TOMATO": {"normal_price": 60, "curve": "sqrt", "halves_after": 135, "hits_floor_after": 529},
        "MELON": {"normal_price": 250, "curve": "sq", "halves_after": 112, "hits_floor_after": 158},
        "MILK": {"normal_price": 160, "curve": "linear", "halves_after": 38, "hits_floor_after": 76},
        "STRAWBERRY": {"normal_price": 120, "curve": "linear", "halves_after": 31, "hits_floor_after": 62},
        "WOOL": {"normal_price": 200, "curve": "sq", "halves_after": 42, "hits_floor_after": 59},
    },
    "animal_params": {
        "GOOSE": {"cost": 300, "home": "COOP", "produces": "EGG", "starts_day": 4, "then_every_days": 1},
        "COW": {"cost": 400, "home": "PASTURE", "produces": "MILK", "starts_day": 8, "then_every_days": 2},
        "SHEEP": {"cost": 500, "home": "PASTURE", "produces": "WOOL", "starts_day": 6, "then_every_days": 3},
    },
    "town_model": {
        "centre_eats_every": 24,
        "shop_opens_every": 3,
        "max_shops": 8,
        "shop_eats_every": 4,
        "shop_amount": 6,
        "specialized_amount": 12,
    },
    "daily_routines": {
        "WATER": {"priority_dying": 9500, "priority_bonus": 8000, "priority_ongoing": 7000, "priority_normal": 5000},
        "HARVEST_ANIMAL": {"priority": 9100},
        "HARVEST_CROP": {"priority": 9000},
        "FEED": {"priority": 8800},
        "PLANT": {"priority": 6000},
        "DIG": {"priority": 5500},
        "PLACE_ANIMAL": {"priority": 4800},
        "BUILD_HOME": {"priority": 4600},
        "CARE": {"priority": 4000},
    },
    "policy": {
        "target_hands": 10,
        "drop_pressure": 0.8,
        "endgame_start_turn": 670,
        "ANIMALS_ENABLED": False,
        "FERTILIZER_ENABLED": False,
        "demand_forecast": False,
    },
    "ops": {
        "walkie_talkie": [
            "NORTH", "SOUTH", "EAST", "WEST",
            "WATER", "HARVEST", "PLANT", "DIG",
            "FEED", "CARE", "DROP", "SELL",
        ]
    },
}


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

    def _resolve_path(self):
        """Find the rules file across the provenance chain, or None.

        The Kaggle image extracts the agent and its rules into a directory that
        is not necessarily the process cwd, so relying on cwd alone (as the old
        loader did) risked a season-long PASS. We look, in order:
            1. ``rules_path`` exactly as given (absolute, or relative to cwd),
            2. next to this module -- the agent's own directory,
            3. explicitly under the current working directory.
        The first existing file wins; None means "found nowhere".
        """
        base = os.path.basename(self.rules_path)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            self.rules_path,
            os.path.join(here, base),
            os.path.join(os.getcwd(), base),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def load_rules(self) -> dict:
        if self._rules is not None:
            return self._rules

        path = self._resolve_path()
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    # A present-but-corrupt file is a real defect: surface it so
                    # the release gates (build_submission / validate_rules) fail
                    # loudly rather than silently running on stale embedded data.
                    raise ValueError(f"Failed to parse rules JSON: {e}")
            provenance = path
        else:
            # No rules file anywhere -> fall back to the embedded copy so the
            # agent still plays instead of PASSing the whole season.
            data = copy.deepcopy(EMBEDDED_RULES)
            provenance = "embedded"

        # Validate mandatory constants (the embedded copy always satisfies this).
        constants = data.get("constants", {})
        missing_keys = [key for key in self.MANDATORY_KEYS if key not in constants]

        if missing_keys:
            raise ValueError(f"Rules file missing mandatory keys in 'constants': {missing_keys}")

        self._rules = data
        self._provenance = provenance
        return self._rules

    def get_provenance(self) -> str:
        return self._provenance
