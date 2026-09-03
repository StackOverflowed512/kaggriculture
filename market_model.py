import math

class MarketModel:
    """
    Stateless pure functions for computing market prices and crop values.
    """
    
    @staticmethod
    def _shape(curve_type: str, normal_price: float, halves_after: float,
               current_stock: int, log_decay: float = 0.0315) -> float:
        if current_stock <= 0:
            return normal_price

        if halves_after <= 0:
            # For 'never' halving curves like log for Wheat and Egg
            if curve_type == 'log':
                # Slow decay curve: P = P0 * (1 - c * log(1 + Q)), where the
                # coefficient ``c`` (``log_decay``) is tuned in the rule table,
                # not hard-coded. For Wheat (c=0.0315): 25 * (1 - c * log(2001))
                # ~= 19 at 2000 units.
                return normal_price * (1 - log_decay * math.log(1 + current_stock))
            return normal_price

        ratio = current_stock / halves_after

        if curve_type == 'linear':
            factor = 1 - 0.5 * ratio
        elif curve_type == 'sq':
            factor = 1 - 0.5 * (ratio ** 2)
        elif curve_type == 'sqrt':
            factor = 1 - 0.5 * math.sqrt(ratio)
        elif curve_type == 'log':
            # If log had a halves_after
            factor = 1 - 0.5 * (math.log(1 + current_stock) / math.log(1 + halves_after))
        else:
            factor = 1 - 0.5 * ratio # default linear

        return normal_price * factor

    @staticmethod
    def market_price(rules: dict, item: str, current_stock: int) -> int:
        """
        Computes the live price of an item given the current market inventory,
        enforcing the $1 price floor.
        """
        floor = rules.get("constants", {}).get("price_floor", 1)
        params = rules["market_params"].get(item)
        if not params:
            return int(floor)  # unknown items default to the floor

        # ``hits_floor_after`` (from the rule table) is the stock level at which
        # a curve is defined to bottom out at the floor. Honour it explicitly so
        # every curve type agrees with the documented floor point instead of
        # each shape's own asymptote (e.g. the sqrt curves otherwise never quite
        # reach $1 at their stated stock).
        floor_after = params.get("hits_floor_after", -1)
        if isinstance(floor_after, (int, float)) and floor_after > 0 and current_stock >= floor_after:
            return int(floor)

        raw_price = MarketModel._shape(
            curve_type=params["curve"],
            normal_price=params["normal_price"],
            halves_after=params["halves_after"],
            current_stock=current_stock,
            log_decay=params.get("log_decay", 0.0315),
        )

        return max(int(floor), int(round(raw_price)))

    @staticmethod
    def avg_price(rules: dict, item: str, current_stock: int, qty: int) -> float:
        """
        Computes the average price received when selling 'qty' units.
        Since price drops with each unit sold, this averages the price over the batch.
        """
        if qty <= 0:
            return 0.0
        
        total = 0
        for i in range(qty):
            total += MarketModel.market_price(rules, item, current_stock + i)
            
        return total / qty

    @staticmethod
    def crop_values(rules: dict, current_market_stocks: dict) -> dict:
        """
        Ranks crops dynamically based on expected economic value per tile per turn.
        Formula: (expected_units * expected_price - seed_cost) / cycle_length
        """
        values = {}
        for crop, params in rules["crop_params"].items():
            seed_cost = params["seed"]
            # We assume no fertilizer for baseline calculation
            expected_units = params.get("yield_no_fertilizer", 1) 
            
            if params["type"] == "repeater":
                # For repeaters, it fruits max_fruits times.
                expected_units = params["max_fruits"]
                cycle_length = params["first_fruit"] + (params["max_fruits"] - 1) * params["fruit_every"]
            else:
                cycle_length = params["full_harvest"]
                
            stock = current_market_stocks.get(crop, 0)
            
            # Predict the average price we'd get for this yield
            # Since market prices drop as others sell, we might want to add a safety buffer 
            # to the current stock. For now, we use current stock.
            avg_p = MarketModel.avg_price(rules, crop, stock, expected_units)
            
            expected_revenue = avg_p * expected_units
            profit = expected_revenue - seed_cost
            
            ev_per_turn = profit / cycle_length if cycle_length > 0 else 0
            values[crop] = ev_per_turn
            
        # Return sorted dict by EV descending
        return dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
