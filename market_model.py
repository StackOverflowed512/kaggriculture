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
        params = rules.get("market_params", {}).get(item)
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

        Improved model accounts for:
        * Market depletion from our own production (price drops as we sell).
        * A stock buffer estimating opponent-induced price pressure.
        * Watering burden (worker-turns spent watering during the cycle).
        * Opportunity cost (longer cycles = fewer harvests per season).

        Formula:
            EV/turn = (avg_price(yield, stock + buffer) * yield - seed_cost
                       - watering_cost) / cycle_length
        where buffer ≈ yield * 2 (conservative opponent production estimate)
        and watering_cost = watering_events * WATER_COST_PER_TURN.
        """
        WATER_COST_PER_TURN = 2  # estimated worker-turn cost per watering
        STOCK_BUFFER_MULT = 2    # opponent production multiplier

        values = {}
        for crop, params in rules.get("crop_params", {}).items():
            seed_cost = params.get("seed", 0)
            expected_units = params.get("yield_no_fertilizer", 1)

            if params.get("type") == "repeater":
                expected_units = params.get("max_fruits", 1)
                cycle_length = (params.get("first_fruit", 1)
                                + (params.get("max_fruits", 1) - 1)
                                * params.get("fruit_every", 1))
                # Repeaters need ongoing watering every fruit_every days
                watering_events = max(1, cycle_length // max(1, params.get("fruit_every", 1)))
            else:
                cycle_length = params.get("full_harvest", 1)
                # One-time crops need watering during their watering window
                ww = params.get("watering_window", [1, 1])
                watering_events = max(1, ww[1] - ww[0] + 1) if len(ww) >= 2 else 1

            stock = current_market_stocks.get(crop, 0)

            # Add a stock buffer to model the price depression caused by our
            # own production and estimated opponent output.  This prevents the
            # optimizer from recommending a crop whose market is about to crash.
            buffered_stock = stock + expected_units * STOCK_BUFFER_MULT
            avg_p = MarketModel.avg_price(rules, crop, buffered_stock, expected_units)

            expected_revenue = avg_p * expected_units
            watering_cost = watering_events * WATER_COST_PER_TURN
            profit = expected_revenue - seed_cost - watering_cost

            ev_per_turn = profit / cycle_length if cycle_length > 0 else 0
            values[crop] = ev_per_turn

        return dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
