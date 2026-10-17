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

        # The log-decay coefficient's default lives in the rule table
        # (``market_heuristics.default_log_decay``) so it is tuned in data, not
        # buried as a literal here; a per-item ``log_decay`` still overrides it.
        default_decay = rules.get("market_heuristics", {}).get("default_log_decay", 0.0315)

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
            log_decay=params.get("log_decay", default_decay),
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

        Both heuristics -- the per-watering worker-turn cost and the opponent
        stock-buffer multiplier -- are read from ``market_heuristics`` in the
        rule table so they are tuned in data (single source), not hard-coded
        here where the scheduler and forecaster could each grow their own copy.
        """
        heuristics = rules.get("market_heuristics", {})
        WATER_COST_PER_TURN = heuristics.get("water_cost_per_turn", 2)  # worker-turn cost per watering
        STOCK_BUFFER_MULT = heuristics.get("stock_buffer_mult", 2)      # opponent production multiplier

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

    @staticmethod
    def town_demand(rules: dict, horizon: int, enabled: bool = None) -> dict:
        """Expected town crop consumption over the next ``horizon`` turns (#19).

        DORMANT by default: returns ``{}`` unless ``policy.demand_forecast`` is
        on, keeping unproven strategy-level demand modelling out of the live
        path (the shipped rules keep the flag off).  Its purpose is purely
        architectural discipline: every quantity is read from the validated
        rules artifact's ``town_model`` block -- the single authoritative source
        -- so the strategy layer approximates demand *from the rules we ship*
        rather than re-encoding (and risking drift from) engine facts as magic
        numbers (#19).  Nothing here claims exact-engine fidelity; it is a
        transparent, rules-driven upper bound (all shops open), which is why it
        stays gated until validated against the real engine.
        """
        policy = rules.get("policy", {})
        if enabled is None:
            enabled = policy.get("demand_forecast", False)
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            return {}
        if not enabled or horizon <= 0:
            return {}

        tm = rules.get("town_model", {})
        centre_every = tm.get("centre_eats_every", 0)
        shop_every = tm.get("shop_eats_every", 0)
        shop_amount = tm.get("shop_amount", 0)
        specialized = tm.get("specialized_amount", 0)
        max_shops = tm.get("max_shops", 0)

        # Feedings that land inside the horizon, straight from the cadences in
        # town_model (guard against a zero cadence rather than dividing by it).
        centre_feeds = horizon // centre_every if centre_every else 0
        shop_feeds = horizon // shop_every if shop_every else 0

        # The town centre is the specialized consumer; shops are counted at full
        # capacity for a conservative upper bound.  Both amounts come from the
        # artifact -- no engine constant is written here.
        centre_units = centre_feeds * specialized
        shop_units = max_shops * shop_feeds * shop_amount
        total = centre_units + shop_units
        if total <= 0:
            return {}
        return {
            "horizon": horizon,
            "centre_units": centre_units,
            "shop_units": shop_units,
            "total_units": total,
        }
