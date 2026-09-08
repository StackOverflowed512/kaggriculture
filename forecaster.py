from market_model import MarketModel

class Forecaster:
    """
    Projects terminal bank and tracks forecast calibration/diagnosis.

    Calibration compares the *delta* in current money between yesterday and
    today against the *delta* the previous projection implied.  This is a
    like-for-like comparison (both are changes in current cash over one day),
    unlike the old approach which compared current cash to a terminal projection
    (two fundamentally different quantities).
    """
    def __init__(self):
        self.history = []
        self.last_prediction = None
        self.last_current_money = None  # snapshot of current money at last projection
        self.calibration_ratio = 1.0

    def project(self, rules: dict, current_money: int, shed_inventory: dict,
                standing_crops: list, current_market_stocks: dict) -> int:
        """Projects terminal bank heuristically based on current money, shed
        inventory, and standing crops."""
        projected = current_money

        # 1. Add value of shed inventory (only sellable products)
        market = rules.get("market_params", {})
        for item, qty in shed_inventory.items():
            if qty > 0 and item in market:
                stock = current_market_stocks.get(item, 0)
                avg_p = MarketModel.avg_price(rules, item, stock, qty)
                projected += avg_p * qty

        # 2. Add value of standing crops (simplified heuristic)
        for crop in standing_crops:
            item = crop.get("type")
            if item is None:
                continue
            params = rules.get("crop_params", {}).get(item)
            if not params:
                continue

            expected_units = params.get("yield_no_fertilizer", 1)
            if params.get("type") == "repeater":
                expected_units = params.get("max_fruits", 1)

            # Only value crops whose product is sellable
            if item not in market:
                continue

            stock = current_market_stocks.get(item, 0)
            avg_p = MarketModel.avg_price(rules, item, stock, expected_units)
            projected += avg_p * expected_units

        # Apply calibration based on past accuracy
        calibrated_projection = int(projected * self.calibration_ratio)
        self.last_prediction = calibrated_projection
        # Snapshot current money so the next observe() can compare deltas
        self.last_current_money = current_money

        return calibrated_projection

    def observe(self, day: int, realized_money: int, projected_yesterday: int):
        """Records realized vs predicted behavior every evening.

        Calibration now compares like-for-like: the change in current money
        since the last projection (Δrealized) against the change the
        projection implied (Δpredicted = projected_yesterday - last_current_money).
        Both quantities are one-day deltas in current cash, so the ratio has
        a sound interpretation as "how much of the projected daily gain
        actually materialised."
        """
        if projected_yesterday > 0 and self.last_current_money is not None:
            delta_realized = realized_money - self.last_current_money
            delta_predicted = projected_yesterday - self.last_current_money
            if delta_predicted != 0:
                ratio = delta_realized / delta_predicted
            else:
                # No gain was predicted; if we also gained nothing, ratio=1.
                # If we gained/lost, that's an unexpected event -- don't
                # penalise the calibration heavily.
                ratio = 1.0 if delta_realized == 0 else 1.0
            # Clamp to [0, 2] so a single bad day can't drive the ratio
            # negative or to an absurd value.
            ratio = max(0.0, min(2.0, ratio))
            self.calibration_ratio = 0.8 * self.calibration_ratio + 0.2 * ratio

        self.history.append({
            "day": day,
            "realized": realized_money,
            "predicted": projected_yesterday,
            "ratio": self.calibration_ratio
        })

    def diagnose(self) -> list:
        """Returns diagnostic tags for logging/telemetry."""
        tags = []
        if self.calibration_ratio < 0.8:
            tags.append("SEVERE_UNDERPERFORMANCE")
        elif self.calibration_ratio > 1.2:
            tags.append("UNEXPECTED_WINDFALL")
        return tags
