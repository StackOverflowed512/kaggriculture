from market_model import MarketModel

class Forecaster:
    """
    Projects terminal bank and tracks forecast calibration/diagnosis.
    """
    def __init__(self):
        self.history = []
        self.last_prediction = None
        self.calibration_ratio = 1.0
        
    def project(self, rules: dict, current_money: int, shed_inventory: dict, standing_crops: list, current_market_stocks: dict) -> int:
        """
        Projects terminal bank heuristically based on current money, shed inventory, and standing crops.
        """
        projected = current_money
        
        # 1. Add value of shed inventory
        for item, qty in shed_inventory.items():
            if qty > 0:
                stock = current_market_stocks.get(item, 0)
                avg_p = MarketModel.avg_price(rules, item, stock, qty)
                projected += avg_p * qty
                
        # 2. Add value of standing crops (simplified heuristic)
        for crop in standing_crops:
            item = crop["type"]
            params = rules["crop_params"].get(item)
            if not params:
                continue
                
            expected_units = params.get("yield_no_fertilizer", 1)
            if params["type"] == "repeater":
                expected_units = params["max_fruits"]
                
            stock = current_market_stocks.get(item, 0)
            avg_p = MarketModel.avg_price(rules, item, stock, expected_units)
            projected += avg_p * expected_units
            
        # Apply calibration based on past accuracy
        calibrated_projection = int(projected * self.calibration_ratio)
        self.last_prediction = calibrated_projection
        
        return calibrated_projection

    def observe(self, day: int, realized_money: int, projected_yesterday: int):
        """
        Records realized vs predicted behavior every evening.
        """
        if projected_yesterday > 0:
            ratio = realized_money / projected_yesterday
            # Moving average for calibration
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
