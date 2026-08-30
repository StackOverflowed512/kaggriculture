class ActionEmitter:
    def __init__(self, max_market_orders: int = 10):
        self.max_market_orders = max_market_orders
        self.farmer_action = []
        self.hands_actions = []
        self.market_orders = []

    def set_farmer_action(self, action: list):
        """Sets the farmer's action. Must be a list."""
        if not isinstance(action, list):
            # Silently discard non-list actions as per engine contract
            return
        self.farmer_action = action

    def add_hand_action(self, action: list):
        """Adds a hired hand's action. Must be a list."""
        if not isinstance(action, list):
            return
        self.hands_actions.append(action)

    def add_market_order(self, order: list):
        """Adds a market order. Truncated at emission time."""
        if not isinstance(order, list):
            return
        self.market_orders.append(order)

    def emit(self) -> dict:
        """Returns the engine-compliant action dictionary."""
        # Enforce market order limit
        final_market = self.market_orders[:self.max_market_orders]
        
        return {
            "farmer": self.farmer_action,
            "hands": self.hands_actions,
            "market": final_market
        }
