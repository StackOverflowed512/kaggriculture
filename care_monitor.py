class CareMonitor:
    """
    Tracks crop-care coverage, losses, and production capacity.
    Adjusts acreage based on operational performance.
    """
    def __init__(self, initial_capacity: int = 100):
        self.capacity_target = initial_capacity
        self.deaths = 0
        self.coverage_log = []
        
    def note_idle(self, idle_fraction: float):
        """
        Increases capacity if the crew is persistently idle.
        """
        if idle_fraction > 0.1:
            # Expand slowly
            self.capacity_target = min(100, self.capacity_target + 1)
            
    def observe_day(self, rules: dict, day: int, plants_state: list, missed_waterings: int):
        """
        Observes the daily outcome. Shrinks capacity if plants missed water.
        """
        if missed_waterings > 0:
            # Shrink acreage aggressively to save remaining crops
            self.capacity_target = max(1, self.capacity_target - missed_waterings * 2)
            
        self.coverage_log.append({
            "day": day,
            "missed_waterings": missed_waterings,
            "capacity": self.capacity_target
        })
        
    def record_death(self):
        """Records a plant dying from thirst."""
        self.deaths += 1
        
    def capacity(self) -> int:
        """Returns the current target acreage limit."""
        return self.capacity_target
