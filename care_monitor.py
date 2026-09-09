import collections

class CareMonitor:
    """
    Tracks crop-care coverage, losses, and production capacity.
    Adjusts acreage based on operational performance using a rolling window
    of daily observations rather than reacting to a single transient event.

    The rolling window (default 3 days) prevents the agent from shrinking
    productive acreage after a single noisy day and then slowly recovering it.
    Capacity adjustments only fire when the *average* missed waterings over
    the window exceed a threshold, which is evidence of a sustained care
    deficit rather than a transient operational hiccup.
    """
    WINDOW_SIZE = 3  # rolling window for evidence-based adjustment

    def __init__(self, initial_capacity: int = 100):
        self.capacity_target = initial_capacity
        self.deaths = 0
        self.coverage_log = []
        self._recent_misses = []  # rolling window of missed_waterings per day
        self._recent_idle = collections.deque(maxlen=24) # 24 turns = 1 day

    def note_idle(self, idle_fraction: float):
        """
        Increases capacity if the crew is persistently idle.

        Uses a sustained-idle check: only expands when the *average* idle
        fraction over the recent window exceeds the threshold, preventing
        expansion from a single idle turn.
        """
        self._recent_idle.append(idle_fraction)
        if len(self._recent_idle) == self._recent_idle.maxlen:
            avg_idle = sum(self._recent_idle) / len(self._recent_idle)
            if avg_idle > 0.1:
                # Expand slowly -- but only when we have evidence of sustained
                # spare capacity, not just one idle turn.
                self.capacity_target = min(100, self.capacity_target + 1)

    def observe_day(self, day: int, missed_waterings: int):
        """
        Observes the daily outcome. Shrinks capacity only when the rolling
        average of missed waterings exceeds a threshold, not on a single event.
        """
        self._recent_misses.append(missed_waterings)
        if len(self._recent_misses) > self.WINDOW_SIZE:
            self._recent_misses.pop(0)

        # Evidence-based adjustment: only shrink when the *average* over the
        # window shows a sustained care deficit.  A single bad day no longer
        # triggers a capacity reduction -- at least half the window must have
        # had misses for the adjustment to fire.
        if len(self._recent_misses) >= self.WINDOW_SIZE:
            days_with_misses = sum(1 for m in self._recent_misses if m > 0)
            if days_with_misses >= (self.WINDOW_SIZE + 1) // 2:
                avg_misses = sum(self._recent_misses) / len(self._recent_misses)
                self.capacity_target = max(
                    1, self.capacity_target - int(avg_misses * 2))

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
