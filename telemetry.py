import traceback
import json
import os
from datetime import datetime

class Telemetry:
    def __init__(self):
        self.exception_count = 0
        self.exceptions_log = []

    def record_exception(self, step: int, exception: Exception):
        """Records an exception with the step (turn hour) it happened."""
        self.exception_count += 1
        exc_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        self.exceptions_log.append({
            "step": step,
            "exception": exc_str
        })

    def get_exception_count(self) -> int:
        return self.exception_count

    def flush_to_disk(self):
        """Flushes telemetry data to disk safely."""
        try:
            filename = f"telemetry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            data = {
                "exception_count": self.exception_count,
                "exceptions_log": self.exceptions_log
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
