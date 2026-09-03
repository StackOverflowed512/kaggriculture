import traceback
import json

# Single, fixed run-log filename. The agent flushes telemetry on every caught
# exception and at the season end; a fixed name overwrites one file in place
# instead of littering the working directory with a timestamped file per flush.
TELEMETRY_FILENAME = "telemetry_run_latest.json"

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
        """Flushes telemetry data to disk safely (overwriting the prior run log)."""
        try:
            data = {
                "exception_count": self.exception_count,
                "exceptions_log": self.exceptions_log
            }
            with open(TELEMETRY_FILENAME, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
