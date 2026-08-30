import traceback

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
