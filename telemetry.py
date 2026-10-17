import traceback
import json

# Single, fixed run-log filename. The agent flushes telemetry on every caught
# exception and at the season end; a fixed name overwrites one file in place
# instead of littering the working directory with a timestamped file per flush.
TELEMETRY_FILENAME = "telemetry_run_latest.json"

# Coarse triage buckets keyed by exception class name (#16). Recurring contained
# exceptions silently burn the season, so tagging each one lets the release gate
# and a human reviewer see *what kind* of failure dominates rather than only a
# raw count. Anything unlisted falls through to "other".
_EXCEPTION_CATEGORIES = {
    "TimeoutError": "timeout",
    "MemoryError": "resource",
    "RecursionError": "resource",
    "KeyError": "data",
    "IndexError": "data",
    "TypeError": "data",
    "AttributeError": "data",
    "ValueError": "data",
    "ZeroDivisionError": "arithmetic",
    "ArithmeticError": "arithmetic",
}


def _classify(exception) -> str:
    """Map an exception to a coarse category, walking the class MRO.

    Using the MRO means a subclass (e.g. a custom error deriving from
    ValueError) inherits its base's category instead of collapsing to "other".
    """
    for cls in type(exception).__mro__:
        category = _EXCEPTION_CATEGORIES.get(cls.__name__)
        if category is not None:
            return category
    return "other"


class Telemetry:
    def __init__(self):
        self.exception_count = 0
        self.exceptions_log = []

    def record_exception(self, step: int, exception: Exception):
        """Records an exception with the step (turn hour) it happened.

        Besides the full traceback string, each entry is tagged with the
        exception's class name (``type``) and a coarse ``category`` (#16) so the
        flushed log can be triaged by failure kind without re-parsing tracebacks.
        """
        self.exception_count += 1
        exc_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        self.exceptions_log.append({
            "step": step,
            "type": type(exception).__name__,
            "category": _classify(exception),
            "exception": exc_str,
        })

    def get_exception_count(self) -> int:
        return self.exception_count

    def category_counts(self) -> dict:
        """Aggregate contained exceptions by category, most-common first."""
        counts = {}
        for entry in self.exceptions_log:
            category = entry.get("category", "other")
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def flush_to_disk(self):
        """Flushes telemetry data to disk safely (overwriting the prior run log)."""
        try:
            data = {
                "exception_count": self.exception_count,
                "exceptions_by_category": self.category_counts(),
                "exceptions_log": self.exceptions_log
            }
            with open(TELEMETRY_FILENAME, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
