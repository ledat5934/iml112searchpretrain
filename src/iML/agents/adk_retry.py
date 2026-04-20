import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_transient_adk_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    transient_markers = (
        "503",
        "unavailable",
        "high demand",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "deadline exceeded",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "internal error",
        "service unavailable",
        "connection reset",
    )
    return any(marker in text for marker in transient_markers)


def run_with_adk_retry(
    fn: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 8,
    base_delay_sec: float = 8.0,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_transient_adk_error(exc) or attempt >= max_attempts:
                raise
            delay = base_delay_sec * (2 ** (attempt - 1))
            logger.warning(
                f"{operation_name} failed with transient ADK error on attempt {attempt}/{max_attempts}: {exc}. "
                f"Retrying in {delay:.0f}s."
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
