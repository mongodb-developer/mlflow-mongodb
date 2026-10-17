"""Small retry helpers."""

import time
from collections.abc import Callable, Sequence
from functools import wraps
from typing import ParamSpec, TypeVar

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def retry_on_exception(
    exception_type: type[Exception],
    *,
    attempts: int,
    backoff_seconds: Sequence[float] = (),
    on_exhausted: Callable[[Exception], Exception] | None = None,
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Retry a call when it raises the specified exception type.

    Other exceptions propagate immediately. After all attempts are exhausted,
    on_exhausted receives the final matching exception and returns the exception
    to raise, chained from the original. Without a factory, the final exception
    is re-raised. If fewer backoff values than retries are supplied, the remaining
    retries happen immediately.
    """
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if len(backoff_seconds) > attempts - 1:
        raise ValueError("backoff_seconds cannot contain more values than retries")
    if any(seconds < 0 for seconds in backoff_seconds):
        raise ValueError("backoff_seconds values must be non-negative")

    def decorator(
        function: Callable[Parameters, Result],
    ) -> Callable[Parameters, Result]:
        @wraps(function)
        def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            attempt = 0
            while True:
                try:
                    return function(*args, **kwargs)
                except exception_type as exc:
                    attempt += 1
                    if attempt == attempts:
                        if on_exhausted is not None:
                            raise on_exhausted(exc) from exc
                        raise
                    if attempt <= len(backoff_seconds):
                        time.sleep(backoff_seconds[attempt - 1])

        return wrapper

    return decorator
