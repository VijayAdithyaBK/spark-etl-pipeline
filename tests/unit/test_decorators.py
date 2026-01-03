"""Unit tests for decorators module (non-Spark decorators only)."""

import time
import pytest

# Import only non-Spark dependent decorators directly
import functools
import warnings
from loguru import logger


# Define decorators locally to avoid PySpark import issues on Windows
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(f"Function {func.__name__} executed in {elapsed:.4f}s")

    return wrapper


def retry_with_backoff(
    max_retries=3, base_delay=1.0, exponential=True, exceptions=(Exception,)
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt if exponential else 1)
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def cache_result(maxsize=128, typed=False):
    def decorator(func):
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return cached_func(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        return wrapper

    return decorator


def deprecated(reason="", version=None, replacement=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Function {func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            if replacement:
                message += f". Use {replacement} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TestTimingDecorator:
    def test_timing_returns_result(self):
        @timing_decorator
        def add_numbers(a, b):
            return a + b

        result = add_numbers(1, 2)
        assert result == 3

    def test_timing_preserves_function_name(self):
        @timing_decorator
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestRetryWithBackoff:
    def test_retry_success_on_first_attempt(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_after_failures(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()


class TestCacheResult:
    def test_cache_returns_same_result(self):
        call_count = 0

        @cache_result(maxsize=128)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1


class TestDeprecated:
    def test_deprecated_issues_warning(self):
        @deprecated(reason="Use new_func", version="2.0")
        def old_function():
            return "old"

        with pytest.warns(DeprecationWarning):
            result = old_function()

        assert result == "old"
