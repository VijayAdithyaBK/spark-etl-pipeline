"""
Advanced Python Decorators Module.

Demonstrates:
- Function decorators with and without arguments
- Class decorators
- Decorator factories
- Preserving function metadata with functools.wraps
- Generic type hints for decorators
"""

from __future__ import annotations

import functools
import time
import warnings
from typing import Any, Callable, Optional, ParamSpec, TypeVar, Union, overload

from loguru import logger
from pyspark.sql import DataFrame

# Type variables for generic decorator typing
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def timing_decorator(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to measure and log function execution time.
    
    This is a simple decorator without arguments that wraps any function
    and logs its execution time using loguru.
    
    Args:
        func: The function to wrap.
        
    Returns:
        Wrapped function that logs execution time.
        
    Example:
        >>> @timing_decorator
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> slow_function()
        # Logs: "Function slow_function executed in 1.0012s"
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(f"Function {func.__name__} executed in {elapsed:.4f}s")
    
    return wrapper


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for retrying failed operations with exponential backoff.
    
    This demonstrates a decorator factory pattern - a function that returns
    a decorator based on configuration parameters.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        exponential: If True, use exponential backoff; otherwise, fixed delay.
        exceptions: Tuple of exception types to catch and retry on.
        
    Returns:
        A decorator function.
        
    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=0.5)
        ... def unstable_api_call():
        ...     # May fail intermittently
        ...     pass
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt if exponential else 1)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                            f"{func.__name__}: {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


def validate_dataframe(
    min_rows: Optional[int] = None,
    required_columns: Optional[list[str]] = None,
    non_nullable: Optional[list[str]] = None,
) -> Callable[[Callable[P, DataFrame]], Callable[P, DataFrame]]:
    """
    Decorator factory for validating Spark DataFrame outputs.
    
    Validates that the returned DataFrame meets specified criteria.
    
    Args:
        min_rows: Minimum number of rows expected.
        required_columns: List of columns that must exist.
        non_nullable: Columns that should not contain nulls.
        
    Returns:
        A decorator that validates DataFrame outputs.
        
    Example:
        >>> @validate_dataframe(min_rows=1, required_columns=["id", "amount"])
        ... def load_transactions(spark):
        ...     return spark.read.csv("transactions.csv")
    """
    def decorator(func: Callable[P, DataFrame]) -> Callable[P, DataFrame]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> DataFrame:
            result = func(*args, **kwargs)
            
            if not isinstance(result, DataFrame):
                raise TypeError(
                    f"Function {func.__name__} must return a DataFrame, "
                    f"got {type(result).__name__}"
                )
            
            # Validate minimum rows
            if min_rows is not None:
                row_count = result.count()
                if row_count < min_rows:
                    raise ValueError(
                        f"DataFrame from {func.__name__} has {row_count} rows, "
                        f"expected at least {min_rows}"
                    )
            
            # Validate required columns
            if required_columns:
                actual_columns = set(result.columns)
                missing = set(required_columns) - actual_columns
                if missing:
                    raise ValueError(
                        f"DataFrame from {func.__name__} missing columns: {missing}"
                    )
            
            # Validate non-nullable columns
            if non_nullable:
                from pyspark.sql.functions import col, sum as spark_sum
                
                for column in non_nullable:
                    if column in result.columns:
                        null_count = result.select(
                            spark_sum(col(column).isNull().cast("int"))
                        ).collect()[0][0]
                        
                        if null_count and null_count > 0:
                            raise ValueError(
                                f"Column {column} in {func.__name__} "
                                f"has {null_count} null values"
                            )
            
            return result
        
        return wrapper
    return decorator


def spark_job(
    job_name: Optional[str] = None,
    log_plan: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for wrapping Spark jobs with logging and error handling.
    
    Provides consistent logging, error handling, and optional query plan logging
    for Spark operations.
    
    Args:
        job_name: Custom name for the job; defaults to function name.
        log_plan: If True, log the query execution plan.
        
    Returns:
        A decorator for Spark job functions.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = job_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.info(f"Starting Spark job: {name}")
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                # Log execution plan if requested and result is DataFrame
                if log_plan and isinstance(result, DataFrame):
                    logger.debug(f"Execution plan for {name}:\n{result._jdf.queryExecution()}")
                
                elapsed = time.perf_counter() - start_time
                logger.info(f"Spark job {name} completed in {elapsed:.4f}s")
                
                return result
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"Spark job {name} failed after {elapsed:.4f}s: {e}")
                raise
        
        return wrapper
    return decorator


def cache_result(
    maxsize: int = 128,
    typed: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for caching function results using LRU cache.
    
    This is a wrapper around functools.lru_cache that works with
    functions that have complex arguments by converting them to strings.
    
    Note: Not suitable for functions with unhashable arguments.
    
    Args:
        maxsize: Maximum cache size.
        typed: If True, cache different types separately.
        
    Returns:
        A caching decorator.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return cached_func(*args, **kwargs)
            except TypeError:
                # Fallback for unhashable arguments
                logger.warning(
                    f"Cache miss for {func.__name__} due to unhashable arguments"
                )
                return func(*args, **kwargs)
        
        # Expose cache info and clear methods
        wrapper.cache_info = cached_func.cache_info  # type: ignore
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore
        
        return wrapper
    return decorator


def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    replacement: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to mark functions as deprecated.
    
    Issues a DeprecationWarning when the decorated function is called.
    
    Args:
        reason: Explanation for deprecation.
        version: Version when the function was deprecated.
        replacement: Suggested replacement function.
        
    Returns:
        A decorator that warns on function usage.
        
    Example:
        >>> @deprecated(reason="Use new_function instead", version="2.0")
        ... def old_function():
        ...     pass
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            message = f"Function {func.__name__} is deprecated"
            
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            if replacement:
                message += f". Use {replacement} instead."
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_execution(
    level: str = "INFO",
    include_args: bool = True,
    include_result: bool = False,
    max_arg_length: int = 100,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for detailed execution logging.
    
    Logs function entry, exit, arguments, and optionally the result.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        include_args: Log function arguments.
        include_result: Log function return value.
        max_arg_length: Maximum length for argument string representation.
        
    Returns:
        A logging decorator.
    """
    log_func = getattr(logger, level.lower(), logger.info)
    
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Build entry log message
            entry_msg = f"Entering {func.__name__}"
            
            if include_args:
                args_repr = ", ".join(
                    [repr(a)[:max_arg_length] for a in args] +
                    [f"{k}={repr(v)[:max_arg_length]}" for k, v in kwargs.items()]
                )
                entry_msg += f"({args_repr})"
            
            log_func(entry_msg)
            
            try:
                result = func(*args, **kwargs)
                
                exit_msg = f"Exiting {func.__name__}"
                if include_result:
                    result_repr = repr(result)[:max_arg_length]
                    exit_msg += f" -> {result_repr}"
                
                log_func(exit_msg)
                return result
                
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


# Class decorator example
def singleton(cls: type[T]) -> type[T]:
    """
    Class decorator to make a class a singleton.
    
    Ensures only one instance of the class exists.
    
    Args:
        cls: The class to make a singleton.
        
    Returns:
        The singleton class.
        
    Example:
        >>> @singleton
        ... class DatabaseConnection:
        ...     pass
    """
    instances: dict[type, Any] = {}
    
    @functools.wraps(cls, updated=[])
    class SingletonWrapper(cls):  # type: ignore
        def __new__(wrapper_cls, *args: Any, **kwargs: Any) -> T:
            if cls not in instances:
                instances[cls] = super().__new__(wrapper_cls)
            return instances[cls]
    
    return SingletonWrapper  # type: ignore


# Module testing
if __name__ == "__main__":
    # Test timing decorator
    @timing_decorator
    def test_timing() -> str:
        time.sleep(0.1)
        return "completed"
    
    test_timing()
    
    # Test retry decorator
    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def test_retry() -> str:
        import random
        if random.random() < 0.5:
            raise ValueError("Random failure")
        return "success"
    
    try:
        print(test_retry())
    except ValueError:
        print("All retries failed")
    
    # Test deprecated decorator
    @deprecated(reason="This is old", version="1.0", replacement="new_func")
    def old_func() -> None:
        pass
    
    old_func()
