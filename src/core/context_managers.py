"""
Advanced Python Context Managers Module.

Demonstrates:
- Context managers with __enter__ and __exit__
- contextlib.contextmanager decorator
- Async context managers
- Resource management patterns
- Nested context managers
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

from loguru import logger
from pyspark.sql import SparkSession

T = TypeVar("T")


class TimerContext:
    """
    Context manager for timing code blocks.

    Demonstrates the basic context manager protocol with
    __enter__ and __exit__ methods.

    Example:
        >>> with TimerContext("data_processing") as timer:
        ...     # Long-running operation
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed:.2f}s")
    """

    def __init__(self, name: str = "operation", log_level: str = "info"):
        """
        Initialize timer context.

        Args:
            name: Name of the operation being timed.
            log_level: Logging level for output.
        """
        self.name = name
        self.log_level = log_level
        self.start_time: float = 0
        self.end_time: float = 0
        self._log_func = getattr(logger, log_level.lower(), logger.info)

    def __enter__(self) -> "TimerContext":
        """Enter context and start timer."""
        self.start_time = time.perf_counter()
        self._log_func(f"Starting: {self.name}")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit context and stop timer."""
        self.end_time = time.perf_counter()

        if exc_type is not None:
            logger.error(f"Failed: {self.name} after {self.elapsed:.4f}s - {exc_val}")
            return False  # Re-raise exception

        self._log_func(f"Completed: {self.name} in {self.elapsed:.4f}s")
        return False

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time


class SparkSessionContext:
    """
    Context manager for Spark session lifecycle.

    Ensures proper creation and cleanup of Spark sessions,
    preventing resource leaks.

    Example:
        >>> with SparkSessionContext("MyApp") as spark:
        ...     df = spark.read.csv("data.csv")
        ...     df.show()
        # Session automatically stopped on exit
    """

    def __init__(
        self,
        app_name: str = "SparkApp",
        master: str = "local[*]",
        config: Optional[dict[str, str]] = None,
        enable_hive: bool = False,
    ):
        """
        Initialize Spark session context.

        Args:
            app_name: Application name.
            master: Spark master URL.
            config: Additional Spark configurations.
            enable_hive: Enable Hive support.
        """
        self.app_name = app_name
        self.master = master
        self.config = config or {}
        self.enable_hive = enable_hive
        self._session: Optional[SparkSession] = None

    def __enter__(self) -> SparkSession:
        """Create and return Spark session."""
        logger.info(f"Creating Spark session: {self.app_name}")

        builder = SparkSession.builder.appName(self.app_name).master(self.master)

        # Apply configuration
        for key, value in self.config.items():
            builder = builder.config(key, value)

        # Default configurations for local mode
        builder = builder.config("spark.sql.adaptive.enabled", "true")
        builder = builder.config("spark.driver.memory", "4g")

        if self.enable_hive:
            builder = builder.enableHiveSupport()

        self._session = builder.getOrCreate()

        # Set log level
        self._session.sparkContext.setLogLevel("WARN")

        logger.info(
            f"Spark session created: {self._session.version} "
            f"(master: {self.master})"
        )

        return self._session

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Stop Spark session on exit."""
        if self._session:
            logger.info("Stopping Spark session")
            self._session.stop()
            self._session = None

        if exc_type:
            logger.error(f"Spark session exited with error: {exc_val}")

        return False


class TempTableContext:
    """
    Context manager for temporary Spark SQL tables.

    Automatically registers a DataFrame as a temp table and
    unregisters it when exiting the context.

    Example:
        >>> with TempTableContext(df, "transactions") as table_name:
        ...     result = spark.sql(f"SELECT * FROM {table_name}")
    """

    def __init__(
        self,
        spark: SparkSession,
        df: Any,  # DataFrame
        table_name: str,
        cache: bool = False,
    ):
        """
        Initialize temp table context.

        Args:
            spark: Active Spark session.
            df: DataFrame to register.
            table_name: Name for the temporary table.
            cache: Whether to cache the table.
        """
        self.spark = spark
        self.df = df
        self.table_name = table_name
        self.cache = cache

    def __enter__(self) -> str:
        """Register DataFrame as temp table."""
        if self.cache:
            self.df.cache()

        self.df.createOrReplaceTempView(self.table_name)
        logger.debug(f"Created temp table: {self.table_name}")

        return self.table_name

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Drop temp table on exit."""
        try:
            self.spark.catalog.dropTempView(self.table_name)
            logger.debug(f"Dropped temp table: {self.table_name}")
        except Exception as e:
            logger.warning(f"Failed to drop temp table {self.table_name}: {e}")

        if self.cache:
            self.df.unpersist()

        return False


@contextmanager
def resource_manager(
    acquire: Callable[[], T],
    release: Callable[[T], None],
    name: str = "resource",
) -> Generator[T, None, None]:
    """
    Generic context manager factory for resource management.

    Demonstrates the contextlib.contextmanager decorator pattern
    for creating context managers from generator functions.

    Args:
        acquire: Function to acquire the resource.
        release: Function to release the resource.
        name: Resource name for logging.

    Yields:
        The acquired resource.

    Example:
        >>> def open_connection():
        ...     return Database.connect()
        >>> def close_connection(conn):
        ...     conn.close()
        >>> with resource_manager(open_connection, close_connection) as conn:
        ...     conn.execute("SELECT 1")
    """
    resource = None
    try:
        logger.debug(f"Acquiring resource: {name}")
        resource = acquire()
        yield resource
    finally:
        if resource is not None:
            logger.debug(f"Releasing resource: {name}")
            try:
                release(resource)
            except Exception as e:
                logger.error(f"Error releasing {name}: {e}")


@contextmanager
def spark_checkpoint_context(
    spark: SparkSession,
    checkpoint_dir: str = "/tmp/spark_checkpoints",
) -> Generator[None, None, None]:
    """
    Context manager for Spark checkpoint directory.

    Sets up and cleans up checkpoint directory for operations
    that require checkpointing (e.g., breaking lineage).

    Args:
        spark: Active Spark session.
        checkpoint_dir: Directory for checkpoints.

    Yields:
        None (checkpoint is configured in Spark context).
    """
    from pathlib import Path
    import shutil

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    logger.debug(f"Set checkpoint directory: {checkpoint_dir}")

    try:
        yield
    finally:
        # Clean up checkpoint files
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.debug(f"Cleaned up checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean checkpoint dir: {e}")


@contextmanager
def suppress_spark_logs() -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress Spark logging.

    Useful for clean output during operations that generate
    excessive Spark logs.

    Yields:
        None (logs are suppressed).
    """
    import logging

    # Get loggers
    spark_logger = logging.getLogger("py4j")
    spark_java_logger = logging.getLogger("pyspark")

    # Store original levels
    original_py4j = spark_logger.level
    original_pyspark = spark_java_logger.level

    try:
        spark_logger.setLevel(logging.ERROR)
        spark_java_logger.setLevel(logging.ERROR)
        yield
    finally:
        spark_logger.setLevel(original_py4j)
        spark_java_logger.setLevel(original_pyspark)


class TransactionContext:
    """
    Context manager for transactional operations.

    Demonstrates rollback capability for operations that
    need atomicity guarantees.
    """

    def __init__(self, name: str = "transaction"):
        """Initialize transaction context."""
        self.name = name
        self._operations: list[tuple[Callable, Callable]] = []
        self._committed = False

    def add_operation(
        self,
        execute: Callable[[], Any],
        rollback: Callable[[], Any],
    ) -> None:
        """Add an operation with its rollback function."""
        self._operations.append((execute, rollback))

    def __enter__(self) -> "TransactionContext":
        """Enter transaction context."""
        logger.info(f"Starting transaction: {self.name}")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Commit or rollback based on exception."""
        if exc_type is not None and not self._committed:
            logger.warning(f"Rolling back transaction: {self.name}")
            self._rollback()
            return False

        if not self._committed:
            self._commit()

        return False

    def _commit(self) -> None:
        """Execute all operations."""
        for execute, _ in self._operations:
            execute()
        self._committed = True
        logger.info(f"Committed transaction: {self.name}")

    def _rollback(self) -> None:
        """Execute rollback for completed operations."""
        for _, rollback in reversed(self._operations):
            try:
                rollback()
            except Exception as e:
                logger.error(f"Rollback error: {e}")


# Module testing
if __name__ == "__main__":
    # Test TimerContext
    with TimerContext("test_operation") as timer:
        time.sleep(0.1)
    print(f"Timer elapsed: {timer.elapsed:.4f}s")

    # Test resource_manager
    def acquire():
        print("Acquiring resource")
        return {"data": "value"}

    def release(r):
        print(f"Releasing resource: {r}")

    with resource_manager(acquire, release, "test_resource") as res:
        print(f"Using resource: {res}")
