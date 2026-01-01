"""Core Python advanced concepts module."""
from .decorators import (
    timing_decorator,
    retry_with_backoff,
    validate_dataframe,
    spark_job,
    cache_result,
    deprecated,
    log_execution,
)
from .generators import (
    TransactionGenerator,
    BatchIterator,
    lazy_file_reader,
    infinite_counter,
    chunked,
)
from .context_managers import (
    SparkSessionContext,
    TempTableContext,
    TimerContext,
    resource_manager,
)
from .metaclasses import (
    SingletonMeta,
    RegistryMeta,
    ValidatedMeta,
)

__all__ = [
    # Decorators
    "timing_decorator",
    "retry_with_backoff",
    "validate_dataframe",
    "spark_job",
    "cache_result",
    "deprecated",
    "log_execution",
    # Generators
    "TransactionGenerator",
    "BatchIterator",
    "lazy_file_reader",
    "infinite_counter",
    "chunked",
    # Context managers
    "SparkSessionContext",
    "TempTableContext",
    "TimerContext",
    "resource_manager",
    # Metaclasses
    "SingletonMeta",
    "RegistryMeta",
    "ValidatedMeta",
]
