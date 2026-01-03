"""Spark module for PySpark operations and optimizations."""

from .session import SparkSessionFactory, get_spark_session
from .transformations import (
    DataFrameTransformations,
    clean_column_names,
    add_processing_metadata,
    standardize_dates,
    handle_nulls,
)
from .udfs import (
    register_all_udfs,
    amount_category_udf,
    mask_sensitive_udf,
    extract_hour_udf,
    is_weekend_udf,
    transaction_risk_score_udf,
)
from .window_functions import (
    WindowAnalytics,
    running_total,
    running_average,
    rank_within_group,
    lag_analysis,
    session_analysis,
)
from .optimizations import (
    PartitionOptimizer,
    CacheManager,
    BroadcastManager,
    optimize_for_join,
    repartition_by_key,
)

__all__ = [
    # Session
    "SparkSessionFactory",
    "get_spark_session",
    # Transformations
    "DataFrameTransformations",
    "clean_column_names",
    "add_processing_metadata",
    "standardize_dates",
    "handle_nulls",
    # UDFs
    "register_all_udfs",
    "amount_category_udf",
    "mask_sensitive_udf",
    "extract_hour_udf",
    "is_weekend_udf",
    "transaction_risk_score_udf",
    # Window functions
    "WindowAnalytics",
    "running_total",
    "running_average",
    "rank_within_group",
    "lag_analysis",
    "session_analysis",
    # Optimizations
    "PartitionOptimizer",
    "CacheManager",
    "BroadcastManager",
    "optimize_for_join",
    "repartition_by_key",
]
