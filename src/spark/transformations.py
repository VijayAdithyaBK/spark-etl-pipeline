"""
Spark DataFrame Transformations Module.

Demonstrates:
- DataFrame API operations
- Column transformations
- Schema manipulation
- Chained transformations
- Type-safe DataFrame operations
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def clean_column_names(df: DataFrame, lowercase: bool = True) -> DataFrame:
    """
    Clean column names by removing special characters and optionally lowercasing.
    
    Transforms column names to snake_case format suitable for databases.
    
    Args:
        df: Input DataFrame.
        lowercase: Convert names to lowercase.
        
    Returns:
        DataFrame with cleaned column names.
        
    Example:
        >>> df = spark.createDataFrame([("a",)], ["First Name!"])
        >>> clean_column_names(df).columns
        ['first_name']
    """
    def clean_name(name: str) -> str:
        # Remove special characters, replace spaces with underscores
        cleaned = re.sub(r"[^\w\s]", "", name)
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        return cleaned.lower() if lowercase else cleaned
    
    new_columns = [clean_name(col) for col in df.columns]
    
    # Rename columns
    for old_name, new_name in zip(df.columns, new_columns):
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
    
    logger.debug(f"Cleaned {len(df.columns)} column names")
    return df


def add_processing_metadata(
    df: DataFrame,
    pipeline_name: str = "default",
    include_row_id: bool = True,
) -> DataFrame:
    """
    Add processing metadata columns to DataFrame.
    
    Adds columns for audit trail and lineage tracking.
    
    Args:
        df: Input DataFrame.
        pipeline_name: Name of the processing pipeline.
        include_row_id: Add monotonically increasing row ID.
        
    Returns:
        DataFrame with metadata columns.
    """
    result = df.withColumn(
        "_processing_timestamp",
        F.lit(datetime.now().isoformat())
    ).withColumn(
        "_pipeline_name",
        F.lit(pipeline_name)
    ).withColumn(
        "_spark_partition_id",
        F.spark_partition_id()
    )
    
    if include_row_id:
        result = result.withColumn(
            "_row_id",
            F.monotonically_increasing_id()
        )
    
    logger.debug(f"Added metadata columns. New column count: {len(result.columns)}")
    return result


def standardize_dates(
    df: DataFrame,
    date_columns: list[str],
    input_format: str = "yyyy-MM-dd",
    output_format: Optional[str] = None,
) -> DataFrame:
    """
    Standardize date columns to consistent format.
    
    Parses date strings and converts them to timestamp type or
    reformats them as strings.
    
    Args:
        df: Input DataFrame.
        date_columns: List of date column names.
        input_format: Expected input date format.
        output_format: Output format (None for timestamp type).
        
    Returns:
        DataFrame with standardized dates.
    """
    for col_name in date_columns:
        if col_name not in df.columns:
            logger.warning(f"Date column {col_name} not found in DataFrame")
            continue
        
        # Parse to timestamp
        df = df.withColumn(
            col_name,
            F.to_timestamp(F.col(col_name), input_format)
        )
        
        # Optionally reformat to string
        if output_format:
            df = df.withColumn(
                col_name,
                F.date_format(F.col(col_name), output_format)
            )
    
    logger.debug(f"Standardized {len(date_columns)} date columns")
    return df


def handle_nulls(
    df: DataFrame,
    strategy: str = "drop",
    subset: Optional[list[str]] = None,
    fill_values: Optional[dict[str, any]] = None,
) -> DataFrame:
    """
    Handle null values in DataFrame.
    
    Supports multiple strategies: drop rows, fill with values,
    or fill with column statistics.
    
    Args:
        df: Input DataFrame.
        strategy: One of 'drop', 'fill', 'mean', 'median', 'mode'.
        subset: Columns to consider (None for all).
        fill_values: Dict of column names to fill values (for 'fill' strategy).
        
    Returns:
        DataFrame with nulls handled.
    """
    original_count = df.count()
    
    if strategy == "drop":
        df = df.dropna(subset=subset)
    
    elif strategy == "fill":
        if fill_values:
            df = df.fillna(fill_values)
        else:
            # Fill numeric with 0, strings with empty
            for col_name in (subset or df.columns):
                dtype = dict(df.dtypes).get(col_name)
                if dtype in ("int", "bigint", "double", "float"):
                    df = df.fillna({col_name: 0})
                elif dtype == "string":
                    df = df.fillna({col_name: ""})
    
    elif strategy == "mean":
        for col_name in (subset or df.columns):
            dtype = dict(df.dtypes).get(col_name)
            if dtype in ("int", "bigint", "double", "float"):
                mean_val = df.select(F.mean(F.col(col_name))).collect()[0][0]
                if mean_val is not None:
                    df = df.fillna({col_name: mean_val})
    
    elif strategy == "median":
        for col_name in (subset or df.columns):
            dtype = dict(df.dtypes).get(col_name)
            if dtype in ("int", "bigint", "double", "float"):
                median_val = df.select(
                    F.percentile_approx(F.col(col_name), 0.5)
                ).collect()[0][0]
                if median_val is not None:
                    df = df.fillna({col_name: median_val})
    
    new_count = df.count()
    logger.debug(
        f"Handled nulls with '{strategy}' strategy. "
        f"Rows: {original_count} -> {new_count}"
    )
    
    return df


class DataFrameTransformations:
    """
    Class providing chainable DataFrame transformations.
    
    Demonstrates builder pattern for readable transformation pipelines.
    
    Example:
        >>> transformer = DataFrameTransformations(df)
        >>> result = (transformer
        ...     .clean_columns()
        ...     .filter_by_date("date_col", "2023-01-01", "2023-12-31")
        ...     .add_derived_columns()
        ...     .get_result())
    """
    
    def __init__(self, df: DataFrame):
        """Initialize with source DataFrame."""
        self._df = df
        self._transformation_log: list[str] = []
    
    def clean_columns(self, lowercase: bool = True) -> "DataFrameTransformations":
        """Clean column names."""
        self._df = clean_column_names(self._df, lowercase)
        self._log("clean_columns")
        return self
    
    def add_metadata(
        self,
        pipeline_name: str = "default",
    ) -> "DataFrameTransformations":
        """Add processing metadata."""
        self._df = add_processing_metadata(self._df, pipeline_name)
        self._log(f"add_metadata({pipeline_name})")
        return self
    
    def filter_by_date(
        self,
        column: str,
        start_date: str,
        end_date: str,
    ) -> "DataFrameTransformations":
        """Filter DataFrame by date range."""
        self._df = self._df.filter(
            (F.col(column) >= start_date) & (F.col(column) <= end_date)
        )
        self._log(f"filter_by_date({column}, {start_date}, {end_date})")
        return self
    
    def filter_by_amount(
        self,
        column: str = "amount",
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> "DataFrameTransformations":
        """Filter by amount range."""
        if min_amount is not None:
            self._df = self._df.filter(F.col(column) >= min_amount)
        if max_amount is not None:
            self._df = self._df.filter(F.col(column) <= max_amount)
        self._log(f"filter_by_amount({column}, {min_amount}, {max_amount})")
        return self
    
    def add_derived_columns(self) -> "DataFrameTransformations":
        """Add common derived columns for transaction data."""
        # Add year, month, day from transaction time
        if "transaction_time" in self._df.columns:
            self._df = (
                self._df
                .withColumn("year", F.year(F.col("transaction_time")))
                .withColumn("month", F.month(F.col("transaction_time")))
                .withColumn("day", F.dayofmonth(F.col("transaction_time")))
                .withColumn("hour", F.hour(F.col("transaction_time")))
                .withColumn("day_of_week", F.dayofweek(F.col("transaction_time")))
                .withColumn(
                    "is_weekend",
                    F.when(F.col("day_of_week").isin([1, 7]), True).otherwise(False)
                )
            )
        
        # Add amount category
        if "amount" in self._df.columns:
            self._df = self._df.withColumn(
                "amount_category",
                F.when(F.col("amount") < 50, "small")
                .when(F.col("amount") < 200, "medium")
                .when(F.col("amount") < 1000, "large")
                .otherwise("high_value")
            )
        
        self._log("add_derived_columns")
        return self
    
    def categorize_merchants(self) -> "DataFrameTransformations":
        """Categorize merchants into high-level groups."""
        category_mapping = {
            "grocery": "essentials",
            "gas_station": "essentials",
            "utilities": "essentials",
            "healthcare": "essentials",
            "restaurant": "dining",
            "entertainment": "leisure",
            "travel": "leisure",
            "online_shopping": "shopping",
            "electronics": "shopping",
            "clothing": "shopping",
            "subscription": "recurring",
            "cash_withdrawal": "cash",
        }
        
        if "merchant_category" in self._df.columns:
            # Create mapping expression
            mapping_expr = F.when(F.lit(False), F.lit("other"))  # Start chain
            for cat, group in category_mapping.items():
                mapping_expr = mapping_expr.when(
                    F.col("merchant_category") == cat, group
                )
            mapping_expr = mapping_expr.otherwise("other")
            
            self._df = self._df.withColumn("category_group", mapping_expr)
        
        self._log("categorize_merchants")
        return self
    
    def aggregate_by_customer(self) -> "DataFrameTransformations":
        """Aggregate transaction statistics by customer."""
        if "customer_id" in self._df.columns and "amount" in self._df.columns:
            self._df = self._df.groupBy("customer_id").agg(
                F.count("*").alias("transaction_count"),
                F.sum("amount").alias("total_amount"),
                F.avg("amount").alias("avg_amount"),
                F.min("amount").alias("min_amount"),
                F.max("amount").alias("max_amount"),
                F.stddev("amount").alias("stddev_amount"),
                F.countDistinct("merchant_id").alias("unique_merchants"),
            )
        self._log("aggregate_by_customer")
        return self
    
    def select_columns(self, columns: list[str]) -> "DataFrameTransformations":
        """Select specific columns."""
        self._df = self._df.select(*columns)
        self._log(f"select_columns({columns})")
        return self
    
    def drop_columns(self, columns: list[str]) -> "DataFrameTransformations":
        """Drop specified columns."""
        self._df = self._df.drop(*columns)
        self._log(f"drop_columns({columns})")
        return self
    
    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> "DataFrameTransformations":
        """Rename columns using mapping."""
        for old_name, new_name in mapping.items():
            if old_name in self._df.columns:
                self._df = self._df.withColumnRenamed(old_name, new_name)
        self._log(f"rename_columns({mapping})")
        return self
    
    def cast_types(
        self,
        type_mapping: dict[str, str],
    ) -> "DataFrameTransformations":
        """Cast column types."""
        type_map = {
            "string": StringType(),
            "int": IntegerType(),
            "double": DoubleType(),
            "timestamp": TimestampType(),
        }
        
        for col_name, type_str in type_mapping.items():
            if col_name in self._df.columns and type_str in type_map:
                self._df = self._df.withColumn(
                    col_name,
                    F.col(col_name).cast(type_map[type_str])
                )
        self._log(f"cast_types({type_mapping})")
        return self
    
    def apply(
        self,
        func: Callable[[DataFrame], DataFrame],
    ) -> "DataFrameTransformations":
        """Apply custom transformation function."""
        self._df = func(self._df)
        self._log(f"apply({func.__name__})")
        return self
    
    def cache(self) -> "DataFrameTransformations":
        """Cache the DataFrame."""
        self._df = self._df.cache()
        self._log("cache")
        return self
    
    def repartition(
        self,
        num_partitions: int,
        *cols: str,
    ) -> "DataFrameTransformations":
        """Repartition the DataFrame."""
        if cols:
            self._df = self._df.repartition(num_partitions, *[F.col(c) for c in cols])
        else:
            self._df = self._df.repartition(num_partitions)
        self._log(f"repartition({num_partitions}, {cols})")
        return self
    
    def _log(self, operation: str) -> None:
        """Log transformation for audit trail."""
        self._transformation_log.append(operation)
        logger.debug(f"Applied transformation: {operation}")
    
    def get_result(self) -> DataFrame:
        """Get the transformed DataFrame."""
        logger.info(
            f"Transformation pipeline complete. "
            f"Operations: {len(self._transformation_log)}"
        )
        return self._df
    
    def get_transformation_log(self) -> list[str]:
        """Get list of applied transformations."""
        return self._transformation_log.copy()
    
    @property
    def df(self) -> DataFrame:
        """Direct access to underlying DataFrame."""
        return self._df


# Schema definitions for common data types
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("merchant_id", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("amount", DoubleType(), False),
    StructField("currency", StringType(), True),
    StructField("city", StringType(), True),
    StructField("transaction_time", StringType(), True),
    StructField("is_fraud", IntegerType(), True),
    StructField("card_type", StringType(), True),
    StructField("device_type", StringType(), True),
])


# Module testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("TransformTest").getOrCreate()
    
    # Create sample data
    data = [
        ("TXN_001", "CUST_001", "grocery", 45.50, "2023-06-15 14:30:00"),
        ("TXN_002", "CUST_001", "restaurant", 125.00, "2023-06-16 19:45:00"),
        ("TXN_003", "CUST_002", "electronics", 599.99, "2023-06-15 10:00:00"),
    ]
    
    df = spark.createDataFrame(
        data,
        ["transaction_id", "customer_id", "merchant_category", "amount", "transaction_time"]
    )
    
    # Apply transformations
    transformer = DataFrameTransformations(df)
    result = (
        transformer
        .add_metadata("test_pipeline")
        .add_derived_columns()
        .categorize_merchants()
        .get_result()
    )
    
    result.show(truncate=False)
    print(f"Transformations applied: {transformer.get_transformation_log()}")
    
    spark.stop()
