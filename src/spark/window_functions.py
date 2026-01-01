"""
Spark Window Functions Module.
Demonstrates: Window specs, ranking, rolling calculations, session windows.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class WindowAnalytics:
    """Analytics class for window function operations on financial data."""
    
    def __init__(self, df: DataFrame, partition_by: str | list[str], order_by: str | list[str]):
        self._df = df
        self._partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by
        self._order_cols = [order_by] if isinstance(order_by, str) else order_by
        self._window = Window.partitionBy(*[F.col(c) for c in self._partition_cols]).orderBy(*[F.col(c) for c in self._order_cols])
        self._unbounded = self._window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    def add_running_total(self, column: str, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_running_total"
        self._df = self._df.withColumn(out, F.sum(F.col(column)).over(self._unbounded))
        return self
    
    def add_running_average(self, column: str, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_running_avg"
        self._df = self._df.withColumn(out, F.avg(F.col(column)).over(self._unbounded))
        return self
    
    def add_rank(self, alias: str = "rank", dense: bool = False) -> "WindowAnalytics":
        self._df = self._df.withColumn(alias, (F.dense_rank() if dense else F.rank()).over(self._window))
        return self
    
    def add_row_number(self, alias: str = "row_number") -> "WindowAnalytics":
        self._df = self._df.withColumn(alias, F.row_number().over(self._window))
        return self
    
    def add_lag(self, column: str, offset: int = 1, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_lag_{offset}"
        self._df = self._df.withColumn(out, F.lag(F.col(column), offset).over(self._window))
        return self
    
    def add_lead(self, column: str, offset: int = 1, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_lead_{offset}"
        self._df = self._df.withColumn(out, F.lead(F.col(column), offset).over(self._window))
        return self
    
    def add_percent_change(self, column: str, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_pct_change"
        prev = F.lag(F.col(column), 1).over(self._window)
        self._df = self._df.withColumn(out, F.when(prev.isNull() | (prev == 0), None).otherwise(((F.col(column) - prev) / prev) * 100))
        return self
    
    def add_moving_average(self, column: str, window_size: int = 7, alias: Optional[str] = None) -> "WindowAnalytics":
        out = alias or f"{column}_ma_{window_size}"
        rolling = self._window.rowsBetween(-(window_size - 1), Window.currentRow)
        self._df = self._df.withColumn(out, F.avg(F.col(column)).over(rolling))
        return self
    
    def add_ntile(self, n: int = 4, alias: str = "quartile") -> "WindowAnalytics":
        self._df = self._df.withColumn(alias, F.ntile(n).over(self._window))
        return self
    
    def get_result(self) -> DataFrame:
        return self._df


def running_total(df: DataFrame, value_column: str, partition_by: str, order_by: str, alias: str = "running_total") -> DataFrame:
    window = Window.partitionBy(F.col(partition_by)).orderBy(F.col(order_by)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn(alias, F.sum(F.col(value_column)).over(window))


def running_average(df: DataFrame, value_column: str, partition_by: str, order_by: str, alias: str = "running_avg") -> DataFrame:
    window = Window.partitionBy(F.col(partition_by)).orderBy(F.col(order_by)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn(alias, F.avg(F.col(value_column)).over(window))


def rank_within_group(df: DataFrame, partition_by: str, order_by: str, alias: str = "rank", descending: bool = True) -> DataFrame:
    order_expr = F.col(order_by).desc() if descending else F.col(order_by).asc()
    window = Window.partitionBy(F.col(partition_by)).orderBy(order_expr)
    return df.withColumn(alias, F.rank().over(window))


def lag_analysis(df: DataFrame, value_column: str, partition_by: str, order_by: str, lag_periods: list[int] = [1, 7, 30]) -> DataFrame:
    window = Window.partitionBy(F.col(partition_by)).orderBy(F.col(order_by))
    for period in lag_periods:
        df = df.withColumn(f"{value_column}_lag_{period}", F.lag(F.col(value_column), period).over(window))
    return df


def session_analysis(df: DataFrame, partition_by: str, timestamp_column: str, session_gap_minutes: int = 30) -> DataFrame:
    window = Window.partitionBy(F.col(partition_by)).orderBy(F.col(timestamp_column))
    df = df.withColumn("_prev_ts", F.lag(F.col(timestamp_column)).over(window))
    df = df.withColumn("_diff_min", (F.unix_timestamp(F.col(timestamp_column)) - F.unix_timestamp(F.col("_prev_ts"))) / 60)
    df = df.withColumn("_new_session", F.when(F.col("_prev_ts").isNull() | (F.col("_diff_min") > session_gap_minutes), 1).otherwise(0))
    session_window = window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = df.withColumn("session_id", F.sum(F.col("_new_session")).over(session_window))
    return df.drop("_prev_ts", "_diff_min", "_new_session")
