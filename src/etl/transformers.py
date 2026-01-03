"""
Data Transformers Module.
Demonstrates: Transformer pattern, chain-of-responsibility, business logic encapsulation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataTransformer(ABC):
    """Abstract base class for data transformers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Transformer name for logging."""
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply transformation to DataFrame."""
        pass

    def __call__(self, df: DataFrame) -> DataFrame:
        """Allow transformer to be called as a function."""
        logger.debug(f"Applying transformer: {self.name}")
        return self.transform(df)


class CleansingTransformer(DataTransformer):
    """Clean and normalize data."""

    name = "cleansing"

    def __init__(
        self,
        drop_duplicates: bool = True,
        drop_nulls: bool = False,
        subset: Optional[list[str]] = None,
    ):
        self.drop_duplicates = drop_duplicates
        self.drop_nulls = drop_nulls
        self.subset = subset

    def transform(self, df: DataFrame) -> DataFrame:
        if self.drop_duplicates:
            df = df.dropDuplicates(self.subset)
        if self.drop_nulls:
            df = df.dropna(subset=self.subset)
        return df


class DateTransformer(DataTransformer):
    """Add date-based derived columns."""

    name = "date_features"

    def __init__(self, date_column: str):
        self.date_column = date_column

    def transform(self, df: DataFrame) -> DataFrame:
        col = F.col(self.date_column)
        return (
            df.withColumn("year", F.year(col))
            .withColumn("month", F.month(col))
            .withColumn("day", F.dayofmonth(col))
            .withColumn("hour", F.hour(col))
            .withColumn("day_of_week", F.dayofweek(col))
            .withColumn(
                "is_weekend",
                F.when(F.dayofweek(col).isin([1, 7]), True).otherwise(False),
            )
        )


class AmountTransformer(DataTransformer):
    """Add amount-based features."""

    name = "amount_features"

    def __init__(self, amount_column: str = "amount"):
        self.amount_column = amount_column

    def transform(self, df: DataFrame) -> DataFrame:
        if self.amount_column not in df.columns:
            logger.warning(
                f"Column {self.amount_column} not found, skipping AmountTransformer"
            )
            return df

        col = F.col(self.amount_column)
        return df.withColumn(
            "amount_category",
            F.when(col < 50, "micro")
            .when(col < 100, "small")
            .when(col < 500, "medium")
            .when(col < 1000, "large")
            .otherwise("premium"),
        ).withColumn("amount_log", F.log1p(col))


class FraudFeaturesTransformer(DataTransformer):
    """Add fraud detection features."""

    name = "fraud_features"

    def transform(self, df: DataFrame) -> DataFrame:
        # High amount flag
        if "amount" in df.columns:
            df = df.withColumn(
                "is_high_amount", F.when(F.col("amount") > 1000, True).otherwise(False)
            )

        # Late night transaction flag
        if "hour" in df.columns:
            df = df.withColumn(
                "is_late_night",
                F.when((F.col("hour") >= 23) | (F.col("hour") <= 5), True).otherwise(
                    False
                ),
            )

        return df


class TransformerChain:
    """Chain multiple transformers together."""

    def __init__(self, transformers: Optional[list[DataTransformer]] = None):
        self._transformers: list[DataTransformer] = transformers or []
        self._execution_log: list[str] = []

    def add(self, transformer: DataTransformer) -> "TransformerChain":
        """Add transformer to chain."""
        self._transformers.append(transformer)
        return self

    def add_custom(
        self, func: Callable[[DataFrame], DataFrame], name: str = "custom"
    ) -> "TransformerChain":
        """Add custom transformation function."""

        class CustomTransformer(DataTransformer):
            def __init__(self, fn, n):
                self._fn = fn
                self._name = n

            @property
            def name(self) -> str:
                return self._name

            def transform(self, df: DataFrame) -> DataFrame:
                return self._fn(df)

        self._transformers.append(CustomTransformer(func, name))
        return self

    def execute(self, df: DataFrame) -> DataFrame:
        """Execute all transformers in sequence."""
        self._execution_log = []

        for transformer in self._transformers:
            df = transformer(df)
            self._execution_log.append(transformer.name)

        logger.info(f"Executed {len(self._execution_log)} transformations")
        return df

    @property
    def execution_log(self) -> list[str]:
        return self._execution_log.copy()
