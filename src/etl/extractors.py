"""
Data Extractors Module.
Demonstrates: Multiple data source extraction, schema enforcement, incremental loading.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


class DataExtractor(ABC):
    """Abstract base class for data extractors."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    @abstractmethod
    def extract(self, path: str | Path, **options) -> DataFrame:
        """Extract data from source."""
        pass

    def validate_path(self, path: str | Path) -> Path:
        """Validate that path exists."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return p


class CSVExtractor(DataExtractor):
    """Extract data from CSV files."""

    def extract(
        self,
        path: str | Path,
        schema: Optional[StructType] = None,
        header: bool = True,
        infer_schema: bool = True,
        delimiter: str = ",",
        null_value: str = "",
        **options,
    ) -> DataFrame:
        """Extract data from CSV file(s)."""
        logger.info(f"Extracting CSV from: {path}")

        reader = self.spark.read.format("csv")
        reader = reader.option("header", str(header).lower())
        reader = reader.option("inferSchema", str(infer_schema).lower())
        reader = reader.option("delimiter", delimiter)
        reader = reader.option("nullValue", null_value)

        for key, value in options.items():
            reader = reader.option(key, value)

        if schema:
            reader = reader.schema(schema)

        df = reader.load(str(path))
        logger.info(f"Extracted {df.count()} rows with {len(df.columns)} columns")
        return df


class ParquetExtractor(DataExtractor):
    """Extract data from Parquet files."""

    def extract(
        self, path: str | Path, columns: Optional[list[str]] = None, **options
    ) -> DataFrame:
        """Extract data from Parquet file(s)."""
        logger.info(f"Extracting Parquet from: {path}")

        df = self.spark.read.parquet(str(path))

        if columns:
            df = df.select(*columns)

        logger.info(f"Extracted {df.count()} rows")
        return df


class JSONExtractor(DataExtractor):
    """Extract data from JSON files."""

    def extract(
        self,
        path: str | Path,
        multiline: bool = False,
        schema: Optional[StructType] = None,
        **options,
    ) -> DataFrame:
        """Extract data from JSON file(s)."""
        logger.info(f"Extracting JSON from: {path}")

        reader = self.spark.read.format("json").option(
            "multiline", str(multiline).lower()
        )

        if schema:
            reader = reader.schema(schema)

        return reader.load(str(path))


class IncrementalExtractor(DataExtractor):
    """Extract data incrementally based on watermark."""

    def extract(
        self,
        path: str | Path,
        watermark_column: str,
        last_watermark: Optional[str] = None,
        format_type: str = "parquet",
        **options,
    ) -> DataFrame:
        """Extract data with watermark filtering."""
        logger.info(
            f"Incremental extraction from {path}, watermark column: {watermark_column}"
        )

        if format_type == "parquet":
            df = self.spark.read.parquet(str(path))
        elif format_type == "csv":
            df = self.spark.read.csv(str(path), header=True, inferSchema=True)
        else:
            df = self.spark.read.format(format_type).load(str(path))

        if last_watermark:
            from pyspark.sql import functions as F

            df = df.filter(F.col(watermark_column) > last_watermark)
            logger.info(f"Filtered to records after {last_watermark}")

        return df
