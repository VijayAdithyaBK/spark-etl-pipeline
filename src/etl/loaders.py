"""
Data Loaders Module.
Demonstrates: Multiple output formats, partitioned writes, write modes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional
from loguru import logger
from pyspark.sql import DataFrame


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, df: DataFrame, path: str | Path, **options) -> None:
        """Write DataFrame to destination."""
        pass


class ParquetLoader(DataLoader):
    """Write data to Parquet format."""
    
    def load(
        self,
        df: DataFrame,
        path: str | Path,
        mode: Literal["overwrite", "append", "error", "ignore"] = "overwrite",
        partition_by: Optional[list[str]] = None,
        compression: str = "snappy",
        **options
    ) -> None:
        """Write DataFrame to Parquet."""
        logger.info(f"Writing Parquet to: {path}")
        
        writer = df.write.format("parquet").mode(mode).option("compression", compression)
        
        if partition_by:
            # Only partition by columns that exist
            valid_partitions = [c for c in partition_by if c in df.columns]
            if valid_partitions:
                writer = writer.partitionBy(*valid_partitions)
        
        for key, value in options.items():
            writer = writer.option(key, value)
        
        writer.save(str(path))
        logger.info(f"Successfully wrote Parquet data")


class CSVLoader(DataLoader):
    """Write data to CSV format."""
    
    def load(
        self,
        df: DataFrame,
        path: str | Path,
        mode: Literal["overwrite", "append", "error", "ignore"] = "overwrite",
        header: bool = True,
        delimiter: str = ",",
        **options
    ) -> None:
        """Write DataFrame to CSV."""
        logger.info(f"Writing CSV to: {path}")
        
        writer = df.write.format("csv").mode(mode) \
                   .option("header", str(header).lower()) \
                   .option("delimiter", delimiter)
        
        for key, value in options.items():
            writer = writer.option(key, value)
        
        writer.save(str(path))
        logger.info(f"Successfully wrote CSV data")


class JSONLoader(DataLoader):
    """Write data to JSON format."""
    
    def load(
        self,
        df: DataFrame,
        path: str | Path,
        mode: Literal["overwrite", "append", "error", "ignore"] = "overwrite",
        **options
    ) -> None:
        """Write DataFrame to JSON."""
        logger.info(f"Writing JSON to: {path}")
        df.write.format("json").mode(mode).save(str(path))


class DeltaLoader(DataLoader):
    """Write data in Delta Lake format (simulation for local)."""
    
    def load(
        self,
        df: DataFrame,
        path: str | Path,
        mode: Literal["overwrite", "append"] = "overwrite",
        partition_by: Optional[list[str]] = None,
        **options
    ) -> None:
        """Write DataFrame to Delta-like format (Parquet with metadata)."""
        logger.info(f"Writing Delta-style data to: {path}")
        
        # Use Parquet as Delta simulation
        path_obj = Path(path)
        data_path = path_obj / "data"
        
        writer = df.write.format("parquet").mode(mode)
        if partition_by:
            valid_partitions = [c for c in partition_by if c in df.columns]
            if valid_partitions:
                writer = writer.partitionBy(*valid_partitions)
        
        writer.save(str(data_path))
        
        # Write simple metadata
        meta = {"version": 1, "row_count": df.count(), "columns": df.columns}
        import json
        path_obj.mkdir(parents=True, exist_ok=True)
        with open(path_obj / "_metadata.json", "w") as f:
            json.dump(meta, f)
        
        logger.info(f"Successfully wrote Delta-style data")
