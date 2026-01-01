"""
Delta Lake Loader Module.
Provides Delta Lake integration for ACID transactions and time travel.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame, SparkSession


class DeltaLoader:
    """
    Delta Lake data loader with ACID transactions and time travel.
    
    Features:
    - Write with ACID guarantees
    - Schema evolution support
    - Time travel queries
    - Upsert/merge operations
    - Vacuum for cleanup
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "data/delta"):
        self.spark = spark
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configure Spark for Delta Lake
        self._configure_spark()
    
    def _configure_spark(self):
        """Configure Spark session for Delta Lake."""
        self.spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        self.spark.conf.set(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        logger.info("Delta Lake configuration applied.")
    
    def write_delta(
        self,
        df: DataFrame,
        table_name: str,
        mode: str = "overwrite",
        partition_by: Optional[list[str]] = None
    ) -> str:
        """
        Write DataFrame as Delta table with ACID guarantees.
        
        Args:
            df: DataFrame to write
            table_name: Name of the Delta table
            mode: Write mode ('overwrite', 'append', 'error', 'ignore')
            partition_by: Columns to partition by
        
        Returns:
            Path to the Delta table
        """
        table_path = str(self.base_path / table_name)
        
        logger.info(f"Writing Delta table: {table_name} (mode={mode})")
        
        writer = df.write.format("delta").mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.save(table_path)
        
        row_count = df.count()
        logger.info(f"Delta table '{table_name}' written: {row_count:,} rows")
        
        return table_path
    
    def read_delta(self, table_name: str, version: Optional[int] = None) -> DataFrame:
        """
        Read Delta table with optional time travel.
        
        Args:
            table_name: Name of the Delta table
            version: Optional version number for time travel
        
        Returns:
            DataFrame
        """
        table_path = str(self.base_path / table_name)
        
        reader = self.spark.read.format("delta")
        
        if version is not None:
            reader = reader.option("versionAsOf", version)
            logger.info(f"Reading Delta table '{table_name}' at version {version}")
        else:
            logger.info(f"Reading Delta table '{table_name}' (latest)")
        
        return reader.load(table_path)
    
    def read_at_timestamp(self, table_name: str, timestamp: str) -> DataFrame:
        """
        Read Delta table at a specific timestamp.
        
        Args:
            table_name: Name of the Delta table
            timestamp: Timestamp string (e.g., '2024-01-01 00:00:00')
        
        Returns:
            DataFrame
        """
        table_path = str(self.base_path / table_name)
        
        logger.info(f"Reading Delta table '{table_name}' at timestamp {timestamp}")
        
        return (
            self.spark.read.format("delta")
            .option("timestampAsOf", timestamp)
            .load(table_path)
        )
    
    def upsert(
        self,
        df: DataFrame,
        table_name: str,
        merge_keys: list[str],
        when_matched_update: bool = True
    ) -> dict:
        """
        Upsert (merge) data into Delta table.
        
        Args:
            df: DataFrame with new/updated data
            table_name: Target Delta table name
            merge_keys: Columns to match on
            when_matched_update: Whether to update matching rows
        
        Returns:
            Merge statistics
        """
        from delta.tables import DeltaTable
        
        table_path = str(self.base_path / table_name)
        
        if not Path(table_path).exists():
            logger.info(f"Delta table '{table_name}' doesn't exist. Creating...")
            self.write_delta(df, table_name, mode="overwrite")
            return {"inserted": df.count(), "updated": 0}
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        
        # Build merge condition
        merge_condition = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
        
        logger.info(f"Upserting into Delta table '{table_name}' on keys: {merge_keys}")
        
        merge_builder = (
            delta_table.alias("target")
            .merge(df.alias("source"), merge_condition)
        )
        
        if when_matched_update:
            # Update all columns except merge keys
            update_cols = {c: f"source.{c}" for c in df.columns if c not in merge_keys}
            merge_builder = merge_builder.whenMatchedUpdate(set=update_cols)
        
        merge_builder = merge_builder.whenNotMatchedInsertAll()
        merge_builder.execute()
        
        logger.info(f"Upsert complete for table '{table_name}'")
        
        return {"status": "success", "table": table_name}
    
    def get_history(self, table_name: str, limit: int = 10) -> list[dict]:
        """
        Get version history of a Delta table.
        
        Args:
            table_name: Name of the Delta table
            limit: Max number of versions to return
        
        Returns:
            List of version history entries
        """
        from delta.tables import DeltaTable
        
        table_path = str(self.base_path / table_name)
        
        if not Path(table_path).exists():
            return []
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        history_df = delta_table.history(limit)
        
        return [row.asDict() for row in history_df.collect()]
    
    def vacuum(self, table_name: str, retention_hours: int = 168) -> None:
        """
        Vacuum Delta table to remove old files.
        
        Args:
            table_name: Name of the Delta table
            retention_hours: Hours to retain (default 7 days)
        """
        from delta.tables import DeltaTable
        
        table_path = str(self.base_path / table_name)
        
        if not Path(table_path).exists():
            logger.warning(f"Delta table '{table_name}' doesn't exist.")
            return
        
        logger.info(f"Vacuuming Delta table '{table_name}' with {retention_hours}h retention")
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        delta_table.vacuum(retention_hours)
        
        logger.info("Vacuum complete.")
    
    def get_table_info(self, table_name: str) -> dict:
        """
        Get information about a Delta table.
        
        Args:
            table_name: Name of the Delta table
        
        Returns:
            Table information dictionary
        """
        from delta.tables import DeltaTable
        
        table_path = str(self.base_path / table_name)
        
        if not Path(table_path).exists():
            return {"exists": False, "table_name": table_name}
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        detail = delta_table.detail().collect()[0]
        
        return {
            "exists": True,
            "table_name": table_name,
            "path": table_path,
            "format": detail["format"],
            "num_files": detail["numFiles"],
            "size_bytes": detail["sizeInBytes"],
            "created_at": str(detail["createdAt"]) if detail["createdAt"] else None,
            "last_modified": str(detail["lastModified"]) if detail["lastModified"] else None
        }
    
    def list_tables(self) -> list[str]:
        """List all Delta tables in the base path."""
        tables = []
        
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "_delta_log").exists():
                tables.append(item.name)
        
        return tables
