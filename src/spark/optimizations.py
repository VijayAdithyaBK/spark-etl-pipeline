"""
Spark Optimizations Module.
Demonstrates: Partitioning, caching, broadcast joins, and performance tuning.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel


class PartitionOptimizer:
    """Optimize DataFrame partitioning for better performance."""
    
    @staticmethod
    def analyze_partitions(df: DataFrame) -> dict:
        """Analyze current partition distribution."""
        num_partitions = df.rdd.getNumPartitions()
        partition_sizes = df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
        return {
            "num_partitions": num_partitions,
            "partition_sizes": partition_sizes,
            "min_size": min(partition_sizes) if partition_sizes else 0,
            "max_size": max(partition_sizes) if partition_sizes else 0,
            "avg_size": sum(partition_sizes) / len(partition_sizes) if partition_sizes else 0,
        }
    
    @staticmethod
    def repartition_by_size(df: DataFrame, target_partition_size_mb: int = 128) -> DataFrame:
        """Repartition based on target partition size."""
        row_count = df.count()
        sample_size = min(1000, row_count)
        sample_bytes = df.limit(sample_size).toPandas().memory_usage(deep=True).sum()
        estimated_total_mb = (sample_bytes / sample_size) * row_count / (1024 * 1024)
        target_partitions = max(1, int(estimated_total_mb / target_partition_size_mb))
        logger.info(f"Repartitioning to {target_partitions} partitions (estimated {estimated_total_mb:.1f}MB)")
        return df.repartition(target_partitions)
    
    @staticmethod
    def coalesce_if_needed(df: DataFrame, min_rows_per_partition: int = 10000) -> DataFrame:
        """Coalesce partitions if they are too small."""
        stats = PartitionOptimizer.analyze_partitions(df)
        if stats["avg_size"] < min_rows_per_partition and stats["num_partitions"] > 1:
            target = max(1, int(stats["num_partitions"] * stats["avg_size"] / min_rows_per_partition))
            logger.info(f"Coalescing from {stats['num_partitions']} to {target} partitions")
            return df.coalesce(target)
        return df


class CacheManager:
    """Manage DataFrame caching and persistence."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._cached_dfs: dict[str, DataFrame] = {}
    
    def cache(self, df: DataFrame, name: str, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> DataFrame:
        """Cache DataFrame with specified storage level."""
        cached_df = df.persist(storage_level)
        self._cached_dfs[name] = cached_df
        logger.info(f"Cached DataFrame '{name}' with {storage_level}")
        return cached_df
    
    def uncache(self, name: str) -> None:
        """Unpersist a cached DataFrame."""
        if name in self._cached_dfs:
            self._cached_dfs[name].unpersist()
            del self._cached_dfs[name]
            logger.info(f"Uncached DataFrame '{name}'")
    
    def uncache_all(self) -> None:
        """Unpersist all cached DataFrames."""
        for name in list(self._cached_dfs.keys()):
            self.uncache(name)
    
    def get_cache_info(self) -> dict:
        """Get information about cached DataFrames."""
        return {"cached_dataframes": list(self._cached_dfs.keys()), "count": len(self._cached_dfs)}


class BroadcastManager:
    """Manage broadcast variables for efficient joins."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._broadcasts: dict[str, any] = {}
    
    def broadcast_df(self, df: DataFrame, name: str) -> DataFrame:
        """Return DataFrame marked for broadcast in joins."""
        broadcast_df = F.broadcast(df)
        self._broadcasts[name] = broadcast_df
        logger.info(f"Prepared broadcast DataFrame '{name}'")
        return broadcast_df
    
    def broadcast_join(self, large_df: DataFrame, small_df: DataFrame, join_cols: list[str], join_type: str = "left") -> DataFrame:
        """Perform broadcast join with small DataFrame."""
        logger.info(f"Performing broadcast {join_type} join on {join_cols}")
        return large_df.join(F.broadcast(small_df), join_cols, join_type)
    
    def broadcast_dict(self, data: dict, name: str):
        """Broadcast a dictionary for UDF usage."""
        broadcast_var = self.spark.sparkContext.broadcast(data)
        self._broadcasts[name] = broadcast_var
        logger.info(f"Broadcasted dictionary '{name}'")
        return broadcast_var


def optimize_for_join(df: DataFrame, join_columns: list[str], num_partitions: Optional[int] = None) -> DataFrame:
    """Optimize DataFrame for join operations."""
    if num_partitions:
        return df.repartition(num_partitions, *[F.col(c) for c in join_columns])
    return df.repartition(*[F.col(c) for c in join_columns])


def repartition_by_key(df: DataFrame, key_column: str, num_partitions: int = 200) -> DataFrame:
    """Repartition DataFrame by key column."""
    return df.repartition(num_partitions, F.col(key_column))


def add_salting(df: DataFrame, column: str, num_salts: int = 100) -> DataFrame:
    """Add salt column for skew handling in joins."""
    return df.withColumn("_salt", (F.rand() * num_salts).cast("int"))


def remove_salting(df: DataFrame) -> DataFrame:
    """Remove salt column after join."""
    return df.drop("_salt") if "_salt" in df.columns else df
