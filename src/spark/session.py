"""
Spark Session Factory Module.

Demonstrates:
- Factory pattern for SparkSession creation
- Singleton pattern for session management
- Configuration-driven session setup
- Performance tuning configurations
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession

from src.config.settings import Settings, get_settings
from src.core.metaclasses import SingletonMeta


class SparkSessionFactory(metaclass=SingletonMeta):
    """
    Factory class for creating and managing Spark sessions.
    
    Uses Singleton pattern to ensure only one SparkSession exists
    in the application, preventing resource conflicts.
    
    Example:
        >>> factory = SparkSessionFactory()
        >>> spark = factory.get_or_create()
        >>> df = spark.read.csv("data.csv")
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the Spark session factory.
        
        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._session: Optional[SparkSession] = None
        logger.info("SparkSessionFactory initialized")
    
    def get_or_create(
        self,
        app_name: Optional[str] = None,
        additional_config: Optional[dict[str, str]] = None,
    ) -> SparkSession:
        """
        Get existing or create new SparkSession.
        
        Args:
            app_name: Override application name.
            additional_config: Additional Spark configurations.
            
        Returns:
            Active SparkSession instance.
        """
        if self._session is not None and not self._is_session_stopped():
            logger.debug("Returning existing Spark session")
            return self._session
        
        # Build session with configurations
        builder = SparkSession.builder
        
        # Apply base configurations from settings
        spark_configs = self.settings.get_spark_configs()
        
        # Override app name if provided
        if app_name:
            spark_configs["spark.app.name"] = app_name
        
        # Apply configurations
        for key, value in spark_configs.items():
            builder = builder.config(key, value)
        
        # Apply additional local-mode optimizations
        local_optimizations = self._get_local_optimizations()
        for key, value in local_optimizations.items():
            builder = builder.config(key, value)
        
        # Apply any additional configurations
        if additional_config:
            for key, value in additional_config.items():
                builder = builder.config(key, value)
        
        # Create session
        self._session = builder.getOrCreate()
        
        # Configure logging level
        self._session.sparkContext.setLogLevel("WARN")
        
        logger.info(
            f"Created Spark session: {self._session.sparkContext.appName} "
            f"(version: {self._session.version})"
        )
        
        return self._session
    
    def _is_session_stopped(self) -> bool:
        """Check if the current session is stopped."""
        if self._session is None:
            return True
        
        try:
            # Try to access SparkContext - will fail if stopped
            _ = self._session.sparkContext.applicationId
            return False
        except Exception:
            return True
    
    def _get_local_optimizations(self) -> dict[str, str]:
        """
        Get Spark configurations optimized for local mode.
        
        Returns:
            Dictionary of optimization configurations.
        """
        return {
            # Adaptive Query Execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            
            # UI Configuration
            "spark.ui.showConsoleProgress": "true",
            
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "1g",
            
            # Memory management
            "spark.memory.fraction": "0.8",
            "spark.memory.storageFraction": "0.3",
            
            # Shuffle optimizations (use integer, 'auto' only in Spark 4.x)
            "spark.sql.shuffle.partitions": "200",
            
            # Arrow optimization for Pandas
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
            
            # Column batch size
            "spark.sql.inMemoryColumnarStorage.batchSize": "10000",
        }
    
    def stop(self) -> None:
        """Stop the current Spark session."""
        if self._session is not None:
            logger.info("Stopping Spark session")
            self._session.stop()
            self._session = None
    
    @property
    def session(self) -> Optional[SparkSession]:
        """Get the current session without creating one."""
        return self._session
    
    def get_session_info(self) -> dict[str, str]:
        """
        Get information about the current Spark session.
        
        Returns:
            Dictionary with session information.
        """
        if self._session is None or self._is_session_stopped():
            return {"status": "stopped"}
        
        sc = self._session.sparkContext
        return {
            "status": "active",
            "app_name": sc.appName,
            "app_id": sc.applicationId,
            "master": sc.master,
            "version": self._session.version,
            "default_parallelism": str(sc.defaultParallelism),
        }


# Convenience function for quick access
def get_spark_session(
    app_name: Optional[str] = None,
    config: Optional[dict[str, str]] = None,
) -> SparkSession:
    """
    Get or create a Spark session using the factory.
    
    This is a convenience function for quick access to SparkSession.
    
    Args:
        app_name: Optional application name override.
        config: Optional additional configurations.
        
    Returns:
        Active SparkSession instance.
        
    Example:
        >>> spark = get_spark_session("MyETLJob")
        >>> df = spark.read.parquet("data/")
    """
    factory = SparkSessionFactory()
    return factory.get_or_create(app_name, config)


# Module testing
if __name__ == "__main__":
    # Create session
    spark = get_spark_session("TestApp")
    
    # Get factory info
    factory = SparkSessionFactory()
    info = factory.get_session_info()
    
    print("Session Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create sample DataFrame
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    df = spark.createDataFrame(data, ["name", "id"])
    df.show()
    
    # Stop session
    factory.stop()
