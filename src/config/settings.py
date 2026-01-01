"""
Configuration settings module using Pydantic for validation.
Demonstrates: Type hints, Pydantic models, Singleton pattern, Environment handling.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SparkConfig(BaseModel):
    """Spark-specific configuration settings."""
    
    app_name: str = Field(default="SparkETLPipeline", description="Spark application name")
    master: str = Field(default="local[*]", description="Spark master URL")
    driver_memory: str = Field(default="4g", description="Driver memory allocation")
    executor_memory: str = Field(default="4g", description="Executor memory allocation")
    shuffle_partitions: int = Field(default=200, ge=1, description="Number of shuffle partitions")
    default_parallelism: int = Field(default=8, ge=1, description="Default parallelism level")
    warehouse_dir: str = Field(default="spark-warehouse", description="Spark SQL warehouse directory")
    
    # Performance tuning
    adaptive_execution_enabled: bool = Field(default=True, description="Enable adaptive query execution")
    broadcast_threshold: int = Field(default=10485760, description="Broadcast join threshold in bytes")
    
    @field_validator("driver_memory", "executor_memory")
    @classmethod
    def validate_memory(cls, v: str) -> str:
        """Validate memory string format (e.g., '4g', '512m')."""
        if not v[-1].lower() in ("g", "m", "k"):
            raise ValueError("Memory must end with g, m, or k (e.g., '4g', '512m')")
        try:
            int(v[:-1])
        except ValueError:
            raise ValueError("Memory value must be a number followed by unit")
        return v


class DataConfig(BaseModel):
    """Data paths and format configuration."""
    
    base_path: Path = Field(default=Path("data"), description="Base data directory")
    raw_path: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_path: Path = Field(default=Path("data/processed"), description="Processed data directory")
    schema_path: Path = Field(default=Path("data/schema"), description="Schema definitions directory")
    
    input_format: Literal["csv", "parquet", "json"] = Field(default="csv", description="Input data format")
    output_format: Literal["parquet", "csv", "json"] = Field(default="parquet", description="Output data format")
    
    partition_columns: list[str] = Field(
        default=["year", "month"],
        description="Columns to partition output by"
    )
    
    @field_validator("raw_path", "processed_path", "schema_path", mode="before")
    @classmethod
    def resolve_paths(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        description="Log message format"
    )
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="7 days", description="Log retention period")
    log_file: Optional[Path] = Field(default=Path("logs/etl_pipeline.log"), description="Log file path")


class QualityConfig(BaseModel):
    """Data quality validation configuration."""
    
    null_threshold: float = Field(default=0.05, ge=0, le=1, description="Maximum allowed null ratio")
    duplicate_threshold: float = Field(default=0.01, ge=0, le=1, description="Maximum allowed duplicate ratio")
    enable_strict_mode: bool = Field(default=False, description="Fail on any quality issue")
    generate_reports: bool = Field(default=True, description="Generate quality reports")
    report_format: Literal["html", "json", "both"] = Field(default="both", description="Report output format")


class Settings(BaseModel):
    """
    Main configuration settings class.
    Aggregates all configuration sections into a single validated model.
    """
    
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    
    spark: SparkConfig = Field(default_factory=SparkConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings from environment variables.
        Demonstrates environment-based configuration loading.
        """
        env = os.getenv("ENV", "development")
        debug = os.getenv("DEBUG", "true").lower() == "true"
        
        return cls(
            environment=env,  # type: ignore
            debug=debug,
            spark=SparkConfig(
                app_name=os.getenv("SPARK_APP_NAME", "SparkETLPipeline"),
                master=os.getenv("SPARK_MASTER", "local[*]"),
                driver_memory=os.getenv("SPARK_DRIVER_MEMORY", "4g"),
            ),
        )
    
    def get_spark_configs(self) -> dict[str, str]:
        """
        Get Spark configuration as a dictionary for SparkSession builder.
        
        Returns:
            Dictionary of Spark configuration key-value pairs.
        """
        return {
            "spark.app.name": self.spark.app_name,
            "spark.master": self.spark.master,
            "spark.driver.memory": self.spark.driver_memory,
            "spark.executor.memory": self.spark.executor_memory,
            "spark.sql.shuffle.partitions": str(self.spark.shuffle_partitions),
            "spark.default.parallelism": str(self.spark.default_parallelism),
            "spark.sql.warehouse.dir": self.spark.warehouse_dir,
            "spark.sql.adaptive.enabled": str(self.spark.adaptive_execution_enabled).lower(),
            "spark.sql.autoBroadcastJoinThreshold": str(self.spark.broadcast_threshold),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance (Singleton pattern via lru_cache).
    
    This ensures only one Settings instance is created and reused,
    providing a thread-safe singleton implementation.
    
    Returns:
        The application settings instance.
    """
    return Settings.from_env()


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate settings usage
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Spark configs: {settings.get_spark_configs()}")
