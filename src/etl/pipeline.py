"""
ETL Pipeline Orchestration Module.
Demonstrates: Pipeline pattern, configuration-driven ETL, error handling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from src.config.settings import Settings, get_settings
from src.core.decorators import timing_decorator, retry_with_backoff
from src.core.context_managers import TimerContext
from src.etl.extractors import CSVExtractor, ParquetExtractor
from src.etl.transformers import TransformerChain, CleansingTransformer, DateTransformer, AmountTransformer, FraudFeaturesTransformer
from src.etl.loaders import ParquetLoader, CSVLoader


@dataclass
class PipelineConfig:
    """Configuration for ETL pipeline execution."""
    name: str = "default_pipeline"
    input_path: str = "data/raw"
    output_path: str = "data/processed"
    input_format: str = "csv"
    output_format: str = "parquet"
    partition_by: list[str] = field(default_factory=lambda: ["year", "month"])
    enable_quality_checks: bool = True
    cache_intermediate: bool = False


class ETLPipeline:
    """
    Main ETL Pipeline orchestrator.
    Coordinates extraction, transformation, and loading of data.
    """
    
    def __init__(self, spark: SparkSession, config: Optional[PipelineConfig] = None, settings: Optional[Settings] = None):
        self.spark = spark
        self.config = config or PipelineConfig()
        self.settings = settings or get_settings()
        self._metrics: dict = {}
        self._start_time: Optional[datetime] = None
        logger.info(f"Initialized ETL Pipeline: {self.config.name}")
    
    @timing_decorator
    def run(self) -> DataFrame:
        """Execute the complete ETL pipeline."""
        self._start_time = datetime.now()
        logger.info(f"Starting pipeline: {self.config.name}")
        
        try:
            # Extract
            with TimerContext("extraction"):
                df = self._extract()
                self._metrics["extract_count"] = df.count()
            
            # Transform
            with TimerContext("transformation"):
                df = self._transform(df)
                self._metrics["transform_count"] = df.count()
            
            # Quality Checks
            if self.config.enable_quality_checks:
                with TimerContext("quality_checks"):
                    self._run_quality_checks(df)
            
            # Load
            with TimerContext("loading"):
                self._load(df)
            
            self._metrics["success"] = True
            self._metrics["end_time"] = datetime.now().isoformat()
            logger.info(f"Pipeline completed successfully: {self.config.name}")
            return df
            
        except Exception as e:
            self._metrics["success"] = False
            self._metrics["error"] = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _extract(self) -> DataFrame:
        """Extract data from source."""
        logger.info(f"Extracting from: {self.config.input_path}")
        
        if self.config.input_format == "csv":
            extractor = CSVExtractor(self.spark)
            df = extractor.extract(self.config.input_path)
        elif self.config.input_format == "parquet":
            extractor = ParquetExtractor(self.spark)
            df = extractor.extract(self.config.input_path)
        else:
            raise ValueError(f"Unsupported input format: {self.config.input_format}")
        
        logger.info(f"Extracted {df.count()} rows")
        return df
    
    def _transform(self, df: DataFrame) -> DataFrame:
        """Apply transformations."""
        chain = TransformerChain()
        
        # Add standard transformers
        chain.add(CleansingTransformer(drop_duplicates=True))
        
        # Add date transformer if transaction_time column exists
        if "trans_date_trans_time" in df.columns:
            chain.add_custom(
                lambda d: d.withColumnRenamed("trans_date_trans_time", "transaction_time"),
                "rename_date_column"
            )
        
        if "amt" in df.columns:
            chain.add_custom(lambda d: d.withColumnRenamed("amt", "amount"), "rename_amount")
        
        # Add amount transformer
        chain.add(AmountTransformer("amount" if "amount" in df.columns or "amt" in df.columns else "amt"))
        
        # Execute chain
        df = chain.execute(df)
        
        if self.config.cache_intermediate:
            df.cache()
        
        return df
    
    def _run_quality_checks(self, df: DataFrame) -> None:
        """Run data quality checks."""
        row_count = df.count()
        null_counts = {c: df.filter(df[c].isNull()).count() for c in df.columns[:5]}
        
        self._metrics["quality"] = {
            "row_count": row_count,
            "sample_null_counts": null_counts,
        }
        
        logger.info(f"Quality checks passed. Row count: {row_count}")
    
    def _load(self, df: DataFrame) -> None:
        """Load data to destination."""
        logger.info(f"Loading to: {self.config.output_path}")
        
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "parquet":
            loader = ParquetLoader()
            loader.load(df, self.config.output_path, partition_by=self.config.partition_by)
        elif self.config.output_format == "csv":
            loader = CSVLoader()
            loader.load(df, self.config.output_path)
        
        logger.info("Data loaded successfully")
    
    def get_metrics(self) -> dict:
        """Get pipeline execution metrics."""
        return self._metrics.copy()


# Main execution for testing
if __name__ == "__main__":
    from src.spark.session import get_spark_session
    
    spark = get_spark_session("ETLPipelineTest")
    config = PipelineConfig(name="test_pipeline", input_path="data/raw/transactions.csv")
    
    pipeline = ETLPipeline(spark, config)
    
    # Generate test data first
    from src.core.generators import TransactionGenerator
    gen = TransactionGenerator(num_records=1000, seed=42)
    gen.to_csv(Path("data/raw/transactions.csv"))
    
    # Run pipeline
    result = pipeline.run()
    result.show()
    
    print(f"Metrics: {pipeline.get_metrics()}")
