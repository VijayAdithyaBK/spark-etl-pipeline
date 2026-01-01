"""
Main entry point for the Spark ETL Pipeline.
Downloads Kaggle dataset and runs the complete pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.config.settings import get_settings
from src.utils.logging_config import setup_logging
from src.spark.session import get_spark_session
from src.etl.extractors import CSVExtractor
from src.etl.transformers import TransformerChain, CleansingTransformer, AmountTransformer
from src.quality.validators import NullValidator, DuplicateValidator, CompositeValidator
from src.quality.reports import QualityReporter
from src.analytics import FraudAnalytics, AnalyticsReporter
from src.features import FeatureEngineer
from src.core.context_managers import TimerContext


def download_kaggle_dataset() -> Path:
    """Download Credit Card Fraud Detection dataset from Kaggle."""
    import kagglehub
    
    logger.info("Downloading Credit Card Fraud Detection dataset from Kaggle...")
    
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    
    logger.info(f"Dataset downloaded to: {path}")
    return Path(path)


def main():
    """Run the complete ETL pipeline."""
    setup_logging()
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("Spark ETL Pipeline - Financial Transaction Processing")
    logger.info("=" * 60)
    
    # Create directories
    data_dir = Path("data")
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    
    # Download or use existing data
    logger.info("Checking for dataset...")
    
    try:
        kaggle_path = download_kaggle_dataset()
        # Find the CSV file in the downloaded path
        csv_files = list(kaggle_path.glob("*.csv")) + list(kaggle_path.glob("**/*.csv"))
        
        if csv_files:
            input_file = str(csv_files[0])
            logger.info(f"Using dataset: {input_file}")
        else:
            logger.warning("No CSV files found in Kaggle dataset, generating synthetic data")
            from src.core.generators import TransactionGenerator
            gen = TransactionGenerator(num_records=10000, seed=42)
            input_file = str(data_dir / "raw" / "synthetic_transactions.csv")
            gen.to_csv(input_file)
    except Exception as e:
        logger.warning(f"Could not download Kaggle dataset: {e}")
        logger.info("Generating synthetic transaction data instead...")
        
        from src.core.generators import TransactionGenerator
        gen = TransactionGenerator(num_records=10000, seed=42)
        input_file = str(data_dir / "raw" / "synthetic_transactions.csv")
        gen.to_csv(input_file)
    
    # Initialize Spark
    with TimerContext("spark_session"):
        spark = get_spark_session("FraudDetectionETL")
    
    try:
        # EXTRACT
        with TimerContext("extraction"):
            logger.info("Phase 1: EXTRACTION")
            extractor = CSVExtractor(spark)
            df = extractor.extract(input_file)
            
            row_count = df.count()
            logger.info(f"Extracted {row_count} records with {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns[:10]}...")
            df.printSchema()
        
        # TRANSFORM
        with TimerContext("transformation"):
            logger.info("Phase 2: TRANSFORMATION")
            
            chain = TransformerChain()
            chain.add(CleansingTransformer(drop_duplicates=True))
            chain.add(AmountTransformer(amount_column="amt" if "amt" in df.columns else "amount"))
            
            df = chain.execute(df)
            
            logger.info(f"Transformation complete. Columns: {len(df.columns)}")
            logger.info(f"Transformations applied: {chain.execution_log}")
        
        # FEATURE ENGINEERING
        with TimerContext("feature_engineering"):
            logger.info("Phase 2.5: FEATURE ENGINEERING")
            
            feature_engineer = FeatureEngineer(
                df,
                customer_column="cc_num" if "cc_num" in df.columns else "customer_id",
                merchant_column="merchant" if "merchant" in df.columns else "merchant_id",
                amount_column="amt" if "amt" in df.columns else "amount",
                timestamp_column="trans_date_trans_time" if "trans_date_trans_time" in df.columns else "transaction_time",
                fraud_column="is_fraud"
            )
            
            df = feature_engineer.add_all_features()
            feature_summary = feature_engineer.get_feature_summary()
            logger.info(f"Added {feature_summary['total_features_added']} engineered features")
        
        # QUALITY CHECK
        with TimerContext("quality_validation"):
            logger.info("Phase 3: DATA QUALITY VALIDATION")
            
            validators = CompositeValidator([
                NullValidator(columns=df.columns[:5], max_null_ratio=0.1),
                DuplicateValidator(key_columns=[df.columns[0]], max_duplicate_ratio=0.5),
            ])
            
            results = validators.validate_all(df)
            
            if results["all_passed"]:
                logger.info("All quality checks PASSED ✓")
            else:
                logger.warning("Some quality checks FAILED ✗")
            
            # Generate quality report (JSON)
            reporter = QualityReporter(df, "fraud_detection_report")
            reporter.to_json(data_dir / "processed" / "quality_report.json")
            reporter.to_html(data_dir / "processed" / "quality_report.html")
        
        # ANALYTICS
        with TimerContext("analytics"):
            logger.info("Phase 4: FRAUD ANALYTICS")
            
            # Determine column names based on dataset
            amt_col = "amt" if "amt" in df.columns else "amount"
            category_col = "category" if "category" in df.columns else "merchant_category"
            merchant_col = "merchant" if "merchant" in df.columns else "merchant_id"
            timestamp_col = "trans_date_trans_time" if "trans_date_trans_time" in df.columns else "transaction_time"
            
            analytics = FraudAnalytics(
                df,
                fraud_column="is_fraud",
                amount_column=amt_col,
                category_column=category_col,
                merchant_column=merchant_col,
                timestamp_column=timestamp_col
            )
            
            analytics_results = analytics.run_all_analytics()
            
            # Generate analytics reports
            analytics_reporter = AnalyticsReporter(analytics_results, "fraud_analytics")
            analytics_reporter.to_json(data_dir / "processed" / "analytics_report.json")
            analytics_reporter.to_html(data_dir / "processed" / "analytics_report.html")
            
            logger.info("Analytics reports generated ✓")
            
            # Log key insights
            if "executive_summary" in analytics_results:
                summary = analytics_results["executive_summary"]
                logger.info(f"Key Finding: {summary.summary}")
                for action in summary.actionable_items[:3]:
                    logger.info(f"  {action}")
        
        # SHOW RESULTS (Skip Parquet write on Windows without winutils)
        logger.info("Phase 5: RESULTS")
        
        # Show sample output instead of writing to parquet
        logger.info("Sample processed records:")
        df.select(df.columns[:8]).show(10, truncate=False)
        
        # Show aggregate statistics
        from pyspark.sql import functions as F
        
        if "amt" in df.columns or "amount" in df.columns:
            amt_col = "amt" if "amt" in df.columns else "amount"
            stats = df.agg(
                F.count("*").alias("total_transactions"),
                F.sum(amt_col).alias("total_amount"),
                F.avg(amt_col).alias("avg_amount"),
                F.max(amt_col).alias("max_amount"),
                F.min(amt_col).alias("min_amount"),
            ).collect()[0]
            
            logger.info("=" * 60)
            logger.info("TRANSACTION STATISTICS:")
            logger.info(f"  Total Transactions: {stats['total_transactions']:,}")
            logger.info(f"  Total Amount: ${stats['total_amount']:,.2f}")
            logger.info(f"  Average Amount: ${stats['avg_amount']:,.2f}")
            logger.info(f"  Max Amount: ${stats['max_amount']:,.2f}")
            logger.info(f"  Min Amount: ${stats['min_amount']:,.2f}")
        
        if "is_fraud" in df.columns:
            fraud_stats = df.groupBy("is_fraud").count().collect()
            logger.info("\nFRAUD DISTRIBUTION:")
            for row in fraud_stats:
                label = "Fraud" if row["is_fraud"] == 1 else "Legitimate"
                logger.info(f"  {label}: {row['count']:,}")
        
        # SUMMARY
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 60)
        logger.info(f"Records processed: {row_count:,}")
        logger.info(f"Quality reports saved to: {data_dir / 'processed'}")
        logger.info(f"Analytics reports saved to: {data_dir / 'processed'}")
        logger.info("")
        logger.info("View analytics_report.html for detailed fraud insights!")
        logger.info("NOTE: Parquet output skipped (requires winutils.exe on Windows)")
        logger.info("      The data was successfully processed and validated!")
        
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
