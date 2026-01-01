"""
Feature Engineering Module.
Creates derived features for fraud detection modeling.
"""

from __future__ import annotations
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class FeatureEngineer:
    """
    Feature engineering for fraud detection.
    
    Creates velocity, amount, time, and risk-based features
    to improve fraud detection model performance.
    """
    
    def __init__(
        self,
        df: DataFrame,
        customer_column: str = "cc_num",
        merchant_column: str = "merchant",
        amount_column: str = "amt",
        timestamp_column: str = "trans_date_trans_time",
        fraud_column: str = "is_fraud"
    ):
        self.df = df
        self.customer_col = customer_column
        self.merchant_col = merchant_column
        self.amount_col = amount_column
        self.timestamp_col = timestamp_column
        self.fraud_col = fraud_column
        self._feature_count = 0
    
    def add_all_features(self) -> DataFrame:
        """Add all engineered features."""
        logger.info("Starting feature engineering...")
        
        df = self.df
        df = self.add_time_features(df)
        df = self.add_amount_features(df)
        df = self.add_velocity_features(df)
        df = self.add_merchant_risk_features(df)
        
        logger.info(f"Feature engineering complete. Added {self._feature_count} features.")
        return df
    
    def add_time_features(self, df: DataFrame) -> DataFrame:
        """Add time-based features."""
        logger.info("Adding time features...")
        
        df = df.withColumn("hour", F.hour(F.col(self.timestamp_col)))
        df = df.withColumn("day_of_week", F.dayofweek(F.col(self.timestamp_col)))
        df = df.withColumn("day_of_month", F.dayofmonth(F.col(self.timestamp_col)))
        df = df.withColumn("month", F.month(F.col(self.timestamp_col)))
        
        # Binary features
        df = df.withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
        )
        df = df.withColumn(
            "is_night",
            F.when((F.col("hour") >= 22) | (F.col("hour") <= 5), 1).otherwise(0)
        )
        df = df.withColumn(
            "is_business_hours",
            F.when((F.col("hour") >= 9) & (F.col("hour") <= 17), 1).otherwise(0)
        )
        
        self._feature_count += 7
        return df
    
    def add_amount_features(self, df: DataFrame) -> DataFrame:
        """Add amount-based features."""
        logger.info("Adding amount features...")
        
        # Global statistics
        stats = df.agg(
            F.avg(self.amount_col).alias("global_avg"),
            F.stddev(self.amount_col).alias("global_std")
        ).collect()[0]
        
        global_avg = stats["global_avg"] or 0
        global_std = stats["global_std"] or 1
        
        # Amount z-score (global)
        df = df.withColumn(
            "amount_zscore",
            (F.col(self.amount_col) - F.lit(global_avg)) / F.lit(global_std)
        )
        
        # Amount buckets
        df = df.withColumn(
            "amount_bucket",
            F.when(F.col(self.amount_col) < 10, "micro")
            .when(F.col(self.amount_col) < 50, "small")
            .when(F.col(self.amount_col) < 200, "medium")
            .when(F.col(self.amount_col) < 500, "large")
            .otherwise("very_large")
        )
        
        # Log amount (handles skewness)
        df = df.withColumn(
            "log_amount",
            F.log1p(F.col(self.amount_col))
        )
        
        # Is round amount (potential structuring)
        df = df.withColumn(
            "is_round_amount",
            F.when(F.col(self.amount_col) % 50 == 0, 1).otherwise(0)
        )
        
        # Customer-level amount statistics
        customer_window = Window.partitionBy(self.customer_col)
        
        df = df.withColumn(
            "customer_avg_amount",
            F.avg(self.amount_col).over(customer_window)
        )
        df = df.withColumn(
            "customer_std_amount",
            F.stddev(self.amount_col).over(customer_window)
        )
        df = df.withColumn(
            "amount_vs_customer_avg",
            F.col(self.amount_col) / F.col("customer_avg_amount")
        )
        
        self._feature_count += 8
        return df
    
    def add_velocity_features(self, df: DataFrame) -> DataFrame:
        """Add velocity-based features (transaction frequency)."""
        logger.info("Adding velocity features...")
        
        # Customer transaction count
        customer_window = Window.partitionBy(self.customer_col)
        
        df = df.withColumn(
            "customer_txn_count",
            F.count("*").over(customer_window)
        )
        
        df = df.withColumn(
            "customer_total_amount",
            F.sum(self.amount_col).over(customer_window)
        )
        
        # Recent activity window (last N transactions)
        time_window = Window.partitionBy(self.customer_col).orderBy(self.timestamp_col)
        
        df = df.withColumn(
            "txn_number",
            F.row_number().over(time_window)
        )
        
        # Time since last transaction
        df = df.withColumn(
            "prev_txn_time",
            F.lag(self.timestamp_col, 1).over(time_window)
        )
        df = df.withColumn(
            "seconds_since_last_txn",
            F.when(
                F.col("prev_txn_time").isNotNull(),
                F.unix_timestamp(F.col(self.timestamp_col)) - F.unix_timestamp(F.col("prev_txn_time"))
            ).otherwise(F.lit(None))
        )
        
        # Rapid transactions flag (< 60 seconds apart)
        df = df.withColumn(
            "is_rapid_txn",
            F.when(
                (F.col("seconds_since_last_txn").isNotNull()) & 
                (F.col("seconds_since_last_txn") < 60), 1
            ).otherwise(0)
        )
        
        # Drop intermediate column
        df = df.drop("prev_txn_time")
        
        self._feature_count += 6
        return df
    
    def add_merchant_risk_features(self, df: DataFrame) -> DataFrame:
        """Add merchant-based risk features."""
        logger.info("Adding merchant risk features...")
        
        # Calculate merchant fraud rates
        merchant_stats = df.groupBy(self.merchant_col).agg(
            F.count("*").alias("merchant_txn_count"),
            F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias("merchant_fraud_count"),
            F.avg(self.amount_col).alias("merchant_avg_amount")
        ).withColumn(
            "merchant_fraud_rate",
            F.col("merchant_fraud_count") / F.col("merchant_txn_count")
        )
        
        # Join back to main df
        df = df.join(
            merchant_stats.select(
                self.merchant_col,
                "merchant_txn_count",
                "merchant_fraud_rate",
                "merchant_avg_amount"
            ),
            on=self.merchant_col,
            how="left"
        )
        
        # Merchant risk tier
        df = df.withColumn(
            "merchant_risk_tier",
            F.when(F.col("merchant_fraud_rate") > 0.02, "high")
            .when(F.col("merchant_fraud_rate") > 0.005, "medium")
            .otherwise("low")
        )
        
        # Amount vs merchant average
        df = df.withColumn(
            "amount_vs_merchant_avg",
            F.col(self.amount_col) / F.col("merchant_avg_amount")
        )
        
        self._feature_count += 5
        return df
    
    def get_feature_summary(self) -> dict:
        """Get summary of engineered features."""
        return {
            "total_features_added": self._feature_count,
            "feature_categories": {
                "time_features": ["hour", "day_of_week", "day_of_month", "month", 
                                  "is_weekend", "is_night", "is_business_hours"],
                "amount_features": ["amount_zscore", "amount_bucket", "log_amount",
                                    "is_round_amount", "customer_avg_amount", 
                                    "customer_std_amount", "amount_vs_customer_avg"],
                "velocity_features": ["customer_txn_count", "customer_total_amount",
                                      "txn_number", "seconds_since_last_txn", "is_rapid_txn"],
                "merchant_features": ["merchant_txn_count", "merchant_fraud_rate",
                                      "merchant_avg_amount", "merchant_risk_tier",
                                      "amount_vs_merchant_avg"]
            }
        }
