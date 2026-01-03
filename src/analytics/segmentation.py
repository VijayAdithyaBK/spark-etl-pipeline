"""
Customer Segmentation Module.
Clusters customers by behavior patterns for risk profiling.
"""

from __future__ import annotations
from dataclasses import dataclass
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


@dataclass
class SegmentProfile:
    """Profile for a customer segment."""

    segment_id: int
    segment_name: str
    customer_count: int
    avg_transaction_amount: float
    avg_transaction_count: float
    fraud_rate: float
    risk_level: str
    characteristics: list[str]


class CustomerSegmentation:
    """
    Customer segmentation using K-Means clustering.

    Groups customers by spending behavior and transaction patterns
    to identify high-risk customer profiles.
    """

    def __init__(
        self,
        df: DataFrame,
        customer_column: str = "cc_num",
        amount_column: str = "amt",
        fraud_column: str = "is_fraud",
        n_clusters: int = 4,
    ):
        self.df = df
        self.customer_col = customer_column
        self.amount_col = amount_column
        self.fraud_col = fraud_column
        self.n_clusters = n_clusters
        self._model = None
        self._customer_df = None
        self._profiles: list[SegmentProfile] = []

    def fit(self) -> "CustomerSegmentation":
        """Fit segmentation model on customer data."""
        logger.info("Aggregating customer metrics...")

        # Aggregate to customer level
        self._customer_df = (
            self.df.groupBy(self.customer_col)
            .agg(
                F.count("*").alias("txn_count"),
                F.sum(self.amount_col).alias("total_amount"),
                F.avg(self.amount_col).alias("avg_amount"),
                F.max(self.amount_col).alias("max_amount"),
                F.min(self.amount_col).alias("min_amount"),
                F.stddev(self.amount_col).alias("std_amount"),
                F.countDistinct("merchant").alias("unique_merchants"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
                (
                    F.sum(F.when(F.col("is_night") == 1, 1).otherwise(0)).alias(
                        "night_txn_count"
                    )
                    if "is_night" in self.df.columns
                    else F.lit(0).alias("night_txn_count")
                ),
                (
                    F.sum(F.when(F.col("is_weekend") == 1, 1).otherwise(0)).alias(
                        "weekend_txn_count"
                    )
                    if "is_weekend" in self.df.columns
                    else F.lit(0).alias("weekend_txn_count")
                ),
            )
            .withColumn("fraud_rate", F.col("fraud_count") / F.col("txn_count"))
            .fillna(0)
        )

        # Select features for clustering
        feature_cols = ["txn_count", "avg_amount", "max_amount", "unique_merchants"]

        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")

        # Build ML pipeline
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        scaler = StandardScaler(
            inputCol="features_raw", outputCol="features", withStd=True, withMean=True
        )
        kmeans = KMeans(
            k=self.n_clusters, seed=42, featuresCol="features", predictionCol="segment"
        )

        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        self._model = pipeline.fit(self._customer_df)

        # Transform customer data
        self._customer_df = self._model.transform(self._customer_df)

        # Generate segment profiles
        self._generate_profiles()

        logger.info(f"Segmentation complete. {len(self._profiles)} segments created.")
        return self

    def _generate_profiles(self):
        """Generate profiles for each segment."""
        segment_stats = (
            self._customer_df.groupBy("segment")
            .agg(
                F.count("*").alias("customer_count"),
                F.avg("avg_amount").alias("avg_transaction_amount"),
                F.avg("txn_count").alias("avg_transaction_count"),
                F.avg("fraud_rate").alias("fraud_rate"),
                F.avg("max_amount").alias("avg_max_amount"),
                F.avg("unique_merchants").alias("avg_unique_merchants"),
            )
            .orderBy("segment")
            .collect()
        )

        # Calculate global averages for comparison
        global_stats = self._customer_df.agg(
            F.avg("avg_amount").alias("global_avg_amount"),
            F.avg("txn_count").alias("global_avg_txns"),
            F.avg("fraud_rate").alias("global_fraud_rate"),
        ).collect()[0]

        global_avg_amount = global_stats["global_avg_amount"] or 0
        global_avg_txns = global_stats["global_avg_txns"] or 0
        global_fraud_rate = global_stats["global_fraud_rate"] or 0

        # Define segment names and characteristics based on metrics
        self._profiles = []
        for row in segment_stats:
            fraud_rate = row["fraud_rate"] or 0
            avg_amount = row["avg_transaction_amount"] or 0
            avg_txns = row["avg_transaction_count"] or 0

            # Determine risk level
            if fraud_rate > global_fraud_rate * 2:
                risk_level = "HIGH"
            elif fraud_rate > global_fraud_rate:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Generate characteristics
            characteristics = []
            if avg_amount > global_avg_amount * 1.5:
                characteristics.append("High spenders")
            elif avg_amount < global_avg_amount * 0.5:
                characteristics.append("Low spenders")
            else:
                characteristics.append("Average spenders")

            if avg_txns > global_avg_txns * 1.5:
                characteristics.append("High frequency")
            elif avg_txns < global_avg_txns * 0.5:
                characteristics.append("Low frequency")

            if fraud_rate > global_fraud_rate * 2:
                characteristics.append("High fraud exposure")

            # Name segment based on characteristics
            if risk_level == "HIGH":
                name = "High-Risk Segment"
            elif avg_amount > global_avg_amount * 1.5:
                name = "Premium Customers"
            elif avg_txns > global_avg_txns * 1.5:
                name = "Frequent Buyers"
            elif avg_amount < global_avg_amount * 0.5:
                name = "Budget Customers"
            else:
                name = "Standard Customers"

            self._profiles.append(
                SegmentProfile(
                    segment_id=row["segment"],
                    segment_name=name,
                    customer_count=row["customer_count"],
                    avg_transaction_amount=round(avg_amount, 2),
                    avg_transaction_count=round(avg_txns, 2),
                    fraud_rate=round(fraud_rate * 100, 4),
                    risk_level=risk_level,
                    characteristics=characteristics,
                )
            )

    def get_profiles(self) -> list[SegmentProfile]:
        """Get segment profiles."""
        return self._profiles

    def get_customer_segments(self) -> DataFrame:
        """Get customer dataframe with segment assignments."""
        return self._customer_df.select(
            self.customer_col, "segment", "txn_count", "avg_amount", "fraud_rate"
        )

    def get_segment_summary(self) -> list[dict]:
        """Get segment summary for reporting."""
        return [
            {
                "segment_id": p.segment_id,
                "segment_name": p.segment_name,
                "customer_count": p.customer_count,
                "avg_transaction_amount": p.avg_transaction_amount,
                "avg_transaction_count": p.avg_transaction_count,
                "fraud_rate_percent": p.fraud_rate,
                "risk_level": p.risk_level,
                "characteristics": p.characteristics,
            }
            for p in self._profiles
        ]
