"""
Geographic Analysis Module.
Analyzes location-based fraud patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType


@dataclass
class GeoInsight:
    """Geographic insight."""

    insight_type: str
    description: str
    value: float
    risk_level: str


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None

    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


class GeographicAnalyzer:
    """
    Geographic analysis for fraud detection.

    Analyzes transaction locations, calculates distances from
    customer's typical location, and identifies fraud hotspots.
    """

    def __init__(
        self,
        df: DataFrame,
        customer_column: str = "cc_num",
        customer_lat: str = "lat",
        customer_long: str = "long",
        merchant_lat: str = "merch_lat",
        merchant_long: str = "merch_long",
        fraud_column: str = "is_fraud",
    ):
        self.df = df
        self.customer_col = customer_column
        self.cust_lat = customer_lat
        self.cust_long = customer_long
        self.merch_lat = merchant_lat
        self.merch_long = merchant_long
        self.fraud_col = fraud_column
        self._insights: list[GeoInsight] = []

    def add_geographic_features(self) -> DataFrame:
        """Add geographic-based features."""
        logger.info("Adding geographic features...")

        df = self.df

        # Check if required columns exist
        required_cols = [self.cust_lat, self.cust_long, self.merch_lat, self.merch_long]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns for geographic analysis: {missing_cols}")
            return df

        # Register UDF for distance calculation
        haversine_udf = F.udf(haversine_distance, DoubleType())

        # Calculate distance from customer location to merchant
        df = df.withColumn(
            "transaction_distance_km",
            haversine_udf(
                F.col(self.cust_lat),
                F.col(self.cust_long),
                F.col(self.merch_lat),
                F.col(self.merch_long),
            ),
        )

        # Calculate customer's median location (home base)
        customer_window = Window.partitionBy(self.customer_col)

        df = df.withColumn(
            "customer_median_lat",
            F.percentile_approx(F.col(self.cust_lat), 0.5).over(customer_window),
        )
        df = df.withColumn(
            "customer_median_long",
            F.percentile_approx(F.col(self.cust_long), 0.5).over(customer_window),
        )

        # Calculate distance from customer's typical location
        df = df.withColumn(
            "distance_from_home_km",
            haversine_udf(
                F.col("customer_median_lat"),
                F.col("customer_median_long"),
                F.col(self.cust_lat),
                F.col(self.cust_long),
            ),
        )

        # Distance buckets
        df = df.withColumn(
            "distance_bucket",
            F.when(F.col("transaction_distance_km") < 5, "local")
            .when(F.col("transaction_distance_km") < 25, "nearby")
            .when(F.col("transaction_distance_km") < 100, "regional")
            .when(F.col("transaction_distance_km") < 500, "distant")
            .otherwise("very_distant"),
        )

        # Flag unusual locations (far from home)
        df = df.withColumn(
            "is_unusual_location",
            F.when(F.col("distance_from_home_km") > 200, 1).otherwise(0),
        )

        # State-level analysis - get state from existing column if available
        if "state" in df.columns:
            df = df.withColumn(
                "customer_home_state", F.first("state").over(customer_window)
            )
            df = df.withColumn(
                "is_out_of_state",
                F.when(F.col("state") != F.col("customer_home_state"), 1).otherwise(0),
            )

        logger.info("Geographic features added successfully.")
        return df

    def analyze_fraud_hotspots(self) -> list[dict]:
        """Identify geographic regions with high fraud rates."""
        logger.info("Analyzing fraud hotspots...")

        if "state" not in self.df.columns:
            return []

        hotspots = (
            self.df.groupBy("state")
            .agg(
                F.count("*").alias("total_transactions"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
                F.avg(self.fraud_col).alias("fraud_rate"),
            )
            .filter(F.col("total_transactions") >= 100)
            .orderBy(F.col("fraud_rate").desc())
            .limit(10)
            .collect()
        )

        return [
            {
                "state": row["state"],
                "total_transactions": row["total_transactions"],
                "fraud_count": row["fraud_count"],
                "fraud_rate_percent": round(row["fraud_rate"] * 100, 2),
            }
            for row in hotspots
        ]

    def analyze_distance_risk(self) -> list[dict]:
        """Analyze fraud rate by transaction distance."""

        if "distance_bucket" not in self.df.columns:
            # Add it temporarily
            df = self.add_geographic_features()
        else:
            df = self.df

        distance_analysis = (
            df.groupBy("distance_bucket")
            .agg(
                F.count("*").alias("total_transactions"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
            )
            .withColumn(
                "fraud_rate", F.col("fraud_count") / F.col("total_transactions")
            )
            .orderBy(F.col("fraud_rate").desc())
            .collect()
        )

        return [
            {
                "distance_bucket": row["distance_bucket"],
                "total_transactions": row["total_transactions"],
                "fraud_count": row["fraud_count"],
                "fraud_rate_percent": round(row["fraud_rate"] * 100, 2),
            }
            for row in distance_analysis
        ]

    def get_geographic_summary(self) -> dict:
        """Get summary of geographic analysis."""
        hotspots = self.analyze_fraud_hotspots()
        distance_risk = self.analyze_distance_risk()

        # Calculate unusual location fraud rate
        if "is_unusual_location" in self.df.columns:
            unusual_stats = (
                self.df.filter(F.col("is_unusual_location") == 1)
                .agg(
                    F.count("*").alias("count"),
                    F.avg(self.fraud_col).alias("fraud_rate"),
                )
                .collect()[0]
            )
        else:
            unusual_stats = {"count": 0, "fraud_rate": 0}

        return {
            "hotspots": hotspots[:5],
            "distance_risk": distance_risk,
            "unusual_location_stats": {
                "count": unusual_stats["count"] or 0,
                "fraud_rate_percent": round(
                    (unusual_stats["fraud_rate"] or 0) * 100, 2
                ),
            },
        }
