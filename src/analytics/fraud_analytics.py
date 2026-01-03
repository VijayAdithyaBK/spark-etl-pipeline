"""
Fraud Analytics Module.
Demonstrates: Business intelligence, statistical analysis, pattern detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


@dataclass
class AnalyticsResult:
    """Container for analytics results."""

    name: str
    data: list[dict]
    summary: str
    actionable_items: list[str] = field(default_factory=list)


class FraudAnalytics:
    """
    Comprehensive fraud analytics engine.

    Provides insights on:
    - Fraud rate by category and merchant
    - High-risk transaction patterns
    - Statistical anomaly detection
    - Time-based fraud trends
    """

    def __init__(
        self,
        df: DataFrame,
        fraud_column: str = "is_fraud",
        amount_column: str = "amt",
        category_column: str = "category",
        merchant_column: str = "merchant",
        timestamp_column: str = "trans_date_trans_time",
    ):
        self.df = df
        self.fraud_col = fraud_column
        self.amount_col = amount_column
        self.category_col = category_column
        self.merchant_col = merchant_column
        self.timestamp_col = timestamp_column
        self._results: dict[str, AnalyticsResult] = {}

    def run_all_analytics(self) -> dict[str, AnalyticsResult]:
        """Run all analytics and return results."""
        logger.info("Running comprehensive fraud analytics...")

        self._results["executive_summary"] = self._executive_summary()
        self._results["fraud_by_category"] = self._fraud_rate_by_category()
        self._results["fraud_by_merchant"] = self._fraud_rate_by_merchant()
        self._results["high_risk_patterns"] = self._detect_high_risk_patterns()
        self._results["anomalies"] = self._detect_anomalies()
        self._results["time_trends"] = self._analyze_time_trends()

        logger.info(f"Completed {len(self._results)} analytics modules")
        return self._results

    def _executive_summary(self) -> AnalyticsResult:
        """Generate executive summary with key metrics."""
        total_count = self.df.count()
        fraud_count = self.df.filter(F.col(self.fraud_col) == 1).count()
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0

        total_amount = self.df.agg(F.sum(self.amount_col)).collect()[0][0] or 0
        fraud_amount = (
            self.df.filter(F.col(self.fraud_col) == 1)
            .agg(F.sum(self.amount_col))
            .collect()[0][0]
            or 0
        )

        avg_fraud_amount = (
            self.df.filter(F.col(self.fraud_col) == 1)
            .agg(F.avg(self.amount_col))
            .collect()[0][0]
            or 0
        )

        avg_legit_amount = (
            self.df.filter(F.col(self.fraud_col) == 0)
            .agg(F.avg(self.amount_col))
            .collect()[0][0]
            or 0
        )

        data = [
            {
                "total_transactions": total_count,
                "fraud_transactions": fraud_count,
                "legitimate_transactions": total_count - fraud_count,
                "fraud_rate_percent": round(fraud_rate, 4),
                "total_amount": round(total_amount, 2),
                "fraud_amount": round(fraud_amount, 2),
                "amount_at_risk_percent": (
                    round(fraud_amount / total_amount * 100, 2)
                    if total_amount > 0
                    else 0
                ),
                "avg_fraud_transaction": round(avg_fraud_amount, 2),
                "avg_legitimate_transaction": round(avg_legit_amount, 2),
            }
        ]

        actionable = []
        if fraud_rate > 1:
            actionable.append(
                f"⚠️ HIGH ALERT: Fraud rate ({fraud_rate:.2f}%) exceeds 1% threshold"
            )
        if avg_fraud_amount > avg_legit_amount * 2:
            actionable.append(
                f"⚠️ Fraudulent transactions avg ${avg_fraud_amount:.2f} - {avg_fraud_amount/avg_legit_amount:.1f}x higher than legitimate"
            )

        return AnalyticsResult(
            name="Executive Summary",
            data=data,
            summary=f"Detected {fraud_count:,} fraudulent transactions ({fraud_rate:.2f}%) totaling ${fraud_amount:,.2f}",
            actionable_items=actionable,
        )

    def _fraud_rate_by_category(self) -> AnalyticsResult:
        """Calculate fraud rate by transaction category."""
        if self.category_col not in self.df.columns:
            return AnalyticsResult(
                name="Fraud by Category",
                data=[],
                summary="Category column not available",
            )

        result = (
            self.df.groupBy(self.category_col)
            .agg(
                F.count("*").alias("total_transactions"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
                F.sum(self.amount_col).alias("total_amount"),
                F.sum(
                    F.when(
                        F.col(self.fraud_col) == 1, F.col(self.amount_col)
                    ).otherwise(0)
                ).alias("fraud_amount"),
            )
            .withColumn(
                "fraud_rate_percent",
                F.round(F.col("fraud_count") / F.col("total_transactions") * 100, 4),
            )
            .orderBy(F.col("fraud_rate_percent").desc())
            .collect()
        )

        data = [
            {
                "category": row[self.category_col],
                "total_transactions": row["total_transactions"],
                "fraud_count": row["fraud_count"],
                "fraud_rate_percent": float(row["fraud_rate_percent"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "fraud_amount": round(float(row["fraud_amount"]), 2),
            }
            for row in result
        ]

        # Find high-risk categories (fraud rate > overall average)
        avg_rate = sum(d["fraud_rate_percent"] for d in data) / len(data) if data else 0
        high_risk = [d for d in data if d["fraud_rate_percent"] > avg_rate]

        actionable = [
            f"🎯 Focus on '{cat['category']}' category: {cat['fraud_rate_percent']:.2f}% fraud rate, ${cat['fraud_amount']:,.2f} at risk"
            for cat in high_risk[:3]
        ]

        return AnalyticsResult(
            name="Fraud by Category",
            data=data,
            summary=f"Analyzed {len(data)} categories. Highest risk: {data[0]['category'] if data else 'N/A'}",
            actionable_items=actionable,
        )

    def _fraud_rate_by_merchant(self) -> AnalyticsResult:
        """Identify high-risk merchants."""
        if self.merchant_col not in self.df.columns:
            return AnalyticsResult(
                name="Fraud by Merchant",
                data=[],
                summary="Merchant column not available",
            )

        result = (
            self.df.groupBy(self.merchant_col)
            .agg(
                F.count("*").alias("total_transactions"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
                F.sum(self.amount_col).alias("total_amount"),
            )
            .withColumn(
                "fraud_rate_percent",
                F.round(F.col("fraud_count") / F.col("total_transactions") * 100, 4),
            )
            .filter(F.col("total_transactions") >= 10)  # Minimum sample size
            .orderBy(F.col("fraud_rate_percent").desc())
            .limit(20)
            .collect()
        )

        data = [
            {
                "merchant": row[self.merchant_col],
                "total_transactions": row["total_transactions"],
                "fraud_count": row["fraud_count"],
                "fraud_rate_percent": float(row["fraud_rate_percent"]),
                "total_amount": round(float(row["total_amount"]), 2),
            }
            for row in result
        ]

        actionable = [
            f"🏪 Review merchant '{m['merchant'][:40]}...': {m['fraud_count']} frauds ({m['fraud_rate_percent']:.2f}%)"
            for m in data[:5]
            if m["fraud_count"] > 0
        ]

        return AnalyticsResult(
            name="Top 20 High-Risk Merchants",
            data=data,
            summary=f"Identified {len([d for d in data if d['fraud_count'] > 0])} merchants with fraud activity",
            actionable_items=actionable,
        )

    def _detect_high_risk_patterns(self) -> AnalyticsResult:
        """Detect high-risk transaction patterns."""
        patterns = []

        # Pattern 1: High-value transactions
        amount_stats = self.df.agg(
            F.avg(self.amount_col).alias("avg"), F.stddev(self.amount_col).alias("std")
        ).collect()[0]

        avg_amount = amount_stats["avg"] or 0
        std_amount = amount_stats["std"] or 0
        high_threshold = avg_amount + (2 * std_amount)

        high_value_fraud = self.df.filter(
            (F.col(self.amount_col) > high_threshold) & (F.col(self.fraud_col) == 1)
        ).count()

        high_value_total = self.df.filter(
            F.col(self.amount_col) > high_threshold
        ).count()
        high_value_fraud_rate = (
            (high_value_fraud / high_value_total * 100) if high_value_total > 0 else 0
        )

        patterns.append(
            {
                "pattern": "High-Value Transactions",
                "threshold": f">${high_threshold:.2f}",
                "occurrences": high_value_total,
                "fraud_count": high_value_fraud,
                "fraud_rate_percent": round(high_value_fraud_rate, 2),
                "risk_level": (
                    "HIGH"
                    if high_value_fraud_rate > 5
                    else "MEDIUM" if high_value_fraud_rate > 1 else "LOW"
                ),
            }
        )

        # Pattern 2: Very low value transactions (potential testing)
        low_threshold = 5.0
        low_value_fraud = self.df.filter(
            (F.col(self.amount_col) < low_threshold) & (F.col(self.fraud_col) == 1)
        ).count()

        low_value_total = self.df.filter(F.col(self.amount_col) < low_threshold).count()
        low_value_fraud_rate = (
            (low_value_fraud / low_value_total * 100) if low_value_total > 0 else 0
        )

        patterns.append(
            {
                "pattern": "Low-Value Transactions (Card Testing)",
                "threshold": f"<${low_threshold:.2f}",
                "occurrences": low_value_total,
                "fraud_count": low_value_fraud,
                "fraud_rate_percent": round(low_value_fraud_rate, 2),
                "risk_level": (
                    "HIGH"
                    if low_value_fraud_rate > 5
                    else "MEDIUM" if low_value_fraud_rate > 1 else "LOW"
                ),
            }
        )

        # Pattern 3: Round amounts (potential structuring)
        round_amounts_fraud = self.df.filter(
            (F.col(self.amount_col) % 100 == 0) & (F.col(self.fraud_col) == 1)
        ).count()

        round_amounts_total = self.df.filter(F.col(self.amount_col) % 100 == 0).count()
        round_fraud_rate = (
            (round_amounts_fraud / round_amounts_total * 100)
            if round_amounts_total > 0
            else 0
        )

        patterns.append(
            {
                "pattern": "Round Amount Transactions",
                "threshold": "Divisible by $100",
                "occurrences": round_amounts_total,
                "fraud_count": round_amounts_fraud,
                "fraud_rate_percent": round(round_fraud_rate, 2),
                "risk_level": (
                    "HIGH"
                    if round_fraud_rate > 5
                    else "MEDIUM" if round_fraud_rate > 1 else "LOW"
                ),
            }
        )

        actionable = [
            f"📊 {p['pattern']}: {p['fraud_rate_percent']:.2f}% fraud rate - {p['risk_level']} risk"
            for p in patterns
            if p["risk_level"] in ("HIGH", "MEDIUM")
        ]

        return AnalyticsResult(
            name="High-Risk Transaction Patterns",
            data=patterns,
            summary=f"Analyzed {len(patterns)} transaction patterns",
            actionable_items=actionable,
        )

    def _detect_anomalies(self) -> AnalyticsResult:
        """Detect statistical anomalies using Z-score method."""
        # Calculate Z-scores for amounts
        stats = self.df.agg(
            F.avg(self.amount_col).alias("mean"), F.stddev(self.amount_col).alias("std")
        ).collect()[0]

        mean_amt = stats["mean"] or 0
        std_amt = stats["std"] or 1  # Avoid division by zero

        df_with_zscore = self.df.withColumn(
            "z_score",
            F.abs((F.col(self.amount_col) - F.lit(mean_amt)) / F.lit(std_amt)),
        )

        # Anomalies are transactions with Z-score > 3
        anomalies_df = df_with_zscore.filter(F.col("z_score") > 3)

        total_anomalies = anomalies_df.count()
        fraud_anomalies = anomalies_df.filter(F.col(self.fraud_col) == 1).count()
        anomaly_fraud_rate = (
            (fraud_anomalies / total_anomalies * 100) if total_anomalies > 0 else 0
        )

        # Get sample anomalies - only select columns that exist
        select_cols = [self.amount_col, "z_score", self.fraud_col]
        if self.category_col in anomalies_df.columns:
            select_cols.append(self.category_col)
        sample_anomalies = anomalies_df.select(*select_cols).limit(10).collect()

        data = [
            {
                "detection_method": "Z-Score > 3",
                "mean_amount": round(mean_amt, 2),
                "std_amount": round(std_amt, 2),
                "threshold_amount": round(mean_amt + 3 * std_amt, 2),
                "total_anomalies": total_anomalies,
                "fraudulent_anomalies": fraud_anomalies,
                "anomaly_fraud_rate_percent": round(anomaly_fraud_rate, 2),
                "sample_anomalies": [
                    {
                        "amount": round(float(row[self.amount_col]), 2),
                        "z_score": round(float(row["z_score"]), 2),
                        "is_fraud": row[self.fraud_col],
                        "category": (
                            row[self.category_col]
                            if self.category_col in row
                            else "N/A"
                        ),
                    }
                    for row in sample_anomalies
                ],
            }
        ]

        actionable = []
        if anomaly_fraud_rate > 10:
            actionable.append(
                f"🚨 {fraud_anomalies} anomalous transactions are fraudulent ({anomaly_fraud_rate:.1f}%)"
            )
        if total_anomalies > 0:
            actionable.append(
                f"📌 Set alert threshold at ${mean_amt + 3 * std_amt:,.2f} to flag anomalies"
            )

        return AnalyticsResult(
            name="Anomaly Detection",
            data=data,
            summary=f"Detected {total_anomalies:,} statistical anomalies, {fraud_anomalies} are fraud ({anomaly_fraud_rate:.1f}%)",
            actionable_items=actionable,
        )

    def _analyze_time_trends(self) -> AnalyticsResult:
        """Analyze time-based fraud patterns."""
        if self.timestamp_col not in self.df.columns:
            return AnalyticsResult(
                name="Time-Based Trends",
                data=[],
                summary="Timestamp column not available",
            )

        # Add time components
        df_time = (
            self.df.withColumn("hour", F.hour(F.col(self.timestamp_col)))
            .withColumn("day_of_week", F.dayofweek(F.col(self.timestamp_col)))
            .withColumn("month", F.month(F.col(self.timestamp_col)))
        )

        # Hourly fraud distribution
        hourly = (
            df_time.groupBy("hour")
            .agg(
                F.count("*").alias("total"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
            )
            .withColumn(
                "fraud_rate", F.round(F.col("fraud_count") / F.col("total") * 100, 2)
            )
            .orderBy("hour")
            .collect()
        )

        hourly_data = [
            {
                "hour": row["hour"],
                "total": row["total"],
                "fraud_count": row["fraud_count"],
                "fraud_rate": float(row["fraud_rate"]),
            }
            for row in hourly
        ]

        # Day of week analysis
        daily = (
            df_time.groupBy("day_of_week")
            .agg(
                F.count("*").alias("total"),
                F.sum(F.when(F.col(self.fraud_col) == 1, 1).otherwise(0)).alias(
                    "fraud_count"
                ),
            )
            .withColumn(
                "fraud_rate", F.round(F.col("fraud_count") / F.col("total") * 100, 2)
            )
            .orderBy("day_of_week")
            .collect()
        )

        day_names = [
            "",
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        daily_data = [
            {
                "day": day_names[row["day_of_week"]],
                "total": row["total"],
                "fraud_count": row["fraud_count"],
                "fraud_rate": float(row["fraud_rate"]),
            }
            for row in daily
        ]

        data = [
            {
                "hourly_distribution": hourly_data,
                "daily_distribution": daily_data,
            }
        ]

        # Find peak fraud hours
        peak_hours = sorted(hourly_data, key=lambda x: x["fraud_rate"], reverse=True)[
            :3
        ]
        peak_days = sorted(daily_data, key=lambda x: x["fraud_rate"], reverse=True)[:2]

        # Build actionable items (avoid nested f-strings)
        peak_hours_str = (
            ", ".join(str(h["hour"]) + ":00" for h in peak_hours)
            if peak_hours
            else "N/A"
        )
        peak_rate = peak_hours[0]["fraud_rate"] if peak_hours else 0
        peak_days_str = ", ".join(d["day"] for d in peak_days) if peak_days else "N/A"

        actionable = [
            f"🕐 Peak fraud hours: {peak_hours_str} ({peak_rate:.2f}% fraud rate)",
            f"📅 High-risk days: {peak_days_str}",
        ]

        peak_hour_display = str(peak_hours[0]["hour"]) if peak_hours else "N/A"

        return AnalyticsResult(
            name="Time-Based Fraud Trends",
            data=data,
            summary=f"Peak fraud hour: {peak_hour_display}:00",
            actionable_items=actionable,
        )

    def get_results(self) -> dict[str, AnalyticsResult]:
        """Get all analytics results."""
        if not self._results:
            self.run_all_analytics()
        return self._results
