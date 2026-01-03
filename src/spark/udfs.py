"""
Spark User-Defined Functions (UDFs) Module.

Demonstrates:
- Standard Python UDFs
- Pandas UDFs (vectorized)
- Type-safe UDF registration
- UDF performance considerations
"""

from __future__ import annotations

import hashlib
from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
)

# Try to import pandas for vectorized UDFs
try:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available. Vectorized UDFs disabled.")


# ============================================================================
# Standard Python UDFs
# ============================================================================


def _categorize_amount(amount: float) -> str:
    """Categorize transaction amount into buckets."""
    if amount is None:
        return "unknown"
    if amount < 50:
        return "micro"
    elif amount < 100:
        return "small"
    elif amount < 500:
        return "medium"
    elif amount < 1000:
        return "large"
    elif amount < 5000:
        return "very_large"
    else:
        return "premium"


def _mask_sensitive_data(value: str, mask_char: str = "*", visible: int = 4) -> str:
    """Mask sensitive data, showing only last N characters."""
    if value is None or len(value) <= visible:
        return value
    return mask_char * (len(value) - visible) + value[-visible:]


def _extract_hour_bucket(timestamp_str: str) -> str:
    """Extract hour bucket from timestamp string."""
    if timestamp_str is None:
        return "unknown"

    try:
        # Extract hour from ISO format timestamp
        hour = int(timestamp_str[11:13])

        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    except (ValueError, IndexError):
        return "unknown"


def _is_weekend_transaction(day_of_week: int) -> bool:
    """Check if day of week is weekend (1=Sunday, 7=Saturday)."""
    if day_of_week is None:
        return False
    return day_of_week in (1, 7)


def _calculate_risk_score(
    amount: float,
    is_fraud: int,
    device_type: str,
    transaction_type: str,
) -> float:
    """
    Calculate a simple risk score for transactions.

    This is a demonstration of business logic in a UDF.
    In production, this would use ML models.
    """
    score = 0.0

    # Amount-based risk
    if amount is not None:
        if amount > 5000:
            score += 30
        elif amount > 1000:
            score += 15
        elif amount > 500:
            score += 5

    # Device type risk
    device_risks = {
        "mobile": 5,
        "desktop": 3,
        "pos_terminal": 1,
    }
    score += device_risks.get(device_type or "", 10)  # Unknown device = higher risk

    # Transaction type risk
    type_risks = {
        "purchase": 1,
        "transfer": 10,
        "payment": 3,
        "refund": 5,
    }
    score += type_risks.get(transaction_type or "", 5)

    # Known fraud
    if is_fraud:
        score += 50

    # Normalize to 0-100
    return min(100.0, max(0.0, score))


def _hash_customer_id(customer_id: str) -> str:
    """Create anonymized hash of customer ID."""
    if customer_id is None:
        return None
    return hashlib.sha256(customer_id.encode()).hexdigest()[:16]


def _parse_currency_amount(amount_str: str) -> float:
    """Parse currency string to float (e.g., '$1,234.56' -> 1234.56)."""
    if amount_str is None:
        return 0.0

    # Remove currency symbols and commas
    cleaned = amount_str.replace("$", "").replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ============================================================================
# Create UDF wrappers
# ============================================================================

# Register UDFs with Spark types
amount_category_udf = F.udf(_categorize_amount, StringType())
mask_sensitive_udf = F.udf(_mask_sensitive_data, StringType())
extract_hour_udf = F.udf(_extract_hour_bucket, StringType())
is_weekend_udf = F.udf(_is_weekend_transaction, StringType())
hash_customer_udf = F.udf(_hash_customer_id, StringType())
parse_currency_udf = F.udf(_parse_currency_amount, DoubleType())

# Risk score UDF (multi-argument)
transaction_risk_score_udf = F.udf(_calculate_risk_score, DoubleType())


# ============================================================================
# Pandas UDFs (Vectorized - Better Performance)
# ============================================================================

if PANDAS_AVAILABLE:

    @pandas_udf(StringType())
    def vectorized_amount_category(amounts: pd.Series) -> pd.Series:
        """
        Vectorized version of amount categorization.

        Pandas UDFs are significantly faster than row-at-a-time UDFs
        because they operate on batches using vectorized operations.
        """
        return amounts.apply(_categorize_amount)

    @pandas_udf(StringType())
    def vectorized_mask_sensitive(values: pd.Series) -> pd.Series:
        """Vectorized masking of sensitive data."""
        return values.apply(lambda x: _mask_sensitive_data(x) if x else x)

    @pandas_udf(DoubleType())
    def vectorized_normalize_amount(amounts: pd.Series) -> pd.Series:
        """
        Normalize amounts to 0-1 range within the batch.

        Note: This normalizes within each batch, not globally.
        For global normalization, compute statistics first.
        """
        if amounts.empty or amounts.isna().all():
            return amounts

        min_val = amounts.min()
        max_val = amounts.max()

        if max_val == min_val:
            return pd.Series([0.5] * len(amounts))

        return (amounts - min_val) / (max_val - min_val)

    @pandas_udf(IntegerType())
    def vectorized_count_transactions(values: pd.Series) -> pd.Series:
        """Count characters in strings (demonstration)."""
        return values.str.len()


# ============================================================================
# UDF Registration Functions
# ============================================================================


def register_all_udfs(spark: SparkSession) -> None:
    """
    Register all UDFs with the Spark session for SQL usage.

    This allows UDFs to be used in Spark SQL queries.

    Args:
        spark: Active Spark session.

    Example:
        >>> register_all_udfs(spark)
        >>> spark.sql("SELECT categorize_amount(amount) FROM transactions")
    """
    # Register standard UDFs
    spark.udf.register("categorize_amount", _categorize_amount, StringType())
    spark.udf.register("mask_sensitive", _mask_sensitive_data, StringType())
    spark.udf.register("extract_hour_bucket", _extract_hour_bucket, StringType())
    spark.udf.register("is_weekend", _is_weekend_transaction, StringType())
    spark.udf.register("hash_customer", _hash_customer_id, StringType())
    spark.udf.register("parse_currency", _parse_currency_amount, DoubleType())
    spark.udf.register("calculate_risk_score", _calculate_risk_score, DoubleType())

    logger.info("Registered 7 UDFs with Spark session")


def get_available_udfs() -> dict[str, str]:
    """
    Get dictionary of available UDFs with descriptions.

    Returns:
        Dictionary mapping UDF names to descriptions.
    """
    return {
        "amount_category_udf": "Categorize amount into buckets (micro/small/medium/large/etc.)",
        "mask_sensitive_udf": "Mask sensitive data showing only last 4 characters",
        "extract_hour_udf": "Extract time-of-day bucket from timestamp",
        "is_weekend_udf": "Check if day of week is weekend",
        "transaction_risk_score_udf": "Calculate risk score based on transaction features",
        "hash_customer_udf": "Create anonymized hash of customer ID",
        "parse_currency_udf": "Parse currency string to float",
    }


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("UDFTest").getOrCreate()

    # Create sample data
    data = [
        ("CUST_001", 45.50, "2023-06-15 14:30:00", 0, "mobile", "purchase"),
        ("CUST_002", 1250.00, "2023-06-16 23:45:00", 0, "desktop", "transfer"),
        ("CUST_003", 75.25, "2023-06-17 08:15:00", 1, "pos_terminal", "purchase"),
        ("CUST_004", 5500.00, "2023-06-15 20:00:00", 0, "mobile", "payment"),
    ]

    df = spark.createDataFrame(
        data,
        [
            "customer_id",
            "amount",
            "transaction_time",
            "is_fraud",
            "device_type",
            "transaction_type",
        ],
    )

    print("Original DataFrame:")
    df.show()

    # Apply UDFs
    result = df.select(
        "customer_id",
        "amount",
        amount_category_udf("amount").alias("amount_category"),
        mask_sensitive_udf("customer_id").alias("masked_customer"),
        extract_hour_udf("transaction_time").alias("time_bucket"),
        transaction_risk_score_udf(
            "amount", "is_fraud", "device_type", "transaction_type"
        ).alias("risk_score"),
        hash_customer_udf("customer_id").alias("customer_hash"),
    )

    print("After applying UDFs:")
    result.show(truncate=False)

    # Register for SQL and test
    register_all_udfs(spark)
    df.createOrReplaceTempView("transactions")

    sql_result = spark.sql(
        """
        SELECT 
            customer_id,
            amount,
            categorize_amount(amount) as category,
            mask_sensitive(customer_id) as masked
        FROM transactions
    """
    )

    print("SQL Query Result:")
    sql_result.show()

    spark.stop()
