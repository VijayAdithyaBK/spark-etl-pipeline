"""Test configuration and fixtures."""

import os
import sys
import pytest

# Set environment variables for Windows compatibility before importing pyspark
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    from pyspark.sql import SparkSession
    
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("pytest-spark")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    
    yield session
    
    session.stop()


@pytest.fixture
def sample_transaction_df(spark):
    """Create sample transaction DataFrame for testing."""
    data = [
        ("TXN_001", "CUST_001", "MERCH_001", "grocery", 45.50, "2023-06-15 14:30:00", 0),
        ("TXN_002", "CUST_001", "MERCH_002", "restaurant", 125.00, "2023-06-16 19:45:00", 0),
        ("TXN_003", "CUST_002", "MERCH_001", "electronics", 599.99, "2023-06-15 10:00:00", 0),
        ("TXN_004", "CUST_002", "MERCH_003", "gas_station", 35.00, "2023-06-17 08:00:00", 0),
        ("TXN_005", "CUST_003", "MERCH_002", "online_shopping", 1500.00, "2023-06-15 23:30:00", 1),
    ]
    
    return spark.createDataFrame(
        data,
        ["transaction_id", "customer_id", "merchant_id", "merchant_category", "amount", "transaction_time", "is_fraud"]
    )


@pytest.fixture
def empty_df(spark):
    """Create empty DataFrame for edge case testing."""
    return spark.createDataFrame([], "id: int, value: string")
