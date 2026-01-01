"""Unit tests for the analytics module."""

import pytest
from pyspark.sql import functions as F


class TestFraudAnalytics:
    """Test suite for FraudAnalytics class."""
    
    def test_executive_summary(self, spark, sample_transaction_df):
        """Test executive summary calculations."""
        from src.analytics.fraud_analytics import FraudAnalytics
        
        analytics = FraudAnalytics(
            sample_transaction_df,
            fraud_column="is_fraud",
            amount_column="amount",
            category_column="merchant_category",
            merchant_column="merchant_id",
            timestamp_column="transaction_time"
        )
        
        results = analytics.run_all_analytics()
        
        assert "executive_summary" in results
        summary = results["executive_summary"]
        assert summary.data[0]["total_transactions"] == 5
        assert summary.data[0]["fraud_transactions"] == 1
        assert summary.data[0]["fraud_rate_percent"] == 20.0
    
    def test_fraud_by_category(self, spark, sample_transaction_df):
        """Test fraud rate by category."""
        from src.analytics.fraud_analytics import FraudAnalytics
        
        analytics = FraudAnalytics(
            sample_transaction_df,
            fraud_column="is_fraud",
            amount_column="amount",
            category_column="merchant_category"
        )
        
        results = analytics.run_all_analytics()
        
        assert "fraud_by_category" in results
        category_data = results["fraud_by_category"].data
        
        # online_shopping should have 100% fraud rate (1 fraud out of 1 transaction)
        online = next((c for c in category_data if c["category"] == "online_shopping"), None)
        assert online is not None
        assert online["fraud_rate_percent"] == 100.0
    
    def test_anomaly_detection(self, spark, sample_transaction_df):
        """Test anomaly detection."""
        from src.analytics.fraud_analytics import FraudAnalytics
        
        analytics = FraudAnalytics(
            sample_transaction_df,
            fraud_column="is_fraud",
            amount_column="amount"
        )
        
        results = analytics.run_all_analytics()
        
        assert "anomalies" in results
        anomaly_data = results["anomalies"].data[0]
        
        assert "mean_amount" in anomaly_data
        assert "std_amount" in anomaly_data
        assert "threshold_amount" in anomaly_data
    
    def test_empty_dataframe(self, spark, empty_df):
        """Test handling of empty dataframe."""
        from src.analytics.fraud_analytics import FraudAnalytics
        
        # Create empty df with required columns
        empty_with_schema = spark.createDataFrame(
            [],
            "is_fraud: int, amt: double, category: string, merchant: string, trans_date_trans_time: timestamp"
        )
        
        analytics = FraudAnalytics(empty_with_schema)
        
        # Should not raise an exception
        results = analytics.run_all_analytics()
        assert "executive_summary" in results


class TestAnalyticsReporter:
    """Test suite for AnalyticsReporter class."""
    
    def test_json_export(self, spark, sample_transaction_df, tmp_path):
        """Test JSON report generation."""
        from src.analytics.fraud_analytics import FraudAnalytics
        from src.analytics.reports import AnalyticsReporter
        
        analytics = FraudAnalytics(
            sample_transaction_df,
            fraud_column="is_fraud",
            amount_column="amount"
        )
        results = analytics.run_all_analytics()
        
        reporter = AnalyticsReporter(results, "test_report")
        json_path = tmp_path / "test_report.json"
        json_output = reporter.to_json(json_path)
        
        assert json_path.exists()
        assert "report_name" in json_output
        assert "test_report" in json_output
    
    def test_html_export(self, spark, sample_transaction_df, tmp_path):
        """Test HTML report generation."""
        from src.analytics.fraud_analytics import FraudAnalytics
        from src.analytics.reports import AnalyticsReporter
        
        analytics = FraudAnalytics(
            sample_transaction_df,
            fraud_column="is_fraud",
            amount_column="amount"
        )
        results = analytics.run_all_analytics()
        
        reporter = AnalyticsReporter(results, "test_report")
        html_path = tmp_path / "test_report.html"
        html_output = reporter.to_html(html_path)
        
        assert html_path.exists()
        assert "<!DOCTYPE html>" in html_output
        assert "Executive Summary" in html_output
        assert "Fraud" in html_output  # More flexible assertion
