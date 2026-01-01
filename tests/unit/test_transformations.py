"""Unit tests for transformations module."""

import pytest
from pyspark.sql import functions as F
from src.spark.transformations import (
    clean_column_names,
    add_processing_metadata,
    handle_nulls,
    DataFrameTransformations,
)


class TestCleanColumnNames:
    def test_clean_spaces(self, spark):
        df = spark.createDataFrame([("a",)], ["First Name"])
        result = clean_column_names(df)
        assert result.columns == ["first_name"]
    
    def test_clean_special_chars(self, spark):
        df = spark.createDataFrame([("a",)], ["Amount ($)"])
        result = clean_column_names(df)
        assert "amount" in result.columns[0].lower()


class TestAddProcessingMetadata:
    def test_adds_metadata_columns(self, spark):
        df = spark.createDataFrame([(1,), (2,)], ["id"])
        result = add_processing_metadata(df, "test_pipeline")
        
        assert "_processing_timestamp" in result.columns
        assert "_pipeline_name" in result.columns
        assert "_row_id" in result.columns


class TestHandleNulls:
    def test_drop_nulls(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, None), (3, "c")], ["id", "value"])
        result = handle_nulls(df, strategy="drop")
        assert result.count() == 2
    
    def test_fill_nulls(self, spark):
        df = spark.createDataFrame([(1, None)], "id: int, value: string")
        result = handle_nulls(df, strategy="fill", fill_values={"value": "default"})
        
        value = result.select("value").collect()[0][0]
        assert value == "default"


class TestDataFrameTransformations:
    def test_chain_transformations(self, sample_transaction_df):
        transformer = DataFrameTransformations(sample_transaction_df)
        
        result = (
            transformer
            .add_metadata("test")
            .get_result()
        )
        
        assert "_pipeline_name" in result.columns
        assert len(transformer.get_transformation_log()) == 1
    
    def test_filter_by_amount(self, sample_transaction_df):
        transformer = DataFrameTransformations(sample_transaction_df)
        
        result = (
            transformer
            .filter_by_amount("amount", min_amount=100)
            .get_result()
        )
        
        assert result.count() < sample_transaction_df.count()
