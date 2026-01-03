"""Unit tests for validators module."""

import pytest
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from src.quality.validators import (
    SchemaValidator,
    NullValidator,
    RangeValidator,
    DuplicateValidator,
)


class TestSchemaValidator:
    def test_valid_schema(self, spark):
        expected = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )

        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        validator = SchemaValidator(expected)

        passed, details = validator.validate(df)
        assert passed
        assert len(details["missing_columns"]) == 0

    def test_missing_columns(self, spark):
        expected = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
                StructField("email", StringType()),
            ]
        )

        df = spark.createDataFrame([(1, "a")], ["id", "name"])
        validator = SchemaValidator(expected)

        passed, details = validator.validate(df)
        assert not passed
        assert "email" in details["missing_columns"]


class TestNullValidator:
    def test_no_nulls(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        validator = NullValidator(max_null_ratio=0.0)

        passed, details = validator.validate(df)
        assert passed

    def test_exceeds_threshold(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, None), (3, None)], ["id", "name"])
        validator = NullValidator(columns=["name"], max_null_ratio=0.5)

        passed, details = validator.validate(df)
        assert not passed


class TestRangeValidator:
    def test_within_range(self, spark):
        df = spark.createDataFrame([(50,), (100,), (150,)], ["amount"])
        validator = RangeValidator(ranges={"amount": (0, 200)})

        passed, details = validator.validate(df)
        assert passed

    def test_out_of_range(self, spark):
        df = spark.createDataFrame([(50,), (500,)], ["amount"])
        validator = RangeValidator(ranges={"amount": (0, 200)})

        passed, details = validator.validate(df)
        assert not passed


class TestDuplicateValidator:
    def test_no_duplicates(self, spark):
        df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
        validator = DuplicateValidator(key_columns=["id"])

        passed, details = validator.validate(df)
        assert passed
        assert details["duplicate_count"] == 0

    def test_has_duplicates(self, spark):
        df = spark.createDataFrame([(1,), (1,), (2,)], ["id"])
        validator = DuplicateValidator(key_columns=["id"], max_duplicate_ratio=0.0)

        passed, details = validator.validate(df)
        assert not passed
        assert details["duplicate_count"] == 1
