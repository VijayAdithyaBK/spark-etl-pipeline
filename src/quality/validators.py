"""
Data Validators Module.
Demonstrates: Validation patterns, schema enforcement, data quality checks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


class DataValidator(ABC):
    """Abstract base class for data validators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def validate(self, df: DataFrame) -> tuple[bool, dict]:
        """Validate DataFrame, return (passed, details)."""
        pass


class SchemaValidator(DataValidator):
    """Validate DataFrame schema matches expected."""
    
    name = "schema_validator"
    
    def __init__(self, expected_schema: StructType):
        self.expected_schema = expected_schema
    
    def validate(self, df: DataFrame) -> tuple[bool, dict]:
        actual_columns = set(df.columns)
        expected_columns = set([f.name for f in self.expected_schema.fields])
        
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns
        
        passed = len(missing) == 0
        details = {
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "passed": passed,
        }
        
        if not passed:
            logger.warning(f"Schema validation failed. Missing: {missing}")
        
        return passed, details


class NullValidator(DataValidator):
    """Validate null ratios are within thresholds."""
    
    name = "null_validator"
    
    def __init__(self, columns: Optional[list[str]] = None, max_null_ratio: float = 0.05):
        self.columns = columns
        self.max_null_ratio = max_null_ratio
    
    def validate(self, df: DataFrame) -> tuple[bool, dict]:
        check_columns = self.columns or df.columns
        total_rows = df.count()
        
        if total_rows == 0:
            return True, {"message": "Empty DataFrame", "passed": True}
        
        null_stats = {}
        failed_columns = []
        
        for col in check_columns:
            if col not in df.columns:
                continue
            
            null_count = df.filter(F.col(col).isNull()).count()
            null_ratio = null_count / total_rows
            
            null_stats[col] = {"count": null_count, "ratio": round(null_ratio, 4)}
            
            if null_ratio > self.max_null_ratio:
                failed_columns.append(col)
        
        passed = len(failed_columns) == 0
        details = {
            "null_stats": null_stats,
            "failed_columns": failed_columns,
            "threshold": self.max_null_ratio,
            "passed": passed,
        }
        
        return passed, details


class RangeValidator(DataValidator):
    """Validate numeric columns are within expected ranges."""
    
    name = "range_validator"
    
    def __init__(self, ranges: dict[str, tuple[float, float]]):
        self.ranges = ranges  # {column: (min, max)}
    
    def validate(self, df: DataFrame) -> tuple[bool, dict]:
        violations = {}
        
        for col, (min_val, max_val) in self.ranges.items():
            if col not in df.columns:
                continue
            
            out_of_range = df.filter(
                (F.col(col) < min_val) | (F.col(col) > max_val)
            ).count()
            
            if out_of_range > 0:
                violations[col] = {
                    "out_of_range_count": out_of_range,
                    "expected_range": (min_val, max_val)
                }
        
        passed = len(violations) == 0
        return passed, {"violations": violations, "passed": passed}


class DuplicateValidator(DataValidator):
    """Validate for duplicate records."""
    
    name = "duplicate_validator"
    
    def __init__(self, key_columns: list[str], max_duplicate_ratio: float = 0.01):
        self.key_columns = key_columns
        self.max_duplicate_ratio = max_duplicate_ratio
    
    def validate(self, df: DataFrame) -> tuple[bool, dict]:
        total_rows = df.count()
        unique_rows = df.dropDuplicates(self.key_columns).count()
        duplicate_count = total_rows - unique_rows
        duplicate_ratio = duplicate_count / total_rows if total_rows > 0 else 0
        
        passed = duplicate_ratio <= self.max_duplicate_ratio
        details = {
            "total_rows": total_rows,
            "unique_rows": unique_rows,
            "duplicate_count": duplicate_count,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "threshold": self.max_duplicate_ratio,
            "passed": passed,
        }
        
        return passed, details


class CompositeValidator:
    """Combine multiple validators."""
    
    def __init__(self, validators: list[DataValidator]):
        self.validators = validators
    
    def validate_all(self, df: DataFrame) -> dict:
        results = {}
        all_passed = True
        
        for validator in self.validators:
            passed, details = validator.validate(df)
            results[validator.name] = details
            if not passed:
                all_passed = False
        
        results["all_passed"] = all_passed
        return results
