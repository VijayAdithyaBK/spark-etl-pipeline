"""
Business Rules Engine Module.
Demonstrates: Rule pattern, validation framework, configurable rules.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class ValidationResult:
    """Result of a validation rule check."""
    rule_name: str
    passed: bool
    failed_count: int = 0
    total_count: int = 0
    message: str = ""
    
    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return (self.total_count - self.failed_count) / self.total_count


@dataclass
class Rule:
    """A data validation rule."""
    name: str
    description: str
    condition: Callable[[DataFrame], DataFrame]  # Returns filtered DataFrame of failures
    severity: str = "error"  # error, warning, info
    
    def check(self, df: DataFrame) -> ValidationResult:
        """Execute the rule and return results."""
        total = df.count()
        failures_df = self.condition(df)
        failed = failures_df.count() if failures_df is not None else 0
        
        passed = failed == 0
        message = f"Rule '{self.name}': {failed}/{total} records failed" if not passed else f"Rule '{self.name}': passed"
        
        logger.log("WARNING" if not passed and self.severity == "warning" else "INFO", message)
        
        return ValidationResult(
            rule_name=self.name,
            passed=passed,
            failed_count=failed,
            total_count=total,
            message=message
        )


class RuleEngine:
    """Engine to execute multiple validation rules."""
    
    def __init__(self):
        self._rules: list[Rule] = []
    
    def add_rule(self, rule: Rule) -> "RuleEngine":
        """Add a rule to the engine."""
        self._rules.append(rule)
        return self
    
    def add_not_null_rule(self, column: str) -> "RuleEngine":
        """Add a not-null validation rule."""
        rule = Rule(
            name=f"not_null_{column}",
            description=f"Column {column} should not be null",
            condition=lambda df, c=column: df.filter(F.col(c).isNull())
        )
        return self.add_rule(rule)
    
    def add_positive_rule(self, column: str) -> "RuleEngine":
        """Add a positive value validation rule."""
        rule = Rule(
            name=f"positive_{column}",
            description=f"Column {column} should be positive",
            condition=lambda df, c=column: df.filter(F.col(c) <= 0)
        )
        return self.add_rule(rule)
    
    def add_range_rule(self, column: str, min_val: float, max_val: float) -> "RuleEngine":
        """Add a range validation rule."""
        rule = Rule(
            name=f"range_{column}",
            description=f"Column {column} should be between {min_val} and {max_val}",
            condition=lambda df, c=column, mn=min_val, mx=max_val: df.filter(
                (F.col(c) < mn) | (F.col(c) > mx)
            )
        )
        return self.add_rule(rule)
    
    def add_unique_rule(self, columns: list[str]) -> "RuleEngine":
        """Add a uniqueness validation rule."""
        cols_str = "_".join(columns)
        rule = Rule(
            name=f"unique_{cols_str}",
            description=f"Columns {columns} should be unique",
            condition=lambda df, cols=columns: (
                df.groupBy(*cols).count().filter(F.col("count") > 1)
            )
        )
        return self.add_rule(rule)
    
    def execute(self, df: DataFrame) -> list[ValidationResult]:
        """Execute all rules and return results."""
        logger.info(f"Executing {len(self._rules)} validation rules")
        return [rule.check(df) for rule in self._rules]
    
    def execute_and_summarize(self, df: DataFrame) -> dict:
        """Execute rules and return summary."""
        results = self.execute(df)
        
        return {
            "total_rules": len(results),
            "passed_rules": sum(1 for r in results if r.passed),
            "failed_rules": sum(1 for r in results if not r.passed),
            "results": [
                {"rule": r.rule_name, "passed": r.passed, "pass_rate": r.pass_rate}
                for r in results
            ]
        }
