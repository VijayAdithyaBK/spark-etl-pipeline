"""
Quality Reports Module.
Demonstrates: Report generation, metrics collection, HTML/JSON output.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class QualityReporter:
    """Generate data quality reports."""

    def __init__(self, df: DataFrame, name: str = "quality_report"):
        self.df = df
        self.name = name
        self._metrics: dict = {}

    def collect_metrics(self) -> dict:
        """Collect comprehensive quality metrics."""
        self._metrics = {
            "report_name": self.name,
            "generated_at": datetime.now().isoformat(),
            "summary": self._collect_summary(),
            "column_stats": self._collect_column_stats(),
            "null_analysis": self._collect_null_analysis(),
        }
        return self._metrics

    def _collect_summary(self) -> dict:
        return {
            "row_count": self.df.count(),
            "column_count": len(self.df.columns),
            "columns": self.df.columns,
        }

    def _collect_column_stats(self) -> dict:
        stats = {}
        for col in self.df.columns[:10]:  # Limit for performance
            dtype = dict(self.df.dtypes).get(col)
            col_stats = {"dtype": dtype}

            if dtype in ("int", "bigint", "double", "float"):
                agg_result = self.df.agg(
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                    F.avg(col).alias("avg"),
                    F.stddev(col).alias("stddev"),
                ).collect()[0]

                col_stats.update(
                    {
                        "min": agg_result["min"],
                        "max": agg_result["max"],
                        "avg": (
                            round(agg_result["avg"], 2) if agg_result["avg"] else None
                        ),
                        "stddev": (
                            round(agg_result["stddev"], 2)
                            if agg_result["stddev"]
                            else None
                        ),
                    }
                )
            elif dtype == "string":
                col_stats["distinct_count"] = self.df.select(col).distinct().count()

            stats[col] = col_stats
        return stats

    def _collect_null_analysis(self) -> dict:
        total = self.df.count()
        null_stats = {}

        for col in self.df.columns:
            null_count = self.df.filter(F.col(col).isNull()).count()
            null_stats[col] = {
                "null_count": null_count,
                "null_ratio": round(null_count / total, 4) if total > 0 else 0,
            }

        return null_stats

    def to_json(self, output_path: Optional[Path] = None) -> str:
        """Export report as JSON."""
        if not self._metrics:
            self.collect_metrics()

        json_str = json.dumps(self._metrics, indent=2, default=str)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"Saved JSON report to {output_path}")

        return json_str

    def to_html(self, output_path: Optional[Path] = None) -> str:
        """Export report as HTML."""
        if not self._metrics:
            self.collect_metrics()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Report: {self.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .passed {{ color: green; }} .failed {{ color: red; }}
    </style>
</head>
<body>
    <h1>Data Quality Report: {self.name}</h1>
    <p>Generated: {self._metrics.get('generated_at', 'N/A')}</p>
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Row Count</td><td>{self._metrics.get('summary', {}).get('row_count', 'N/A')}</td></tr>
        <tr><td>Column Count</td><td>{self._metrics.get('summary', {}).get('column_count', 'N/A')}</td></tr>
    </table>
    <h2>Column Statistics</h2>
    <table>
        <tr><th>Column</th><th>Type</th><th>Details</th></tr>
        {''.join(f"<tr><td>{col}</td><td>{stats.get('dtype')}</td><td>{stats}</td></tr>" for col, stats in self._metrics.get('column_stats', {}).items())}
    </table>
</body>
</html>"""

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"Saved HTML report to {output_path}")

        return html
