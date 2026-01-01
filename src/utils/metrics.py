"""Metrics collection module."""

from __future__ import annotations
import time
from datetime import datetime
from typing import Optional
from loguru import logger


class MetricsCollector:
    """Collect and report execution metrics."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._start_time: Optional[float] = None
        self._metrics: dict = {}
        self._timers: dict[str, float] = {}
    
    def start(self) -> None:
        """Start metrics collection."""
        self._start_time = time.perf_counter()
        self._metrics["start_time"] = datetime.now().isoformat()
    
    def stop(self) -> None:
        """Stop metrics collection."""
        if self._start_time:
            self._metrics["duration_seconds"] = time.perf_counter() - self._start_time
        self._metrics["end_time"] = datetime.now().isoformat()
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name in self._timers:
            elapsed = time.perf_counter() - self._timers[name]
            self._metrics[f"timer_{name}"] = elapsed
            del self._timers[name]
            return elapsed
        return 0.0
    
    def record(self, key: str, value) -> None:
        """Record a metric value."""
        self._metrics[key] = value
    
    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self._metrics[key] = self._metrics.get(key, 0) + amount
    
    def get_metrics(self) -> dict:
        """Get all collected metrics."""
        return {"name": self.name, **self._metrics}
    
    def log_summary(self) -> None:
        """Log a summary of collected metrics."""
        logger.info(f"Metrics Summary for '{self.name}':")
        for key, value in self._metrics.items():
            logger.info(f"  {key}: {value}")
