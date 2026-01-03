"""
Streaming Data Simulator.
Simulates streaming data patterns for testing without Kafka.
"""

from __future__ import annotations
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Generator, Optional
from loguru import logger
from src.core.generators import TransactionGenerator


class StreamingSimulator:
    """Simulate streaming data by writing to files at intervals."""

    def __init__(
        self,
        output_dir: Path,
        records_per_batch: int = 100,
        interval_seconds: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.records_per_batch = records_per_batch
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._batch_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self, num_batches: Optional[int] = None) -> None:
        """Start streaming simulation in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(num_batches,))
        self._thread.daemon = True
        self._thread.start()
        logger.info(
            f"Started streaming simulator: {self.records_per_batch} records every {self.interval}s"
        )

    def stop(self) -> None:
        """Stop the streaming simulation."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info(f"Stopped streaming simulator. Total batches: {self._batch_count}")

    def _run(self, num_batches: Optional[int] = None) -> None:
        """Internal run loop."""
        while self._running and (
            num_batches is None or self._batch_count < num_batches
        ):
            self._write_batch()
            time.sleep(self.interval)

    def _write_batch(self) -> None:
        """Write a batch of records."""
        import csv

        generator = TransactionGenerator(num_records=self.records_per_batch)
        records = list(generator)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"batch_{timestamp}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        self._batch_count += 1
        logger.debug(f"Wrote batch {self._batch_count}: {filename}")


class MicroBatchProcessor:
    """Process micro-batches from a directory."""

    def __init__(self, input_dir: Path, processed_dir: Optional[Path] = None):
        self.input_dir = Path(input_dir)
        self.processed_dir = (
            Path(processed_dir) if processed_dir else input_dir / "_processed"
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_pending_files(self) -> list[Path]:
        """Get list of unprocessed files."""
        return sorted(self.input_dir.glob("batch_*.csv"))

    def process_next_batch(self) -> Optional[Path]:
        """Process the oldest pending batch file."""
        pending = self.get_pending_files()
        if not pending:
            return None

        file_path = pending[0]
        logger.info(f"Processing: {file_path.name}")

        # Move to processed directory
        target = self.processed_dir / file_path.name
        file_path.rename(target)

        return target
