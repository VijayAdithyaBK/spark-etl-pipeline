"""
Advanced Python Generators Module.

Demonstrates:
- Generator functions with yield
- Generator expressions
- Iterator protocol implementation
- Lazy evaluation patterns
- Memory-efficient data processing
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
)

from loguru import logger
from pyspark.sql import DataFrame

T = TypeVar("T")


def infinite_counter(start: int = 0, step: int = 1) -> Generator[int, None, None]:
    """
    Generate an infinite sequence of integers.

    Demonstrates the simplest form of a generator function.

    Args:
        start: Starting value.
        step: Increment step.

    Yields:
        Sequential integers.

    Example:
        >>> counter = infinite_counter(0, 2)
        >>> next(counter)  # 0
        >>> next(counter)  # 2
        >>> next(counter)  # 4
    """
    current = start
    while True:
        yield current
        current += step


def chunked(iterable: Iterable[T], size: int) -> Generator[list[T], None, None]:
    """
    Split an iterable into chunks of specified size.

    Memory-efficient chunking that doesn't load entire iterable into memory.

    Args:
        iterable: Any iterable to chunk.
        size: Maximum chunk size.

    Yields:
        Lists of at most `size` elements.

    Example:
        >>> list(chunked([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive")

    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []

    if chunk:  # Yield remaining items
        yield chunk


def lazy_file_reader(
    file_path: Path | str,
    encoding: str = "utf-8",
    skip_header: bool = False,
    skip_empty: bool = True,
) -> Generator[str, None, None]:
    """
    Memory-efficient file reader that yields lines lazily.

    Unlike reading entire file into memory, this generator yields
    one line at a time, enabling processing of files larger than RAM.

    Args:
        file_path: Path to the file.
        encoding: File encoding.
        skip_header: Skip the first line.
        skip_empty: Skip empty lines.

    Yields:
        File lines as strings (without newline characters).

    Example:
        >>> for line in lazy_file_reader("large_file.txt"):
        ...     process(line)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.debug(f"Opening file for lazy reading: {path}")

    with open(path, "r", encoding=encoding) as f:
        if skip_header:
            next(f, None)

        for line in f:
            stripped = line.rstrip("\n\r")

            if skip_empty and not stripped:
                continue

            yield stripped


class TransactionGenerator:
    """
    Generator class for synthetic financial transaction data.

    Demonstrates:
    - Iterator protocol (__iter__, __next__)
    - Configurable data generation
    - Realistic transaction patterns

    This generates synthetic transaction data suitable for testing
    ETL pipelines without needing real financial data.
    """

    # Sample data for generation
    MERCHANT_CATEGORIES = [
        "grocery",
        "gas_station",
        "restaurant",
        "online_shopping",
        "electronics",
        "travel",
        "entertainment",
        "healthcare",
        "utilities",
        "clothing",
        "subscription",
        "cash_withdrawal",
    ]

    TRANSACTION_TYPES = ["purchase", "refund", "transfer", "payment"]

    CITIES = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ]

    def __init__(
        self,
        num_records: int = 1000,
        num_customers: int = 100,
        num_merchants: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the transaction generator.

        Args:
            num_records: Number of transactions to generate.
            num_customers: Number of unique customers.
            num_merchants: Number of unique merchants.
            start_date: Start date range for transactions.
            end_date: End date range for transactions.
            seed: Random seed for reproducibility.
        """
        self.num_records = num_records
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.start_date = start_date or datetime(2023, 1, 1)
        self.end_date = end_date or datetime(2024, 12, 31)

        if seed is not None:
            random.seed(seed)

        self._current = 0
        self._customer_ids = [f"CUST_{i:06d}" for i in range(num_customers)]
        self._merchant_ids = [f"MERCH_{i:05d}" for i in range(num_merchants)]

        logger.info(
            f"TransactionGenerator initialized: {num_records} records, "
            f"{num_customers} customers, {num_merchants} merchants"
        )

    def __iter__(self) -> "TransactionGenerator":
        """Return iterator (self)."""
        self._current = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Generate next transaction record."""
        if self._current >= self.num_records:
            raise StopIteration

        self._current += 1
        return self._generate_transaction()

    def __len__(self) -> int:
        """Return number of records to generate."""
        return self.num_records

    def _generate_transaction(self) -> dict[str, Any]:
        """Generate a single synthetic transaction."""
        # Random timestamp within range
        time_delta = self.end_date - self.start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        transaction_time = self.start_date + timedelta(seconds=random_seconds)

        # Transaction amount based on category
        category = random.choice(self.MERCHANT_CATEGORIES)
        amount = self._generate_amount(category)

        # Determine if transaction is fraudulent (small percentage)
        is_fraud = random.random() < 0.012  # ~1.2% fraud rate

        if is_fraud:
            # Fraudulent transactions tend to be higher value or unusual
            amount *= random.uniform(2.0, 10.0)

        transaction = {
            "transaction_id": f"TXN_{self._current:010d}",
            "customer_id": random.choice(self._customer_ids),
            "merchant_id": random.choice(self._merchant_ids),
            "merchant_category": category,
            "transaction_type": random.choice(self.TRANSACTION_TYPES),
            "amount": round(amount, 2),
            "currency": "USD",
            "city": random.choice(self.CITIES),
            "transaction_time": transaction_time.isoformat(),
            "is_fraud": int(is_fraud),
            "card_type": random.choice(["credit", "debit"]),
            "device_type": random.choice(["mobile", "desktop", "pos_terminal"]),
        }

        return transaction

    def _generate_amount(self, category: str) -> float:
        """Generate realistic amount based on category."""
        amount_ranges = {
            "grocery": (10, 200),
            "gas_station": (20, 80),
            "restaurant": (15, 150),
            "online_shopping": (20, 500),
            "electronics": (50, 2000),
            "travel": (100, 5000),
            "entertainment": (10, 200),
            "healthcare": (20, 500),
            "utilities": (50, 300),
            "clothing": (30, 400),
            "subscription": (5, 50),
            "cash_withdrawal": (20, 500),
        }

        min_amt, max_amt = amount_ranges.get(category, (10, 100))
        return random.uniform(min_amt, max_amt)

    def to_list(self) -> list[dict[str, Any]]:
        """Convert generator to list (consumes generator)."""
        return list(self)

    def to_csv(self, output_path: Path | str) -> None:
        """Write generated data to CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = list(self)

        if not records:
            logger.warning("No records to write")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Wrote {len(records)} records to {path}")


class BatchIterator(Generic[T]):
    """
    Iterator for processing data in batches.

    Demonstrates:
    - Generic typing
    - Batch processing patterns
    - Progress tracking

    Useful for processing large datasets in memory-efficient chunks.
    """

    def __init__(
        self,
        data: Iterable[T],
        batch_size: int = 1000,
        show_progress: bool = True,
    ):
        """
        Initialize batch iterator.

        Args:
            data: Source data iterable.
            batch_size: Number of items per batch.
            show_progress: Log progress messages.
        """
        self.data = data
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._batch_num = 0
        self._items_processed = 0

    def __iter__(self) -> Generator[list[T], None, None]:
        """Iterate over data in batches."""
        batch: list[T] = []

        for item in self.data:
            batch.append(item)

            if len(batch) >= self.batch_size:
                self._batch_num += 1
                self._items_processed += len(batch)

                if self.show_progress:
                    logger.info(
                        f"Processing batch {self._batch_num} "
                        f"({self._items_processed} items total)"
                    )

                yield batch
                batch = []

        # Yield remaining items
        if batch:
            self._batch_num += 1
            self._items_processed += len(batch)

            if self.show_progress:
                logger.info(
                    f"Processing final batch {self._batch_num} "
                    f"({self._items_processed} items total)"
                )

            yield batch

    @property
    def batches_processed(self) -> int:
        """Return number of batches processed so far."""
        return self._batch_num

    @property
    def items_processed(self) -> int:
        """Return number of items processed so far."""
        return self._items_processed


class DataFrameBatchIterator:
    """
    Iterator for processing Spark DataFrames in batches.

    Converts DataFrame partitions to Pandas for batch processing,
    useful for operations that require local computation.
    """

    def __init__(
        self,
        df: DataFrame,
        batch_size: int = 10000,
        columns: Optional[list[str]] = None,
    ):
        """
        Initialize DataFrame batch iterator.

        Args:
            df: Spark DataFrame to iterate.
            batch_size: Approximate rows per batch.
            columns: Columns to select (None for all).
        """
        self.df = df if columns is None else df.select(*columns)
        self.batch_size = batch_size
        self._batch_count = 0

    def __iter__(self) -> Generator[list[dict[str, Any]], None, None]:
        """Iterate DataFrame in batches using partitions."""
        # Use toLocalIterator for memory efficiency
        for partition in self.df.toLocalIterator():
            self._batch_count += 1
            yield partition.asDict()

    def collect_batches(self) -> Generator[list[dict[str, Any]], None, None]:
        """Collect DataFrame in chunked batches."""
        import pandas as pd

        # Convert to Pandas and iterate in chunks
        pandas_df = self.df.toPandas()

        for start in range(0, len(pandas_df), self.batch_size):
            end = min(start + self.batch_size, len(pandas_df))
            batch = pandas_df.iloc[start:end].to_dict("records")
            self._batch_count += 1
            yield batch


# Pipeline generator for composed transformations
def pipeline_stages(
    *stages: Callable[[T], T]
) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """
    Create a generator pipeline from multiple transformation stages.

    Each stage is a function that transforms a single item.

    Args:
        *stages: Transformation functions.

    Returns:
        A function that creates a generator applying all stages.

    Example:
        >>> transform = pipeline_stages(
        ...     lambda x: x.strip(),
        ...     lambda x: x.upper(),
        ...     lambda x: x.replace(" ", "_")
        ... )
        >>> list(transform(["  hello world  ", "  goodbye  "]))
        ['HELLO_WORLD', 'GOODBYE']
    """

    def apply_pipeline(data: Iterable[T]) -> Generator[T, None, None]:
        for item in data:
            result = item
            for stage in stages:
                result = stage(result)
            yield result

    return apply_pipeline


# Module testing
if __name__ == "__main__":
    # Test infinite counter
    counter = infinite_counter(0, 5)
    for _ in range(5):
        print(f"Counter: {next(counter)}")

    # Test chunked
    data = list(range(10))
    print(f"Chunked: {list(chunked(data, 3))}")

    # Test transaction generator
    generator = TransactionGenerator(num_records=5, seed=42)
    for txn in generator:
        print(f"Transaction: {txn['transaction_id']} - ${txn['amount']}")

    # Test batch iterator
    items = list(range(25))
    for batch in BatchIterator(items, batch_size=10, show_progress=True):
        print(f"Batch contents: {batch}")
