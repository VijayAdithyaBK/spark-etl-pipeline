"""Integration tests for the full ETL pipeline."""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.core.generators import TransactionGenerator


class TestETLPipelineIntegration:
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test data."""
        temp_dir = tempfile.mkdtemp()
        raw_dir = Path(temp_dir) / "raw"
        processed_dir = Path(temp_dir) / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        yield {"raw": raw_dir, "processed": processed_dir, "base": temp_dir}

        shutil.rmtree(temp_dir)

    def test_generator_creates_data(self, temp_dirs):
        """Test that transaction generator creates valid data."""
        generator = TransactionGenerator(num_records=100, seed=42)
        output_path = temp_dirs["raw"] / "transactions.csv"

        generator.to_csv(output_path)

        assert output_path.exists()

    def test_csv_extraction(self, spark, temp_dirs):
        """Test CSV extraction from generated data."""
        from src.etl.extractors import CSVExtractor

        # Generate data
        generator = TransactionGenerator(num_records=50, seed=42)
        output_path = temp_dirs["raw"] / "test_transactions.csv"
        generator.to_csv(output_path)

        # Extract
        extractor = CSVExtractor(spark)
        df = extractor.extract(str(output_path))

        assert df.count() == 50
        assert "transaction_id" in df.columns

    def test_transformation_chain(self, spark, temp_dirs):
        """Test transformation chain on extracted data."""
        from src.etl.extractors import CSVExtractor
        from src.etl.transformers import (
            TransformerChain,
            CleansingTransformer,
            AmountTransformer,
        )

        # Generate and extract
        generator = TransactionGenerator(num_records=100, seed=42)
        output_path = temp_dirs["raw"] / "transform_test.csv"
        generator.to_csv(output_path)

        extractor = CSVExtractor(spark)
        df = extractor.extract(str(output_path))

        # Transform
        chain = TransformerChain()
        chain.add(CleansingTransformer())
        chain.add(AmountTransformer(amount_column="amount"))

        result = chain.execute(df)

        assert "amount_category" in result.columns
        assert result.count() == 100

    def test_quality_validation(self, spark, temp_dirs):
        """Test quality validation on processed data."""
        from src.etl.extractors import CSVExtractor
        from src.quality.validators import NullValidator, DuplicateValidator

        # Generate and extract
        generator = TransactionGenerator(num_records=50, seed=42)
        output_path = temp_dirs["raw"] / "quality_test.csv"
        generator.to_csv(output_path)

        extractor = CSVExtractor(spark)
        df = extractor.extract(str(output_path))

        # Validate
        null_validator = NullValidator(columns=["transaction_id"], max_null_ratio=0.0)
        passed, details = null_validator.validate(df)

        assert passed

        dup_validator = DuplicateValidator(key_columns=["transaction_id"])
        passed, details = dup_validator.validate(df)

        assert passed
