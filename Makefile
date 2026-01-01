.PHONY: install test lint format run clean

# Install dependencies
install:
	uv sync

# Run tests
test:
	uv run pytest tests/ -v --cov=src --cov-report=html

# Run unit tests only
test-unit:
	uv run pytest tests/unit/ -v

# Run integration tests only
test-integration:
	uv run pytest tests/integration/ -v

# Lint code
lint:
	uv run flake8 src/ tests/
	uv run black src/ tests/ --check

# Format code
format:
	uv run black src/ tests/

# Run the main pipeline
run:
	uv run python main.py

# Generate synthetic data
generate-data:
	uv run python -c "from src.core.generators import TransactionGenerator; TransactionGenerator(10000, seed=42).to_csv('data/raw/transactions.csv')"

# Clean build artifacts
clean:
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf data/processed/
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Show help
help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies with uv"
	@echo "  make test           - Run all tests with coverage"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint           - Check code style"
	@echo "  make format         - Format code with black"
	@echo "  make run            - Run the ETL pipeline"
	@echo "  make generate-data  - Generate synthetic test data"
	@echo "  make clean          - Remove build artifacts"
