PYTHON ?= python3

.PHONY: install install-dev lint format type-check test test-cov clean help run run-ml run-baseline run-vol run-meanvar run-riskparity report run-config demo

# Default target
help:
	@echo "Trading System MVP - Available commands:"
	@echo ""
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install all dependencies (including dev)"
	@echo "  make lint         Run linters (ruff)"
	@echo "  make format       Format code with black"
	@echo "  make format-check Check formatting without changes"
	@echo "  make type-check   Run mypy type checker"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make check        Run all checks (format, lint, type-check, test)"
	@echo "  make run          Run backtest (default config)"
	@echo "  make run-baseline Run baseline (fixed sizing, no risk)"
	@echo "  make run-vol      Run volatility sizing + risk management"
	@echo "  make run-meanvar  Run mean-variance optimized portfolio"
	@echo "  make run-ml       Run ML strategy backtest (XGBoost)"
	@echo "  make run-riskparity Run risk parity optimized portfolio"
	@echo "  make run-config CONFIG=path  Run backtest with a custom config file"
	@echo "  make report       Run ALL configs and generate full comparison report"
	@echo "  make clean        Remove cache files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black ruff mypy pytest-cov

# Linting
lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

# Formatting
format:
	black src/ tests/

format-check:
	black --check src/ tests/

# Type checking
type-check:
	mypy src/ --ignore-missing-imports

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -x -q

# Run all checks
check: format-check lint type-check test

# CI simulation (same as GitHub Actions)
ci: format-check lint type-check test

# Clean up
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf src/**/__pycache__
	rm -rf .coverage htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Download sample data
download-data:
	$(PYTHON) scripts/download_data.py --symbols AAPL MSFT GOOGL

# Run backtest (downloads data first if needed)
run: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/backtest_config.yaml

# Run individual Phase 2 configs
run-baseline: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/backtest_baseline.yaml

run-vol: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/backtest_phase2_vol.yaml

run-meanvar: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/backtest_phase2_meanvar.yaml

run-riskparity: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/backtest_phase2_riskparity.yaml

# Run ML strategy backtest
run-ml: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/ml_backtest_config.yaml

# Run with a custom config file: make run-config CONFIG=configs/my_config.yaml
run-config: download-data
	@if [ -z "$(CONFIG)" ]; then echo "Usage: make run-config CONFIG=path/to/config.yaml"; exit 1; fi
	$(PYTHON) ./scripts/run_backtest.py --config $(CONFIG)

# Full report: run ALL configs, generate report + charts
report: download-data
	$(PYTHON) ./scripts/run_full_report.py
