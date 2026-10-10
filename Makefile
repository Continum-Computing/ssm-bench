.PHONY: setup test lint format clean

# Python executable (use python3 on Linux, python on Windows)
PYTHON ?= python
VENV := .venv
BIN := $(VENV)/bin

# Default target
all: test

# Create virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,profiling]"
	@echo "Virtual environment created in $(VENV)/"
	@echo "Activate with: source $(VENV)/bin/activate"

# Run correctness tests
test:
	$(BIN)/pytest

# Run linters
lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

# Format code
format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

# Clean up build artifacts and virtual environment
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ .pytest_cache/ .coverage
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "Available targets:"
	@echo "  setup   - Create virtual environment and install dependencies"
	@echo "  test    - Run correctness tests"
	@echo "  lint    - Run linters (ruff check and format check)"
	@echo "  format  - Format code with ruff"
	@echo "  clean   - Remove virtual environment and build artifacts"
	@echo "  help    - Show this help message"
