# MuseQuill.ink Makefile
# Common development tasks and commands (Conda-based)

.PHONY: help install install-dev install-prod clean test lint format check setup run docs conda-check

# Conda environment name
CONDA_ENV = musequill
CONDA_PYTHON = conda run -n $(CONDA_ENV) python

# Default target
help:
	@echo "MuseQuill.ink Development Commands (Conda)"
	@echo "=========================================="
	@echo ""
	@echo# MuseQuill.ink Makefile
# Common development tasks and commands

.PHONY: help install install-dev install-prod clean test lint format check setup run docs

# Default target
help:
	@echo "MuseQuill.ink Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      - Install base dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  install-prod - Install production dependencies"
	@echo "  setup        - Complete development setup"
	@echo ""
	@echo "Development Commands:"
	@echo "  run          - Run the application in config mode"
	@echo "  run-api      - Run the API server"
	@echo "  test         - Run all tests"
	@echo "  test-fast    - Run tests without slow tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  check        - Run all quality checks"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean        - Clean up temporary files"
	@echo "  docs         - Build documentation"
	@echo "  requirements - Update requirements files"

# Installation targets
install:
	python install.py --mode base

install-dev:
	python install.py --mode dev

install-prod:
	python install.py --mode prod --no-venv

setup: install-dev
	@echo "🎉 Development environment ready!"

# Development targets
run:
	python main.py --mode config --verbose

run-api:
	python main.py --mode api --host 127.0.0.1 --port 8000 --reload

run-dry:
	python main.py --mode api --dry-run --verbose

# Testing targets
test:
	python test_foundation.py
	python verify_main.py

test-fast:
	python test_foundation.py

test-verbose:
	python test_foundation.py
	python verify_main.py
	python main.py --mode config --verbose

# Code quality targets
lint:
	@echo "🔍 Running linting checks..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check musequill/; \
	else \
		echo "⚠️  ruff not installed, skipping lint"; \
	fi

format:
	@echo "🎨 Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black musequill/ main.py install.py test_foundation.py verify_main.py; \
	else \
		echo "⚠️  black not installed, skipping format"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort musequill/ main.py install.py test_foundation.py verify_main.py; \
	else \
		echo "⚠️  isort not installed, skipping import sorting"; \
	fi

type-check:
	@echo "🔍 Running type checks..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy musequill/ || true; \
	else \
		echo "⚠️  mypy not installed, skipping type check"; \
	fi

check: lint type-check test
	@echo "✅ All quality checks completed"

# Maintenance targets
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf logs/*.log
	@echo "✅ Cleanup completed"

docs:
	@echo "📚 Building documentation..."
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs build; \
	else \
		echo "⚠️  mkdocs not installed, skipping docs build"; \
	fi

# Requirements management
requirements:
	@echo "📦 Updating requirements..."
	pip freeze > requirements-frozen.txt
	@echo "✅ Requirements frozen to requirements-frozen.txt"

# Docker targets (if Docker is available)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t musequill:latest .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 --env-file .env musequill:latest

# Git hooks
install-hooks:
	@echo "🪝 Installing git hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "⚠️  pre-commit not installed, skipping hooks installation"; \
	fi

# Development server with auto-reload
dev-server:
	python main.py --mode api --reload --verbose

# Quick start for new developers
quickstart: setup test run
	@echo ""
	@echo "🚀 MuseQuill.ink Quick Start Complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update .env file with your API keys"
	@echo "2. Run 'make run-api' to start the development server"
	@echo "3. Visit http://localhost:8000/docs for API documentation"

# Show system information
info:
	@echo "System Information:"
	@echo "==================="
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Virtual Environment: $(VIRTUAL_ENV)"
	@echo ""
	@echo "MuseQuill Status:"
	@echo "=================="
	@python -c "import musequill; print(f'Version: {musequill.__version__}')" 2>/dev/null || echo "MuseQuill not installed"
	@echo "Configuration: $(shell python main.py --mode config 2>/dev/null | grep -o 'Environment: [^,]*' || echo 'Not configured')"