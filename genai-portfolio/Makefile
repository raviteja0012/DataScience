# ==============================================================================
# GenAI Portfolio - Makefile
# Author: Ravi Potluru
#
# Single-command shortcuts for installing, testing, linting, and running demos
# across all three projects in the GenAI portfolio.
# ==============================================================================

# ---- Configuration -----------------------------------------------------------

# Project directories (relative to this Makefile)
BILLER_DIR      := biller-integration-simulator
PAYMENT_DIR     := payment-intelligence-agent
INTEGRATION_DIR := multi-source-data-integration

PROJECTS := $(BILLER_DIR) $(PAYMENT_DIR) $(INTEGRATION_DIR)

# Python interpreter -- override with `make PYTHON=python3.11 ...` if needed
PYTHON ?= python

# ANSI color codes for visual polish
BLUE   := \033[1;34m
GREEN  := \033[1;32m
RED    := \033[1;31m
YELLOW := \033[1;33m
RESET  := \033[0m

# ---- Phony targets -----------------------------------------------------------

.PHONY: help install test test-biller test-payment test-integration \
        coverage coverage-biller coverage-payment coverage-integration \
        lint format clean pre-commit-install pre-commit-run \
        demo-biller demo-payment demo-integration demo-all streamlit \
        docker-build-all docker-up docker-down

# Default target -- print help
.DEFAULT_GOAL := help

# ==============================================================================
# Help
# ==============================================================================

help:
	@printf "$(BLUE)============================================================$(RESET)\n"
	@printf "$(BLUE)  GenAI Portfolio - Available Make Targets$(RESET)\n"
	@printf "$(BLUE)============================================================$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)Setup$(RESET)\n"
	@printf "  $(GREEN)make install$(RESET)            Install dependencies for all 3 projects\n"
	@printf "\n"
	@printf "$(YELLOW)Testing$(RESET)\n"
	@printf "  $(GREEN)make test$(RESET)               Run all 240 tests across 3 projects\n"
	@printf "  $(GREEN)make test-biller$(RESET)        Run biller-integration-simulator tests (53)\n"
	@printf "  $(GREEN)make test-payment$(RESET)       Run payment-intelligence-agent tests (101)\n"
	@printf "  $(GREEN)make test-integration$(RESET)   Run multi-source-data-integration tests (86)\n"
	@printf "  $(GREEN)make coverage$(RESET)            Run all tests with coverage report\n"
	@printf "\n"
	@printf "$(YELLOW)Code Quality$(RESET)\n"
	@printf "  $(GREEN)make lint$(RESET)               Run flake8 across all genai-portfolio code\n"
	@printf "  $(GREEN)make format$(RESET)             Run black + isort (if installed)\n"
	@printf "  $(GREEN)make pre-commit-install$(RESET) Install pre-commit hooks\n"
	@printf "  $(GREEN)make pre-commit-run$(RESET)     Run pre-commit on all files\n"
	@printf "  $(GREEN)make clean$(RESET)              Remove __pycache__, .pytest_cache, *.pyc, output/\n"
	@printf "\n"
	@printf "$(YELLOW)Demos$(RESET)\n"
	@printf "  $(GREEN)make demo-biller$(RESET)        Run biller integration demo\n"
	@printf "  $(GREEN)make demo-payment$(RESET)       Run payment intelligence agent CLI demos\n"
	@printf "  $(GREEN)make demo-integration$(RESET)   Run multi-source integration demo\n"
	@printf "  $(GREEN)make demo-all$(RESET)           Run all demos sequentially\n"
	@printf "  $(GREEN)make streamlit$(RESET)          Launch the payment intelligence Streamlit app\n"
	@printf "\n"
	@printf "$(YELLOW)Docker$(RESET)\n"
	@printf "  $(GREEN)make docker-build-all$(RESET)   Build Docker images for all 3 projects\n"
	@printf "  $(GREEN)make docker-up$(RESET)          Start docker-compose stacks\n"
	@printf "  $(GREEN)make docker-down$(RESET)        Stop docker-compose stacks\n"
	@printf "\n"

# ==============================================================================
# Setup
# ==============================================================================

install:
	@printf "$(BLUE)>> Installing dependencies for all projects...$(RESET)\n"
	@for proj in $(PROJECTS); do \
		printf "$(YELLOW)-- Installing $$proj...$(RESET)\n"; \
		(cd $$proj && $(PYTHON) -m pip install -r requirements.txt && $(PYTHON) -m pip install -e .) \
			|| { printf "$(RED)!! Install failed for $$proj$(RESET)\n"; exit 1; }; \
	done
	@printf "$(GREEN)>> All dependencies installed.$(RESET)\n"

# ==============================================================================
# Testing
# ==============================================================================

test:
	@printf "$(BLUE)============================================================$(RESET)\n"
	@printf "$(BLUE)  Running all 240 tests across 3 projects$(RESET)\n"
	@printf "$(BLUE)============================================================$(RESET)\n"
	@total_pass=0; total_fail=0; failed_projects=""; \
	for proj in $(PROJECTS); do \
		printf "\n$(YELLOW)>> Testing $$proj...$(RESET)\n"; \
		if (cd $$proj && $(PYTHON) -m pytest tests/ --tb=short -q); then \
			printf "$(GREEN)   [PASS] $$proj$(RESET)\n"; \
			total_pass=$$((total_pass + 1)); \
		else \
			printf "$(RED)   [FAIL] $$proj$(RESET)\n"; \
			total_fail=$$((total_fail + 1)); \
			failed_projects="$$failed_projects $$proj"; \
		fi; \
	done; \
	printf "\n$(BLUE)============================================================$(RESET)\n"; \
	printf "$(BLUE)  Summary$(RESET)\n"; \
	printf "$(BLUE)============================================================$(RESET)\n"; \
	printf "$(GREEN)  Projects passed: $$total_pass$(RESET)\n"; \
	printf "$(RED)  Projects failed: $$total_fail$(RESET)\n"; \
	if [ -n "$$failed_projects" ]; then \
		printf "$(RED)  Failed:$$failed_projects$(RESET)\n"; \
		exit 1; \
	fi

test-biller:
	@printf "$(BLUE)>> Running biller-integration-simulator tests...$(RESET)\n"
	@cd $(BILLER_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

test-payment:
	@printf "$(BLUE)>> Running payment-intelligence-agent tests...$(RESET)\n"
	@cd $(PAYMENT_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

test-integration:
	@printf "$(BLUE)>> Running multi-source-data-integration tests...$(RESET)\n"
	@cd $(INTEGRATION_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

# ==============================================================================
# Coverage
# ==============================================================================

coverage:
	@printf "$(BLUE)============================================================$(RESET)\n"
	@printf "$(BLUE)  Running all tests with coverage$(RESET)\n"
	@printf "$(BLUE)============================================================$(RESET)\n"
	@for proj in $(PROJECTS); do \
		printf "\n$(YELLOW)>> Coverage for $$proj...$(RESET)\n"; \
		(cd $$proj && $(PYTHON) -m pytest tests/ --tb=short -q \
			--cov=src --cov-report=term-missing --cov-report=xml:coverage.xml) \
		|| printf "$(RED)   [FAIL] $$proj$(RESET)\n"; \
	done
	@printf "\n$(GREEN)>> Coverage reports generated.$(RESET)\n"

coverage-biller:
	@printf "$(BLUE)>> Coverage for biller-integration-simulator...$(RESET)\n"
	@cd $(BILLER_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=src --cov-report=term-missing --cov-report=html:htmlcov

coverage-payment:
	@printf "$(BLUE)>> Coverage for payment-intelligence-agent...$(RESET)\n"
	@cd $(PAYMENT_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=src --cov-report=term-missing --cov-report=html:htmlcov

coverage-integration:
	@printf "$(BLUE)>> Coverage for multi-source-data-integration...$(RESET)\n"
	@cd $(INTEGRATION_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=src --cov-report=term-missing --cov-report=html:htmlcov

# ==============================================================================
# Code quality
# ==============================================================================

lint:
	@printf "$(BLUE)>> Running flake8 across genai-portfolio...$(RESET)\n"
	@if ! command -v flake8 >/dev/null 2>&1; then \
		printf "$(RED)!! flake8 not installed. Run: pip install flake8$(RESET)\n"; exit 1; \
	fi
	@flake8 . \
		--ignore=E501,W503,E203,E266,E402 \
		--max-line-length=120 \
		--statistics \
		--count \
		--show-source \
		--exclude=.git,__pycache__,build,dist,*.egg-info,.pytest_cache,output,venv,.venv \
		&& printf "$(GREEN)>> Lint clean.$(RESET)\n"

format:
	@printf "$(BLUE)>> Formatting code with black + isort...$(RESET)\n"
	@if command -v black >/dev/null 2>&1; then \
		black . --line-length=120 --exclude='/(\.git|__pycache__|build|dist|\.pytest_cache|output|venv|\.venv)/'; \
	else \
		printf "$(YELLOW)!! black not installed -- skipping. Install with: pip install black$(RESET)\n"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort . --profile=black --line-length=120 \
			--skip=.git --skip=__pycache__ --skip=build --skip=dist \
			--skip=.pytest_cache --skip=output --skip=venv --skip=.venv; \
	else \
		printf "$(YELLOW)!! isort not installed -- skipping. Install with: pip install isort$(RESET)\n"; \
	fi
	@printf "$(GREEN)>> Format step complete.$(RESET)\n"

clean:
	@printf "$(BLUE)>> Cleaning generated artifacts...$(RESET)\n"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@for proj in $(PROJECTS); do \
		if [ -d "$$proj/output" ]; then \
			rm -rf "$$proj/output"; \
			printf "   removed $$proj/output\n"; \
		fi; \
	done
	@printf "$(GREEN)>> Clean complete.$(RESET)\n"

pre-commit-install:
	@printf "$(BLUE)>> Installing pre-commit hooks...$(RESET)\n"
	@if ! command -v pre-commit >/dev/null 2>&1; then \
		printf "$(YELLOW)-- Installing pre-commit...$(RESET)\n"; \
		$(PYTHON) -m pip install pre-commit; \
	fi
	@pre-commit install
	@printf "$(GREEN)>> Pre-commit hooks installed.$(RESET)\n"

pre-commit-run:
	@printf "$(BLUE)>> Running pre-commit on all files...$(RESET)\n"
	@pre-commit run --all-files || true
	@printf "$(GREEN)>> Pre-commit run complete.$(RESET)\n"

# ==============================================================================
# Demos
# ==============================================================================

demo-biller:
	@printf "$(BLUE)>> Running biller integration demo...$(RESET)\n"
	@cd $(BILLER_DIR) && $(PYTHON) examples/onboard_new_biller.py
	@cd $(BILLER_DIR) && $(PYTHON) examples/process_batch_payments.py
	@cd $(BILLER_DIR) && $(PYTHON) examples/run_settlement.py
	@printf "$(GREEN)>> Biller demo complete.$(RESET)\n"

demo-payment:
	@printf "$(BLUE)>> Running payment intelligence agent CLI demos...$(RESET)\n"
	@cd $(PAYMENT_DIR) && $(PYTHON) demo/demo_analytics.py
	@cd $(PAYMENT_DIR) && $(PYTHON) demo/demo_anomaly.py
	@cd $(PAYMENT_DIR) && $(PYTHON) demo/demo_rag.py
	@printf "$(GREEN)>> Payment intelligence demos complete.$(RESET)\n"

demo-integration:
	@printf "$(BLUE)>> Running multi-source integration demo...$(RESET)\n"
	@cd $(INTEGRATION_DIR) && $(PYTHON) examples/run_full_pipeline.py
	@printf "$(GREEN)>> Multi-source integration demo complete.$(RESET)\n"

demo-all: demo-biller demo-payment demo-integration
	@printf "$(GREEN)>> All demos complete.$(RESET)\n"

streamlit:
	@printf "$(BLUE)>> Launching payment intelligence Streamlit app...$(RESET)\n"
	@if ! command -v streamlit >/dev/null 2>&1; then \
		printf "$(RED)!! streamlit not installed. Run: pip install streamlit$(RESET)\n"; exit 1; \
	fi
	@cd $(PAYMENT_DIR) && streamlit run src/app.py

# ==============================================================================
# Docker
# ==============================================================================

docker-build-all:
	@printf "$(BLUE)>> Building Docker images for all projects...$(RESET)\n"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/Dockerfile" ]; then \
			printf "$(YELLOW)-- Building $$proj...$(RESET)\n"; \
			docker build -t genai-portfolio/$$proj:latest $$proj \
				|| { printf "$(RED)!! Docker build failed for $$proj$(RESET)\n"; exit 1; }; \
		else \
			printf "$(YELLOW)-- Skipping $$proj (no Dockerfile)$(RESET)\n"; \
		fi; \
	done
	@printf "$(GREEN)>> All Docker images built.$(RESET)\n"

docker-up:
	@printf "$(BLUE)>> Starting docker-compose stacks...$(RESET)\n"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/docker-compose.yml" ]; then \
			printf "$(YELLOW)-- Starting $$proj...$(RESET)\n"; \
			(cd $$proj && docker compose up -d) \
				|| { printf "$(RED)!! docker-compose up failed for $$proj$(RESET)\n"; exit 1; }; \
		fi; \
	done
	@printf "$(GREEN)>> docker-compose stacks running.$(RESET)\n"

docker-down:
	@printf "$(BLUE)>> Stopping docker-compose stacks...$(RESET)\n"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/docker-compose.yml" ]; then \
			printf "$(YELLOW)-- Stopping $$proj...$(RESET)\n"; \
			(cd $$proj && docker compose down) || true; \
		fi; \
	done
	@printf "$(GREEN)>> docker-compose stacks stopped.$(RESET)\n"
