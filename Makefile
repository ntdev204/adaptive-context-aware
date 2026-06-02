PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose -f docker/docker-compose.yml
CODEBASE_ENGINE_DIR ?= $(CURDIR)/models/engines
CONTAINER_CODEBASE_ENGINE_DIR ?= /workspace/models/engines
ENGINE_DYNAMIC ?= true
ENGINE_BATCH ?= 2
ENGINE_WORKSPACE_GB ?= 2
ENGINE_FP16 ?= true
ENGINE_INT8 ?= false
ENGINE_DATA ?=
ENGINE_IMGSZ ?= 480 640

.PHONY: help install install-dev test test-unit lint fixtures fixtures-download benchmark-ci benchmark-laptop benchmark-jetson baseline-update \
	build-dev build-prod export-engine up down logs config \
	docker-build-dev docker-build-prod build-engine compose-config compose-up compose-down compose-logs run clean

help:
	@echo "Available targets:"
	@echo "  install            Install runtime dependencies"
	@echo "  install-dev        Install development dependencies"
	@echo "  test               Run all unit tests"
	@echo "  test-unit          Run unit tests"
	@echo "  lint               Run ruff checks"
	@echo "  fixtures           Generate synthetic fixtures"
	@echo "  fixtures-download  Create/download external fixture directories"
	@echo "  benchmark-ci       Run CI benchmark baseline comparison"
	@echo "  benchmark-laptop   Run laptop benchmark flow"
	@echo "  baseline-update    Refresh CI baselines"
	@echo "  docker-build-dev   Build dev runtime image and MLflow image"
	@echo "  docker-build-prod  Build prod runtime image and MLflow image"
	@echo "  export-engine      Export all .pt models under models/ → TensorRT .engine using context-aware:jetson-dev"
	@echo "  compose-up         Start Jetson adaptive runtime and MLflow"
	@echo "  compose-down       Stop Jetson adaptive runtime and MLflow"
	@echo "  compose-logs       Follow Jetson adaptive runtime logs"
	@echo "  compose-config     Validate docker compose config"
	@echo "  build-dev          Build adaptive-context-aware development services"
	@echo "  build-prod         Build adaptive-context-aware production services"
	@echo "  mlops-up           Start MLflow tracking service"
	@echo "  mlops-logs         Follow MLflow logs"
	@echo "  up                 Alias: compose-up"
	@echo "  down               Alias: compose-down"
	@echo "  logs               Alias: compose-logs"
	@echo "  config             Alias: compose-config"
	@echo "  build-engine       Build dev image and export .engine artifacts into models/engines"
	@echo "  run                Run application entrypoint locally"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]
	$(PIP) install ruff

test: test-unit

test-unit:
	$(PYTEST) tests/unit -q

lint:
	ruff check src tests scripts

fixtures:
	$(PYTHON) scripts/generate_synthetic_fixtures.py

fixtures-download:
	$(PYTHON) scripts/download_fixtures.py

benchmark-ci:
	$(PYTHON) scripts/benchmark.py --device ci --compare-baseline

benchmark-laptop:
	$(PYTHON) scripts/benchmark.py --device laptop --frames 1000

benchmark-jetson: benchmark-laptop

baseline-update:
	$(PYTHON) scripts/update_ci_baselines.py --source local --output tests/benchmark/baselines

docker-build-dev:
	$(DOCKER) build -f docker/Dockerfile.jetson --target jetson-dev -t context-aware:jetson-dev .
	$(DOCKER) build -f docker/Dockerfile.jetson --target mlflow -t context-aware:mlflow .

docker-build-prod:
	$(DOCKER) build -f docker/Dockerfile.jetson --target jetson-prod -t context-aware:jetson-prod .
	$(DOCKER) build -f docker/Dockerfile.jetson --target mlflow -t context-aware:mlflow .

build-dev: docker-build-dev
	$(COMPOSE) build jetson-dev

build-prod: docker-build-prod
	$(COMPOSE) build jetson-prod

build-engine: docker-build-dev export-engine

export-engine:
	@$(DOCKER) image inspect context-aware:jetson-dev > /dev/null 2>&1 || (echo "context-aware:jetson-dev not found; run 'make docker-build-dev' first." >&2; exit 1)
	$(PYTHON) -c "from pathlib import Path; Path('models/engines').mkdir(parents=True, exist_ok=True)"
	$(DOCKER) run --rm --gpus all \
		-v "$(CURDIR)/models:/app/models" \
		-v "$(CURDIR)/data:/app/data:ro" \
		-v "$(CODEBASE_ENGINE_DIR):$(CONTAINER_CODEBASE_ENGINE_DIR)" \
		-e ENGINE_DYNAMIC="$(ENGINE_DYNAMIC)" \
		-e ENGINE_BATCH="$(ENGINE_BATCH)" \
		-e ENGINE_WORKSPACE_GB="$(ENGINE_WORKSPACE_GB)" \
		-e ENGINE_FP16="$(ENGINE_FP16)" \
		-e ENGINE_INT8="$(ENGINE_INT8)" \
		-e ENGINE_DATA="$(ENGINE_DATA)" \
		-e ENGINE_IMGSZ="$(ENGINE_IMGSZ)" \
		context-aware:jetson-dev python3 scripts/export_engine.py \
			--root /app/models \
			--output-dir /app/models/engines \
			--codebase-output-dir $(CONTAINER_CODEBASE_ENGINE_DIR)

compose-config:
	$(COMPOSE) config

compose-up:
	$(COMPOSE) --profile mlops up -d jetson-prod mlflow

compose-down:
	$(COMPOSE) --profile mlops down

compose-logs:
	$(COMPOSE) logs -f jetson-prod mlflow

mlops-up:
	$(COMPOSE) --profile mlops up -d mlflow

mlops-logs:
	$(COMPOSE) logs -f mlflow

config: compose-config

up: compose-up

down: compose-down

logs: compose-logs

run:
	$(PYTHON) -m src.main

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('benchmarks').glob('laptop_report_*.json')] if Path('benchmarks').exists() else None"
