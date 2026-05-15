PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose -f docker/docker-compose.yml

.PHONY: help install install-dev test test-unit lint fixtures fixtures-download benchmark-ci benchmark-laptop benchmark-jetson baseline-update \
	docker-build-dev docker-build-test docker-build-prod build-engine export-engine compose-config compose-up compose-down compose-logs run clean

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
	@echo "  docker-build-dev   Build Jetson-native development image"
	@echo "  docker-build-test  Build test image"
	@echo "  docker-build-prod  Build Jetson-native production image"
	@echo "  build-engine       Build engine-export image"
	@echo "  export-engine      Export all .pt models under models/ → TensorRT .engine on target Jetson"
	@echo "  compose-config     Validate docker compose config"
	@echo "  compose-up         Start compose services"
	@echo "  compose-down       Stop compose services"
	@echo "  compose-logs       Follow compose logs"
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
	$(DOCKER) build -f docker/Dockerfile.dev -t ctx-aware:dev .

docker-build-test:
	$(DOCKER) build -f docker/Dockerfile.test -t ctx-aware:test .

docker-build-prod:
	$(DOCKER) build -f docker/Dockerfile.prod -t ctx-aware:prod .

build-engine:
	$(DOCKER) build -f docker/Dockerfile.engine -t ctx-aware:engine .

export-engine: build-engine
	$(DOCKER) run --rm --gpus all \
		-v "$(CURDIR)/models:/app/models" \
		ctx-aware:engine

compose-config:
	$(COMPOSE) config

compose-up:
	$(COMPOSE) up --build -d

compose-down:
	$(COMPOSE) down

compose-logs:
	$(COMPOSE) logs -f

run:
	$(PYTHON) -m src.main

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('benchmarks').glob('laptop_report_*.json')] if Path('benchmarks').exists() else None"
