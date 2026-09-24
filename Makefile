# crystallizer-ai developer targets. All targets run inside the local virtual environment.
PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.PHONY: install lint typecheck test schemas schemas-check check clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck:
	$(BIN)/mypy --strict src tests

test:
	$(BIN)/pytest --cov=src/crystallizer --cov-fail-under=90

schemas:
	$(PY) -m crystallizer schemas export

schemas-check:
	$(PY) -m crystallizer schemas check

check: lint typecheck test schemas-check

clean:
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache .pytest_cache build dist src/*.egg-info
