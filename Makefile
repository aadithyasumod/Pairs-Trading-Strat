SHELL := /bin/bash

# Use the project venv when it exists; `python` is not on PATH on this machine.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help setup run run-analysis run-baseline run-walkforward test lint

help:
	@echo "setup           - create .venv and install requirements"
	@echo "run             - run the live analysis monitor"
	@echo "run-analysis    - run one monitor cycle with monitor_config.json"
	@echo "run-baseline    - run the baseline pairs trading entrypoint (synthetic data)"
	@echo "run-walkforward - out-of-sample walk-forward backtest (synthetic universe)"
	@echo "test            - run the Python test suite"
	@echo "lint            - reserved for future lint tooling"

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

run: run-analysis

run-analysis:
	$(PYTHON) -m pairs_trading.simple_monitor --config monitor_config.json

run-baseline:
	$(PYTHON) -m pairs_trading.main

run-walkforward:
	$(PYTHON) -m pairs_trading.walk_forward

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v

lint:
	@echo "No dedicated lint command is configured yet."
