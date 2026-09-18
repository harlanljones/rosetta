.PHONY: install install-dev test lint backtest leaderboard factors snapshots live-data clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,web]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests apps

snapshots:
	$(PYTHON) -m rosetta.ingest.generate_snapshots

live-data:
	rosetta fetch-statsapi --seasons 2024,2025 --sport 11 --league-label AAA
	rosetta fetch-statsapi --seasons 2024,2025 --sport 1 --league-label MLB
	rosetta fetch-kbo --seasons 2024,2025 --no-with-ages
	$(MAKE) leaderboard

factors:
	@mkdir -p data/outputs
	rosetta fit-factors --snapshots data/snapshots --out data/outputs/factors.json --boot 250

backtest:
	@mkdir -p data/outputs
	@test -f data/outputs/factors.json || $(MAKE) factors
	rosetta backtest --snapshots data/snapshots --factors data/outputs/factors.json --out data/outputs/backtest.json
	@echo "Backtest written to data/outputs/backtest.json"

leaderboard:
	@mkdir -p data/outputs
	@test -f data/outputs/factors.json || $(MAKE) factors
	rosetta leaderboard --snapshots data/snapshots --factors data/outputs/factors.json --out data/outputs/leaderboard.json
	@echo "Leaderboard JSON at data/outputs/leaderboard.json"
	@echo "Run: make serve-leaderboard"

serve-leaderboard: leaderboard
	PYTHONPATH=. $(PYTHON) -m uvicorn apps.leaderboard.app:app --reload --port 8000

clean:
	rm -rf data/outputs/* .pytest_cache **/__pycache__ *.egg-info
