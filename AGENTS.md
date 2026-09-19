# rosetta — Development Agent Guide

## Project intent

Rosetta is a reproducible, leakage-safe baseball projection system for
amateur-to-professional talent evaluation: it translates league performance
(KBO, NPB, CPBL, Cuba, AAA) to an MLB-equivalent scale using a transferred-
player cohort, component-wise league factors with bootstrap confidence
intervals, and temporal backtests. The public-facing artifact is the MLE
leaderboard (`apps/leaderboard`).

## Where to work

| Path | Purpose |
|---|---|
| `src/rosetta/ingest/` | Live data sources: `statsapi` (MLB Stats API, AAA + MLB), `kbo` (koreabaseball.com), `savant`, `chadwick` (ID crosswalk), `transfers` (real transfer-pair generation), `generate_snapshots` (synthetic fixtures) |
| `src/rosetta/models/factors.py` | League-factor model: link estimation, shrinkage, bootstrap CIs |
| `src/rosetta/cohort/builder.py` | Transferred-player cohort assembly |
| `src/rosetta/cli.py` | Typer CLI — all commands (`rosetta --help`) |
| `apps/leaderboard/` | FastAPI + Jinja leaderboard; serves `data/outputs/leaderboard.json` |
| `data/snapshots/` | Season CSVs + `transfers.csv` + `chadwick_people.csv` (git-tracked; `data/outputs/` and `data/raw/` are not) |
| `writeup/`, `docs/` | Methodology and research notes |
| `tests/` | pytest suite (offline, no network) |

## Non-negotiable invariants

- **The public leaderboard is real-data-only.** Rows marked
  `is_synthetic=True` must never render unless `--include-synthetic` is
  passed explicitly. Never "fill in" missing boards with synthetic rows.
- **Live rows and synthetic rows share snapshot files** (the statsapi writer
  appends). Any snapshot-cleaning step must filter on the `is_synthetic`
  column, not on file names.
- **The Stats API ingest must request `playerPool=ALL`.** Without it the API
  silently returns qualified players only, which destroys the transfer
  cohort (near-zero cross-league pairs).
- **Factors are fitted from real transferred-player pairs only.** A fit with
  empty `links` means the holdout filter consumed everything — do not ship
  an empty factor file.
- **Holdout discipline:** pairs are held out by most-recent `to_season`, and
  only when earlier to-seasons exist to train on. Never train on the holdout.
- **Snapshot schemas** are defined in `src/rosetta/schema.py` — extend the
  column lists there, don't freelance CSV shapes.

## Commands

```bash
source .venv/bin/activate
make test          # pytest (offline)
make lint          # ruff check src tests apps
make live-data     # fetch-statsapi (AAA + MLB) + fetch-kbo, then leaderboard
make factors       # fit factors.json (boot 250)
make leaderboard   # rebuild leaderboard.json (refits factors only if missing)
make serve-leaderboard
```

- The `rosetta` CLI must be run from an activated `.venv` (the Makefile
  assumes `rosetta` is on PATH).
- `make leaderboard` does NOT refit factors when `data/outputs/factors.json`
  exists — after a data refresh, delete `factors.json` or run `make factors`.
- Network-touching commands (`fetch-*`, `make live-data`) are slow (KBO
  crawls with 1s delays) and hit live sites; run them deliberately.

## Quality gate

Run `make test && make lint` before declaring any change done. Tests are
offline — a failing test needing network is a bug; mock or fixture it.

## Data safety

- Never commit credentials, tokens, or `.env` files.
- `data/raw/` (Savant CSV cache) and `data/outputs/` are gitignored;
  `data/snapshots/` is the curated, committed layer.
- The Chadwick register (`data/snapshots/chadwick_people.csv`) is large and
  re-downloadable — do not hand-edit it.

## Code style

Python 3.11+, pandas/typer/rich. Match the surrounding code:

- Ruff enforces lint; run `make lint`, don't argue with it.
- Type hints on public functions; docstrings say what the function owns.
- Ingest modules: network fetch and normalization are separate functions so
  they can be tested offline.

## Commits

Conventional Commits (`feat:`, `fix:`, `chore:`) — the repo's history uses
`feat(ingest): ...` style. Imperative subject.

## Progress reporting

Report against outcomes: test/lint green; leaderboard `data_mode: "real"`
with nonzero `source_counts`; factors `links` populated with real n.
