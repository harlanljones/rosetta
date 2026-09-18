# Rosetta — Major League Equivalency Engine

Translate NPB, KBO, CPBL, Cuban, and minor-league performance into **MLB-equivalent rates with uncertainty bands**.

Hiring-signal artifact for international performance translation: transferred-player cohort → shrunken league-link factors → chained MLE → holdout backtest → public leaderboard.

```
ingest/ → cohort/ → models/ → translate/ → backtest/
                              ↘ leaderboard app
```

## Method (v1)

**Age-adjusted rate ratios + sample-size shrinkage + bootstrap CIs**, chained along:

`CPBL → KBO → NPB → AAA → MLB`

- Translate components first (BB%, K%, ISO, BABIP, … / K%, BB%, HR/FB, FIP)
- Shrink noisy links toward 1.0
- Age-adjust with delta-method curves before comparing seasons
- Report distributions, not points

Research notes: [`docs/research/mle-methodology.md`](docs/research/mle-methodology.md)  
Write-up: [`writeup/methodology.md`](writeup/methodology.md)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,web]"
make snapshots    # synthetic fixtures for offline CI
make backtest     # fit factors + holdout metrics
make leaderboard
make serve-leaderboard   # http://127.0.0.1:8000
```

### CLI

```bash
rosetta cohort-summary
rosetta fit-factors
rosetta translate "NPB Hitter 1" NPB --role batter
rosetta backtest
rosetta leaderboard --season 2025
```

### Python API

```python
from rosetta import translate

result = translate(
    "NPB Hitter 1",
    "NPB",
    role="batter",
    factors_path="data/outputs/factors.json",
    snapshot_dir="data/snapshots",
)
print(result.woba, result.woba_low, result.woba_high, result.path)
```

## Data

| Path | Purpose |
|------|---------|
| `data/snapshots/*_batters.csv` | Normalized hitter seasons by league |
| `data/snapshots/*_pitchers.csv` | Normalized pitcher seasons |
| `data/snapshots/transfers.csv` | Mover index (from/to league + seasons) |
| `data/outputs/factors.json` | Fitted link factors |
| `data/outputs/backtest.json` | Holdout metrics |

Snapshots in-repo are **labeled synthetic** cohorts shaped like real league environments so `pytest` and `make backtest` work without network access. Optional live scrapers live under `rosetta.ingest.scrapers` (install `.[scrape]`).

## Architecture

| Module | Role |
|--------|------|
| `rosetta.schema` | One normalized seasonal schema for every league |
| `rosetta.ingest` | Snapshot loaders + synthetic generator + scraper stubs |
| `rosetta.cohort` | Transferred-player pairs |
| `rosetta.models` | Aging, park factors, shrunken link factors, bootstrap |
| `rosetta.translate` | `translate(player, from_league)` |
| `rosetta.backtest` | Hold out recent movers; RMSE/MAE/coverage |
| `apps/leaderboard` | FastAPI + Jinja leaderboard |

## Make targets

| Target | Does |
|--------|------|
| `make install-dev` | Editable install with test/web extras |
| `make snapshots` | Regenerate offline fixtures |
| `make factors` | Fit league-link factors |
| `make backtest` | Full holdout evaluation |
| `make leaderboard` | Export JSON boards |
| `make serve-leaderboard` | Run web app on :8000 |
| `make test` | pytest |

## License

MIT
