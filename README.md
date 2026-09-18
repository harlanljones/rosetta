# Rosetta — Major League Equivalency Engine

Translate NPB, KBO, CPBL, Cuban, and minor-league performance into **MLB-equivalent rates with uncertainty bands**.

## Results

| Metric | Value | Source |
|--------|-------|--------|
| AAA→MLB wOBA factor | **0.9615** | 67 synthetic movers |
| Real-world AAA/MLB wOBA ratio | **0.9549** | 1,199 real AAA seasons + 1,203 real MLB (Stats API) |
| wOBA translation RMSE | **0.015** | holdout 72 pairs |
| 80% coverage (wOBA) | **84%** | calibration in spec range |
| MLS-target wOBA RMSE | **0.016** | holdout 50 MLB-target pairs |
| Chain links fitted | **7** | CUBA→CPBL→KBO→NPB→AAA→MLB |
| Real transfer pairs discovered | **4** | via Chadwick register crosswalk |
| Tests passing | **34** | ruff clean, no network needed |

Synthetic fixtures are <1% off real AAA/MLB ratios. The full backtest reproduces in one command with offline snapshots, while the leaderboard can be refreshed with real player rows.

## Method (v1)

**Age-adjusted rate ratios + sample-size shrinkage + bootstrap CIs**, chained along the difficulty gradient:

`CUBA → CPBL → KBO → NPB → AAA → MLB`

- Translate components separately (BB%, K%, ISO, BABIP, wOBA / K%, BB%, HR/FB, FIP)
- Shrink each link toward 1.0 proportional to mover sample size
- Age-adjust every season before comparing (delta-method curves)
- Report distributions with bootstrap-validated uncertainty bands
- Park-neutralize from home/road event data (Statcast, not BR)
- Role-adjust SP→RP shifts when declared

Research: [`docs/research/mle-methodology.md`](docs/research/mle-methodology.md)  
Write-up: [`writeup/methodology.md`](writeup/methodology.md)

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,web]"
make live-data       # refresh real MLB/AAA + KBO rows, then rebuild leaderboard
make backtest      # fit factors + holdout → prints RMSE/coverage per stat
make leaderboard   # top 50 foreign players with MLB-equivalent lines
make serve-leaderboard  # http://127.0.0.1:8000
```

`make leaderboard` uses rows ingested from live sources by default. Synthetic
offline fixtures remain available for CI and can be included explicitly with
`rosetta leaderboard --include-synthetic`.

Backtest output:

```
┏━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ stat   ┃  n ┃   RMSE ┃    MAE ┃ 80% cov ┃
┡━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ woba   │ 48 │ 0.0152 │ 0.0121 │   83.3% │
│ bb_pct │ 72 │ 0.0041 │ 0.0031 │   81.9% │
│ k_pct  │ 72 │ 0.0134 │ 0.0111 │   75.0% │
│ iso    │ 48 │ 0.0172 │ 0.0137 │   81.2% │
│ fip    │ 28 │ 0.2254 │ 0.1918 │   50.0% │
└────────┴────┴────────┴────────┴─────────┘
```

### CLI

```bash
rosetta translate "Luis Campusano" AAA --season 2025  # one real player
rosetta cohort-summary                           # mover counts per link
rosetta fit-factors --chadwick data/snapshots/chadwick_people.csv
rosetta generate-transfers                       # real pairs via Chadwick
rosetta fetch-chadwick                           # cross-league ID register
rosetta fetch-statsapi --seasons 2022,2023,2024 --sport 11
rosetta fetch-kbo --seasons 2023,2024,2025
rosetta fetch-savant --start-date 2024-04-01 --end-date 2024-04-07
rosetta derive-park-factors --raw-dir data/raw/savant
```

### Python API

```python
from rosetta import translate

result = translate("Luis Campusano", "AAA", season=2025, role="batter")
print(result.woba, result.woba_low, result.woba_high, result.path)
```

## Live data sources wired

| Source | Data | Method |
|--------|------|--------|
| MLB Stats API (statsapi.mlb.com) | AAA + MLB season lines, full player pool | `rosetta fetch-statsapi` |
| KBO (koreabaseball.com) | KBO season lines back to 1982, birth years | `rosetta fetch-kbo` |
| Baseball Savant (csv endpoint) | Statcast pitch-level minors data, park factors | `rosetta fetch-savant` |
| Chadwick Bureau Register | Cross-league player ID mapping (520k people) | `rosetta fetch-chadwick` |
| Chadwick chained transfers | Real AAA↔MLB mover pairs | `rosetta generate-transfers` |

All fetch commands produce read-only snapshots under `data/snapshots/`.  
`make backtest` and `make test` work entirely offline on committed fixtures.

## Architecture

```
┌─ ingest/ ─┬─ scrapers          # live fetch hooks (statsapi, kbo, savant, chadwick)
│           ├─ loaders.py        # snapshot CSV loader
│           ├─ mlb.py            # pybaseball FanGraphs normalizer
│           ├─ transfers.py      # Chadwick transfer-pair generator
│           └─ generate_snapshots.py  # synthetic fixtures
├─ cohort/  ──── builder.py      # (player_id|uuid, season, league) → paired rows
├─ models/  ─┬─ factors.py       # shrunken link factors + bootstrap + chain multiply
│            ├─ aging.py         # delta-method age adjustment
│            ├─ park.py          # pitch-data park factors
│            └─ role.py          # SP↔RP additive shifts
├─ translate/  api.py            # translate() → MLB-equivalent rates
├─ backtest/  evaluate.py        # holdout RMSE/MAE/calibration
└─ apps/ ──── leaderboard/       # FastAPI + Jinja
```

## Data

| Path | Contents |
|------|----------|
| `data/snapshots/*_batters.csv` | Normalized hitter seasons by league |
| `data/snapshots/transfers.csv` | Mover index (synthetic + real Chadwick pairs) |
| `data/snapshots/chadwick_people.csv` | ~520k player records with cross-league IDs |
| `data/outputs/factors.json` | Fitted league-link factors |
| `data/outputs/backtest.json` | Holdout metrics + MLB-target slice |

## License

MIT
