# Rosetta — Major League Equivalency Engine

Translate NPB, KBO, CPBL, Cuban, and minor-league performance into **MLB-equivalent rates with uncertainty bands**.

## Results

Historical validation is intentionally generated from the current real snapshots rather
than hard-coded here. See the checked-in [historical backtest results](docs/historical-backtest-results.md)
for the latest concise readout, and the generated HTML/Markdown artifacts for complete
player-level detail. Synthetic fixtures are for offline tests and demos only; they are
not evidence of real-world performance.

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
make historical-backtest  # real adjacent-season AAA→MLB validation report
make historical-rolling   # same, refit and scored independently per target season
make leaderboard   # top 50 foreign players with MLB-equivalent lines
make showcase-data # export the real-data-only portfolio bundle
make serve-leaderboard  # http://127.0.0.1:8000
```

`make leaderboard` uses rows ingested from live sources by default. Synthetic
offline fixtures remain available for CI and can be included explicitly with
`rosetta leaderboard --include-synthetic`.

The historical report is a retrospective rate-translation check for players with
observed MLB playing time, not a promotion or playing-time forecast.

### Portfolio showcase

The FastAPI app serves the narrative proof page at
[`/showcase`](http://127.0.0.1:8000/showcase). It reads only the static bundle
under `data/outputs/demo/` — `meta.json`, `leaderboard.json`, `backtest.json`,
and `analysis.json` — and refuses to render numbers unless the bundle declares
`data_mode: "real"`.
The page leads with the current target-season backtest (prior AAA inputs → MLB
outcomes), shows all evaluated metric rows and the available rolling comparison,
then links to the full leaderboard at `/`. The separate `/analysis` route is a
diagnostic report: it keeps each stat in native units and adds bias, calibration
against the 80% interval target, baseline win rate, error buckets, and a
current-season player drilldown.
The result is retrospective; the 2026 fold was inspected while comparing
candidate model and interval rules, so it is not an untouched selection holdout.

Build or refresh the bundle with:

```bash
make historical-backtest
make historical-rolling
make showcase-data
make serve-leaderboard
```

For another target season, use the export script directly:

```bash
python scripts/export_demo.py --target-season 2026 --out data/outputs/demo
```

The showcase is intentionally static at view time: it performs no network fetch
and has no chart dependency. Its metadata includes the source checksums,
generation timestamp, estimator, era floor, and git commit used to create the
bundle.

Offline reproduction against the committed real snapshots:

```bash
source .venv/bin/activate
make historical-backtest
make historical-rolling
```

Optional refresh (network access; not required for the committed reproduction):

```bash
rosetta fetch-statsapi --seasons 2018,2019,2020,2021,2022,2023,2024,2025 --sport 11 --league-label AAA
rosetta fetch-statsapi --seasons 2018,2019,2020,2021,2022,2023,2024,2025 --sport 1 --league-label MLB
```

The report records its input checksums, cutoff, source counts, and limitations in
`data/outputs/historical-backtest-2026/`. The demonstrator covers the adjacent AAA→MLB
link only; normalized wOBA/FIP/HR/FB and inferred ages retain approximation caveats.

Legacy `make backtest` output (fixture/transfer holdout illustration; not the
real adjacent-season validation) looks like:

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
