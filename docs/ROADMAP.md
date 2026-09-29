# Rosetta development roadmap

Living plan for data, model validation, and product work. See git history for
completed milestones.

## Tracks

1. **Data plane** — real cross-league season lines and transfer pairs for
   `CUBA → CPBL → KBO → NPB → AAA → MLB`.
2. **Model & validation** — leakage-safe backtests, production calibration, park
   handling, international link reports.
3. **Product & ops** — leaderboard/showcase refresh, scheduled ingest, docs.

## Current capabilities

- AAA→MLB historical and rolling backtests with optional `prior_affine` calibration.
- Live ingest: Stats API (AAA/MLB), KBO, NPB, CPBL (`rosetta fetch-*`).
- KBO crosswalk via romanization aliases + optional `kbo_id_map.csv`.
- Production `fit-factors` embeds calibration; `translate()` applies it on AAA paths.
- `rosetta historical-backtest --from-league NPB --to-league MLB` for international links.
- `rosetta enrich-aaa-splits` merges Savant home/road columns into AAA batters.

## Near-term priorities

| ID | Goal |
|----|------|
| D3 | Deeper census (inactive NPB index, more seasons in `make live-data`) |
| M4 | Interval calibration tuning; optional odds-ratio estimator |
| P2 | Leaderboard player lookup + CSV export |
| P4 | Optional container deploy for leaderboard + demo bundle |

## Commands

```bash
make live-data          # network: refresh snapshots + factors + leaderboard
make historical-backtest
make historical-rolling
rosetta fetch-cpbl --seasons 2024
rosetta generate-transfers
```

## Quality gate

`make test && make lint` — tests stay offline; network belongs in `fetch-*` only.
