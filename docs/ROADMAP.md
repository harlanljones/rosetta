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
- Leaderboard app: `/player` lookup across both boards, `/api/leaderboard.csv` full export.
- Optional `odds_ratio` link estimator (`rosetta fit-factors --estimator odds_ratio`):
  per-pair odds(post)/odds(age-adj pre), weighted mean; proportion stats only
  (era/fip fall back to ratio_of_means in whole-model fits). 2022-2026 rolling
  backtest on real snapshots: ratio_of_means stays the production default
  (rel-MAE 0.706 vs 0.718; odds_ratio wins only on woba). Interval coverage
  already healthy (mean 80% band coverage 0.837) — no interval retune applied.

## Near-term priorities

| ID | Goal |
|----|------|
| D3 | Deeper census (inactive NPB index, more seasons in `make live-data`) |
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
