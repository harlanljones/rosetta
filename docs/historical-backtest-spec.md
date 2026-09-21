# Historical backtest proof of concept

## Outcome

Provide an offline, reproducible report comparing predictions made from the
previous season with actual MLB performance in a completed target season.
Default target: 2025, with 2024 source lines and training outcomes before 2025.
Use real snapshots only. This is retrospective validation of rate translations
among players with observed MLB playing time, not a prediction of promotion or
playing time and not proof of production readiness.

## Delivery contract

- Add `rosetta historical-backtest` and `make historical-backtest` without
  changing the legacy synthetic backtest contract or leaderboard outputs.
- Add `src/rosetta/backtest/historical.py` with
  `run_historical_backtest(snapshot_dir=None, *, target_season=2025,
  n_boot=250, seed=42, min_pa=80.0, min_ip=30.0) -> dict`.
- Build adjacent-season AAA-to-MLB pairs by shared player ID from normalized
  snapshots. This first demonstrator deliberately covers the supported real
  AAA/MLB link; do not imply validation of international links.
- Filter explicit `is_synthetic=False` on both source and actual lines before
  pairing. Reject ambiguous duplicate player/league/season/role keys. Ignore
  transfer-file holdout flags and cached factors. Fit fresh factors using only
  pairs whose destination season is earlier than the target. Require nonempty
  training and evaluation sets and fitted factors with positive real counts.
- Evaluate only target-minus-one to target pairs; apply declared playing-time
  thresholds to both sides. Explain the resulting selection bias. Do not
  mutate source snapshots or ordinary factors/leaderboard artifacts during a run.
- Preserve the current translation model. Identify approximation caveats for
  normalized wOBA, FIP, HR/FB and inferred/default ages. Do not tune on holdout.
- Include prior source rate as a no-translation baseline. Report role-separated
  per-stat counts, MAE, RMSE, baseline MAE/RMSE and 80% interval coverage. Any
  improvement claims must follow the measured result, including regressions.
- Include every player's ID, name, role, source/target seasons and leagues,
  sample sizes, source/actual provenance, stat, prior rate, prediction, actual,
  signed error, absolute error, interval and coverage flag in machine-readable
  detail. Include training pair identities, factor counts and cutoff evidence.
- Return JSON-compatible `metadata`, `metrics`, `detail`, `training_pairs`,
  and `model` keys. `metrics` is a list of dictionaries with keys `role`,
  `stat`, `n`, `mae`, `rmse`, `baseline_mae`, `baseline_rmse`, `coverage_80`.
  `detail` uses `player_id`, `player_name`, `role`, `from_season`, `to_season`,
  `from_league`, `to_league`, `pre_sample`, `post_sample`, `pre_source`,
  `post_source`, `stat`, `prior`, `prediction`, `actual`, `error`,
  `absolute_error`, `lower`, `upper`, `covered_80`.
- Metadata includes `data_mode=real`, target/source seasons, training cutoff,
  counts, parameters, input SHA-256 checksums, source counts and limitations.
  Reject non-finite required values with actionable errors. Stable bootstrap
  seeding must make separate-process runs reproduce identical output.
- Add `src/rosetta/backtest/report.py` with
  `write_historical_report(payload, out_dir) -> dict[str, Path]`, producing
  `historical-backtest.json`, `.csv`, `.md`, `.html` and a training CSV.
  HTML is self-contained, readable offline, and escapes dynamic content.
  Explain MAE/RMSE in ordinary language and show named batter/pitcher examples,
  all-player detail, baseline comparisons, and visible limitations.
- CLI flags: `--snapshots`, `--target-season`, `--boot`, `--seed`, `--min-pa`,
  `--min-ip`, `--out-dir` (default `data/outputs/historical-backtest`).
  Print headline evidence and artifact paths; errors must be actionable.
- Update README with an accurate real-data entry point, acquisition/reproduction
  steps and links to a checked-in concise results document. Avoid stale claims
  presenting synthetic performance as real validation.

## Acceptance and review

1. An actual 2024-to-2025 run has nonzero real training and scored MLB players,
   with latest training destination year at most 2024 and populated factors.
2. Offline tests prove synthetic exclusion on either side, strict cutoff despite
   misleading flags, future/holdout actual perturbations leave factors and
   predictions unchanged, hand-computable metrics/baseline, deterministic
   cross-process output, empty/insufficient-data errors and duplicate handling.
3. A human-readable artifact shows named players' predictions and actuals;
   machine-readable detail independently reproduces aggregate numbers.
4. Missing historical data is acquired deliberately through existing Stats API
   ingestion with `playerPool=ALL`, preserving curated data and recording
   provenance. No synthetic substitution is acceptable.
5. `source .venv/bin/activate && make test && make lint` passes. Parent reviews
   source, runs independent verification, and inspects the generated report.

## Delegation

Luna with high reasoning owns temporal evaluation and data integrity; Luna with
medium reasoning owns report/CLI integration. The data-audit Luna then owns
independent adversarial tests. The parent writes specs, reviews and executes
verification, and writes no application or test code.
