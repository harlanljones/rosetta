# Model performance evidence

This report records the reproducible comparison behind the live demo's
rolling model analysis. It compares the raw `ratio_of_means` factor translation
with the rolling-evaluation `prior_affine` calibration on the same real AAA-to-MLB
snapshots. The calibration is fit separately for each role and stat as
`actual = intercept + slope * raw_prediction`.

## Reproduce

Run this offline from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare-rolling-calibration.py \
  --snapshots data/snapshots \
  --target-seasons 2022,2023,2024,2025,2026 \
  --boot 250 --seed 42 --min-pa 80 --min-ip 30 \
  --output /tmp/rosetta-calibration-comparison.json
```

The command fits both variants from scratch and emits pooled and per-target
metrics. It does not fetch data or write production artifacts.

## 2026 holdout

The 2026 target is never used to choose its calibration. Both rows below score
the same 151 batter and 93 pitcher pairs.

| Role/stat | Raw MAE | Calibrated MAE | Raw RMSE | Calibrated RMSE | Raw 80% coverage | Calibrated 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Batter wOBA | 0.0368 | 0.0322 | 0.0482 | 0.0417 | 86.1% | 83.4% |
| Pitcher FIP | 0.9184 | 0.6399 | 1.1556 | 0.8391 | 84.9% | 88.2% |

The persistence baseline for 2026 is wOBA MAE/RMSE `0.0670/0.0804` and FIP
`0.9746/1.2244`.

## Pooled 2022–2026 comparison

The table reports raw versus calibrated factor-model metrics. Deltas are
calibrated minus raw, so negative MAE/RMSE values are improvements.

| Role/stat | n | Raw MAE | Calibrated MAE | Δ MAE | Raw RMSE | Calibrated RMSE | Δ RMSE | Raw cov. | Calibrated cov. | Δ cov. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batter BABIP | 750 | 0.0449 | 0.0394 | -0.0055 | 0.0586 | 0.0517 | -0.0069 | 84.4% | 81.5% | -2.9pp |
| Batter BB% | 750 | 0.0250 | 0.0224 | -0.0026 | 0.0322 | 0.0294 | -0.0029 | 82.8% | 82.0% | -0.8pp |
| Batter HR% | 750 | 0.0123 | 0.0122 | -0.0002 | 0.0171 | 0.0164 | -0.0008 | 88.7% | 82.0% | -6.7pp |
| Batter ISO | 750 | 0.0493 | 0.0473 | -0.0021 | 0.0656 | 0.0614 | -0.0042 | 88.4% | 81.3% | -7.1pp |
| Batter K% | 750 | 0.0460 | 0.0431 | -0.0029 | 0.0585 | 0.0537 | -0.0047 | 81.2% | 81.1% | -0.1pp |
| Batter wOBA | 750 | 0.0439 | 0.0381 | -0.0058 | 0.0572 | 0.0503 | -0.0069 | 84.0% | 82.8% | -1.2pp |
| Pitcher BB% | 489 | 0.0255 | 0.0203 | -0.0052 | 0.0328 | 0.0263 | -0.0064 | 79.3% | 77.5% | -1.8pp |
| Pitcher ERA | 489 | 1.3924 | 1.0849 | -0.3076 | 1.8107 | 1.4037 | -0.4070 | 84.5% | 81.0% | -3.5pp |
| Pitcher FIP | 489 | 1.0272 | 0.7974 | -0.2298 | 1.2911 | 1.0135 | -0.2777 | 83.8% | 85.1% | +1.2pp |
| Pitcher HR/FB | 489 | 0.0494 | 0.0376 | -0.0117 | 0.0631 | 0.0492 | -0.0139 | 84.9% | 81.6% | -3.3pp |
| Pitcher K% | 489 | 0.0414 | 0.0374 | -0.0040 | 0.0529 | 0.0492 | -0.0036 | 80.0% | 77.1% | -2.9pp |

All pooled MAEs and RMSEs improved. Coverage is a separate uncertainty metric;
it declined for several rate stats and remains a calibration caveat even
though the residual-width rule improves the aggregate interval behavior.
During this iteration I inspected the full rolling artifact, including 2026,
while comparing candidate methods and interval rules. Therefore 2026 is a
retrospective evaluation result, not a pristine model-selection holdout. The
coverage regressions and this selection limitation are exposed in the demo.

As an earlier-fold-only replay, rerunning the same command with target seasons
`2022,2023,2024,2025` gives 11/11 pooled stat MAE and RMSE improvements:
wOBA changes from `0.0457/0.0593` to `0.0396/0.0522`, and FIP from
`1.0528/1.3209` to `0.8344/1.0502`. This supports the method's stability on
earlier folds, but it does not erase the fact that the method was selected
after viewing the complete 2022–2026 comparison.

## Cutoffs and interval handling

Every target's factor model uses only adjacent real AAA-to-MLB pairs whose MLB
destination season is strictly earlier than that target. The rolling folds use
minimum 80 PA for batters and 30 IP for pitchers, with the AAA 2019 era floor.
Calibration uses completed rolling folds only:

| Target | Factor-model cutoff | Calibration folds available |
|---:|---|---|
| 2022 | destination season < 2022 | none; identity correction |
| 2023 | destination season < 2023 | 2022 |
| 2024 | destination season < 2024 | 2022–2023 |
| 2025 | destination season < 2025 | 2022–2024 |
| 2026 | destination season < 2026 | 2022–2025 |

An affine correction transforms the raw interval endpoints. Its half-width is
then widened to the larger of the transformed factor band and the 85th
percentile absolute residual from earlier folds. Rate outputs are clipped to
`[0, 1]`; ERA and FIP outputs are floored at `0.5`. No current-target outcome
is used in either the point correction or interval width.

The calibration applies only to retrospective rolling evaluation and demo
analysis. Production leaderboard translations continue to use the raw factor
model; wiring prior-fold calibration into production projections is outside
this change.

These are overlapping retrospective folds over players who actually received
MLB playing time, not a promotion or playing-time forecast. The result is
evidence that this calibration improves Rosetta's internal translation
backtest; it is not a guarantee for future players or a ranking against MLB
projection systems.
