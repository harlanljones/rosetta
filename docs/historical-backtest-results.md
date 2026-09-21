# Real historical backtest: 2024 AAA → 2025 MLB

The current translation (ratio-of-means link factors with an AAA
2019-ball-standardization era floor, both defaults as of this readout)
beats the no-translation baseline on MAE for all 11 role/statistic groups on
this production target-season readout. That is an improvement over the
prior readout, where pitcher FIP, ERA, and the approximate HR/FB measure
remained slightly worse than baseline; the era floor (see "AAA era floor"
below) removes that gap on this single target season, though the multi-year
rolling pool below still shows pitcher FIP and ERA landing on either side of
baseline depending on target year. This is a reproducible proof of concept
with visible successes and misses, not evidence that every model component
is ready for forecasting.

## Reproduce offline

From the repository root, activate `.venv` and run `make historical-backtest`
and `make historical-rolling`. The committed snapshots supply the data;
these commands make no network calls. Open
[the generated HTML report](../data/outputs/historical-backtest/historical-backtest.html).
The same directory contains JSON, full player-stat CSV, training-pair CSV, and
Markdown. `data/outputs/historical-rolling/` holds the per-target-season
companion artifacts. Outputs are gitignored and recreated by the commands.

This readout uses target season 2025, seed 42, 250 bootstrap resamples, the
AAA 2019 era floor, and minimum playing time on **both** sides of 80 PA for
batters or 30 IP for pitchers. The training set contains 835 adjacent-season
AAA→MLB pairs with MLB outcomes from 2019 through 2024, after the era floor
excludes pairs with a pre-2019 AAA endpoint (see "AAA era floor" below).
AAA 2020 is absent, so training does not contain pairs requiring that source
season.

The evaluation set uses only 2024 AAA inputs and actual 2025 MLB outcomes:
159 batters and 114 pitchers, producing 1,524 player-stat comparisons. Both
endpoints must explicitly have `is_synthetic=False`. The evaluator ignores
cached factors and transfer-file holdout flags, and fits fresh factors with
`to_season < 2025`. All used source rows identify Stats API as their source.

## What the errors show

MAE is the average absolute miss; RMSE gives larger misses more weight. Lower
is better. The baseline simply carries the player's 2024 AAA rate forward.
Metrics average players equally within each role/statistic.

| Outcome | Players | Model MAE | Baseline MAE | Model RMSE | Baseline RMSE | Observed 80% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| Batter wOBA | 159 | 0.0397 | 0.0755 | 0.0512 | 0.0905 | 83.6% |
| Pitcher FIP | 114 | 0.8077 | 0.8235 | 1.0072 | 1.0197 | 91.2% |

All 11 role/statistic groups beat the no-translation baseline on MAE on this
target season:

| Role | Stat | n | MAE | Baseline MAE | 80% coverage |
|---|---|---:|---:|---:|---:|
| batter | babip | 159 | 0.0451 | 0.0587 | 81.1% |
| batter | bb_pct | 159 | 0.0207 | 0.0394 | 88.7% |
| batter | hr_pct | 159 | 0.0106 | 0.0158 | 93.1% |
| batter | iso | 159 | 0.0436 | 0.0729 | 91.2% |
| batter | k_pct | 159 | 0.0475 | 0.0498 | 79.9% |
| batter | woba | 159 | 0.0397 | 0.0755 | 83.6% |
| pitcher | bb_pct | 114 | 0.0228 | 0.0251 | 81.6% |
| pitcher | era | 114 | 1.2140 | 1.2239 | 89.5% |
| pitcher | fip | 114 | 0.8077 | 0.8235 | 91.2% |
| pitcher | hr_fb | 114 | 0.0419 | 0.0437 | 90.4% |
| pitcher | k_pct | 114 | 0.0319 | 0.0458 | 89.5% |

Before the era floor, pitcher ERA, FIP, and the approximate HR/FB measure
remained slightly worse than baseline on this same target season (FIP:
0.8549 vs. 0.8235; ERA: 1.2761 vs. 1.2239; HR/FB: 0.0460 vs. 0.0437 — see
"AAA era floor" below for the full comparison). The floor closes that gap
here; it does not guarantee the same result on every target season (see
"Rolling seasons").

## Estimator choice

The link factor was previously a weighted mean of per-player post/pre
age-adjusted rate ratios (`mean_ratio`). That estimator is biased upward by
Jensen's inequality and explodes when a mover's source-league rate is near
zero: the old AAA→MLB pitcher HR/FB factor was 2.03, even though league
median HR/FB sits around 0.10-0.11 on both sides of that link. The default is
now `ratio_of_means`: `sum(weight * post) / sum(weight * pre_adj)` over the
same cohort rows and weights, which cannot blow up the same way. The
`ratio_of_means` AAA→MLB pitcher HR/FB factor was 1.03 in the run this
comparison used; that run predates the AAA era floor described below, so it
is not comparable in magnitude to the current production factor (0.886, from
the floor-filtered cohort) — the estimator-choice conclusion (bounded,
non-exploding) still holds either way.

Selection was made on the 2022-2024 rolling backtest targets only (`ratio_of_means`
won all 33 target×stat MAE comparisons against `mean_ratio` across those three
seasons and 11 stats), then confirmed unseen on the 2025 holdout, where it also
won all 11 stat comparisons. Both estimators in this table were run **without**
the AAA era floor (the floor did not exist yet at estimator-selection time); the
same comparison re-run with the floor would shift both estimators' absolute
MAE values (see "AAA era floor" below) without changing which estimator wins.

| Stat | 2022–24 MAE old → new (baseline) | 2025 MAE old → new (baseline) |
|---|---|---|
| Batter wOBA | 0.0535 → 0.0515 (0.0818) | 0.0420 → 0.0412 (0.0755) |
| Batter HR% | 0.0248 → 0.0173 (0.0184) | 0.0163 → 0.0119 (0.0158) |
| Pitcher FIP | 1.4098 → 1.2729 (1.1079) | 0.9692 → 0.8549 (0.8235) |
| Pitcher HR/FB | 0.2591 → 0.0719 (0.0549) | 0.1560 → 0.0460 (0.0437) |

2022-24 figures are n-weighted averages of each target season's per-stat MAE
(weights = evaluated-player counts per target), computed from the old
(`mean_ratio`) and new (`ratio_of_means`) rolling summary CSVs; 2025 figures
are the single-target MAE for that holdout season. "Old" comes from a prior
`mean_ratio` rolling run; "new" and "baseline" come from a `ratio_of_means`
rolling run without the era floor, not the current (floor-enabled)
`data/outputs/historical-rolling/historical-rolling-summary.csv`.

## AAA era floor

**Diagnosis.** The AAA(2018)→MLB(2019) cohort was present in every training
set evaluated so far (it is the earliest available adjacent AAA→MLB pair)
and carried roughly 40-55% of pitcher training weight depending on target
season. Its raw FIP link factor was 1.38, versus 1.11-1.19 for every other
adjacent-season cohort. The reason traces to a known league change, not
noise: weighted AAA FIP was 3.76 in 2018, versus 4.3-4.7 in every AAA season
since, because AAA adopted the MLB (livelier) ball starting in 2019. A
single pre-standardization cohort was pulling the pooled pitcher link factor
away from the value every other cohort agreed on.

**Fix.** The floor (`LEAGUE_ERA_FLOOR = {"AAA": 2019}`, `apply_era_floor` in
`src/rosetta/models/factors.py`) excludes training pairs with an AAA
endpoint season before 2019. This is a domain-fixed floor tied to a known
equipment change, not a window tuned to minimize error. It removes 269 real
AAA→MLB training pairs from production factor fitting: 147 batter rows and
122 pitcher rows (batter AAA→MLB pairs: 675 → 528; pitcher AAA→MLB pairs:
451 → 329).

Two alternative explanations were investigated and ruled out. A train/predict
age-adjustment asymmetry was checked as a possible cause of the pre-2019
cohort's outlier factor; correcting it slightly worsened error, so it was not
the driver and was not adopted. Blending fitted factors toward the league
mean was also considered and rejected: it lowers MAE mechanically by adding
bias toward the mean rather than by fixing the underlying pre-standardization
mismatch identified above.

**Selection-period evidence (2022-2024 rolling targets, n-weighted pooled
MAE; the floor was not tuned against these numbers, but they did motivate
investigating the diagnosis above — see the caveat in "Scope and limits"):**

| Stat | No-floor MAE | Floor MAE | Baseline MAE | No-floor signed error | Floor signed error |
|---|---:|---:|---:|---:|---:|
| Batter wOBA | 0.0515 | 0.0478 | 0.0818 | — | — |
| Batter ISO | 0.0647 | 0.0558 | 0.0812 | — | — |
| Pitcher FIP | 1.2729 | 1.1519 | 1.1079 | +0.541 | +0.051 |
| Pitcher ERA | 1.5606 | 1.4893 | 1.4555 | +0.455 | −0.011 |
| Pitcher HR/FB | 0.0719 | 0.0537 | 0.0549 | — | — |

10 of 11 role/stat groups improved MAE with the floor in the selection
period; pitcher K% coverage fell (77.3% → 74.8%). Note that pitcher ERA and
FIP MAE with the floor (1.4893, 1.1519) remained slightly above the
2022-2024 pooled baseline (1.4555, 1.1079) even with the floor applied — the
floor narrows the gap sharply but does not close it in every pooled
statistic over that period.

**2025 confirmation (held out from the diagnosis and not used to choose the
floor year):**

| Stat | n | Baseline MAE | No-floor MAE | Floor MAE | No-floor signed error | Floor signed error | No-floor coverage | Floor coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter babip | 159 | 0.0587 | 0.0454 | 0.0451 | +0.0062 | +0.0053 | 79.9% | 81.1% |
| batter bb_pct | 159 | 0.0394 | 0.0216 | 0.0207 | +0.0091 | +0.0056 | 84.9% | 88.7% |
| batter hr_pct | 159 | 0.0158 | 0.0119 | 0.0106 | +0.0053 | +0.0020 | 92.5% | 93.1% |
| batter iso | 159 | 0.0729 | 0.0480 | 0.0436 | +0.0203 | +0.0087 | 88.7% | 91.2% |
| batter k_pct | 159 | 0.0498 | 0.0488 | 0.0475 | +0.0075 | +0.0027 | 78.0% | 79.9% |
| batter woba | 159 | 0.0755 | 0.0412 | 0.0397 | +0.0155 | +0.0076 | 83.6% | 83.6% |
| pitcher bb_pct | 114 | 0.0251 | 0.0240 | 0.0228 | +0.0050 | −0.0005 | 80.7% | 81.6% |
| pitcher era | 114 | 1.2239 | 1.2761 | 1.2140 | +0.361 | +0.089 | 89.5% | 89.5% |
| pitcher fip | 114 | 0.8235 | 0.8549 | 0.8077 | +0.254 | −0.032 | 91.2% | 91.2% |
| pitcher hr_fb | 114 | 0.0437 | 0.0460 | 0.0419 | +0.0148 | −0.0025 | 87.7% | 90.4% |
| pitcher k_pct | 114 | 0.0458 | 0.0329 | 0.0319 | +0.0039 | −0.0017 | 91.2% | 89.5% |

On 2025, the floor beats the no-floor run on MAE for all 11 role/stat
groups, and it flips pitcher ERA, FIP, and HR/FB from worse-than-baseline to
better-than-baseline (they were the three regressions called out in the
pre-floor headline above). Signed error also shrinks toward zero across the
board, most visibly for ERA and FIP, where the no-floor run had a large
positive (over-prediction) bias. The one small regression is pitcher K%
interval coverage, which drops slightly (91.2% → 89.5%), consistent with the
coverage cost seen in the selection period. This is a genuinely clean
confirmation: 2025 was not used to choose 2019 as the floor year, and the
result was accepted as reported rather than tuned further.

## Rolling seasons

Each target season is fit and scored independently: for target *T*, factors
are fit only on adjacent AAA→MLB pairs with `to_season < T`, then scored
against actual *T* outcomes. 2021 is skipped — there is no 2020 AAA snapshot,
so no adjacent AAA(2020)→MLB(2021) evaluation pairs exist. The earliest
scored target, 2022, trains on adjacent pairs with `to_season` in {2019, 2020}
(i.e. MLB outcomes from 2019 and 2020); no target trains on 2021 outcomes,
for the same missing-2020-AAA-season reason.

| Target | n (wOBA / FIP) | Model MAE (wOBA / FIP) | Baseline MAE (wOBA / FIP) | 80% coverage (wOBA / FIP) |
|---|---|---|---|---|
| 2022 | 141 / 80 | 0.0588 / 1.1567 | 0.0894 / 1.1581 | 76.6% / 83.8% |
| 2023 | 155 / 97 | 0.0405 / 1.0376 | 0.0679 / 1.0714 | 89.0% / 84.5% |
| 2024 | 144 / 105 | 0.0449 / 1.2538 | 0.0894 / 1.1033 | 84.0% / 74.3% |
| 2025 | 159 / 114 | 0.0397 / 0.8077 | 0.0755 / 0.8235 | 83.6% / 91.2% |

These figures are with the AAA era floor applied (current default); they are
not directly comparable to a pre-floor rolling run, since the floor changes
both the training cohort and the fitted factors for every target. Pooled
across all four targets (599 batter-season and 396 pitcher-season
comparisons combined; see `historical-rolling-summary.csv`), pitcher FIP and
ERA remain marginally worse than baseline (FIP 1.0528 vs. 1.0260; ERA 1.4100
vs. 1.3888), driven mostly by the 2024 target, where FIP is still notably
worse than baseline (1.2538 vs. 1.1033) even with the floor. wOBA beats
baseline in every target season shown.

Observed 80% coverage still falls short of nominal in some seasons, most
notably pitcher FIP in 2024 (74.3%) and pitcher bb_pct in 2024 (61.9%, not
shown in this two-stat table). The bootstrap interval is not reliably
calibrated across all target seasons.

## Named players

These are the first three names alphabetically (ignoring case) within each role, selected
without reference to prediction error. Actuals are the repository's normalized
MLB rates, not an assertion of exact agreement with published advanced stats.

| Player | Statistic | 2024 AAA | Predicted 2025 MLB | Actual 2025 MLB | Prediction − actual |
|---|---|---:|---:|---:|---:|
| Addison Barger | wOBA | 0.3785 | 0.3122 | 0.3231 | −0.0109 |
| Adrian Del Castillo | wOBA | 0.4239 | 0.3497 | 0.2990 | +0.0507 |
| Agustín Ramírez | wOBA | 0.3374 | 0.2784 | 0.3047 | −0.0264 |
| Aaron Ashby | FIP | 4.5405 | 4.8186 | 2.6650 | +2.1536 |
| Adam Mazur | FIP | 4.2148 | 4.4729 | 4.8667 | −0.3937 |
| Adrian Houser | FIP | 4.0623 | 4.3111 | 3.7720 | +0.5391 |

The fitted wOBA factor (from this backtest's own training fit, which
excludes the 2025 target season) is 0.8249 from 523 real training pairs
after the era floor; the FIP factor is 1.0613 from 312. These are lower than
the pre-floor values (0.84666 from 670 pairs; 1.13392 from 434) because the
excluded pre-2019 AAA cohort had unusually high raw FIP and hence pulled the
pooled factor upward; they will also differ from the full production
`data/outputs/factors.json` fit, which trains on every available season
rather than holding out 2025. These factors translate rates; they do not
forecast playing time, promotion, injuries, or pitcher roles.

## Scope and limits

- Evaluation selects players who actually received enough MLB playing time.
  Players who never reached MLB, or missed the thresholds, are excluded. Some
  players already had MLB experience. This is not a prospect promotion test.
- A player can appear in earlier training seasons and the target season. The
  split holds out future outcomes, not every prior observation of that person.
- Stats API normalization uses fixed 2024-scale wOBA event weights divided by
  PA, FIP constant 3.10, and approximate HR/FB = HR / (HR + air outs). Those
  values can differ from official season-specific advanced statistics.
- Training uses the existing age adjustment; translation applies the fitted
  factors to source rates without an additional age projection. Upstream ages
  may be inferred or defaulted. No park or pitcher-role adjustment is applied.
- Only AAA→MLB is tested. International links, other holdout years, and the
  statistical significance of improvements are not established by this run.
- Inputs are historical snapshots available now, not archived as-of-2024
  releases. The evaluator prevents outcome-season leakage in fitting but does
  not prove historical publication timing or rule out later data corrections.
- The bootstrap resamples training rows, not player clusters. Repeated players
  and league-era changes can limit interpretation of the reported intervals.
- Observed 80% interval coverage varies substantially by target season (see
  "Rolling seasons" above) and is not reliably close to nominal in all cases.
- The AAA 2019 era floor year comes from a known external league change (the
  ball switch), not from tuning against these results. But the diagnosis that
  motivated adding the floor examined 2022-2024 backtest results before the
  floor existed, so 2022-2024 is not a pristine selection set for this
  particular decision — it informed the hypothesis as well as evaluating it.
  2025 (see "AAA era floor" above) is the clean, held-out check: the floor
  year was fixed before 2025 was scored.

## Input identity

The report records these SHA-256 checksums and all training-pair identities.
The stale general `manifest.json` is not used to determine data provenance.

| Snapshot | SHA-256 |
|---|---|
| aaa_batters.csv | `c715c285a9bc0e217bd57310c66f0207624a5ce62bf391d597ef8e0033d7a4ed` |
| aaa_pitchers.csv | `e5915eab39018b250d4bcb5b1c259eb6046379966e37e2c4188e5399b72a4752` |
| mlb_batters.csv | `e206cd20b8b7785e271f957dcbd6811b9546254e96831ee07f4c0c79255a6ba9` |
| mlb_pitchers.csv | `f012308f507c64a428165eb458481b38c06967ff7b2994a066321acc1aecd210` |
