# Model analysis report

The leaderboard’s `/analysis` page is a compact, human-readable error report
for the real-data rolling backtest. It connects sabermetric questions (“is the
translation useful for this player?”) to model diagnostics (“how large was the
residual, did the interval contain the outcome, and did the model beat the
prior AAA value?”).

## What the page reports

- Current-season wOBA and FIP scorecards against the prior AAA value.
- Player-level signed error, absolute error, baseline improvement, interval
  status, and stat-specific error buckets.
- A stat-by-stat calibration table covering bias, interval coverage versus the
  nominal 80% target, baseline win rate, interval width, and error-size buckets.
- A player drilldown that keeps the prior, prediction, outcome, uncertainty
  band, and baseline improvement in the same native stat units.
- Rolling target-season summaries, with the baseline shown beside each model MAE.
- The largest current-season and historical wOBA/FIP misses.
- Guardrails distinguishing observed error patterns from causal explanations.

`analysis.json` is exported by `scripts/export_demo.py` from the player-level
`historical-rolling.json` detail. It is real-data-only and carries
`schema_version: "analysis.v2"`. The export derives the number of baseline wins
from the metric rows, so the demo does not rely on a hardcoded statistic count.

## Interpretation rules

`signed_error = prediction - actual`; positive means the model overpredicted.
`improvement = abs(prior - actual) - abs(prediction - actual)`; positive means
Rosetta beat the persistence baseline. `bias` is the mean signed error in the
stat's native units. Error buckets are scale-aware: rate stats use
`.025/.050/.075`, while FIP uses `.5/1.0/1.5`. Coverage is compared with the
nominal 80% target and is reported per stat; unlike the old role summary, the
page never pools wOBA and FIP into one mixed-unit number.

The page deliberately avoids claiming that a residual identifies a cause.
Role changes, parks, injuries, playing time, and sample noise require human
review and are not inferable from the backtest alone. The 80% band is a cohort
calibration signal, not an individualized probability of success.

## Current calibrated rolling result

The current `historical-rolling.json` artifact uses the leakage-safe
`prior_affine` calibration: each stat's correction is fit from completed target
seasons strictly earlier than the season being scored, with a minimum of 20
rows. On the 2026 holdout, calibrated wOBA MAE is `0.0322` versus `0.0670` for
the persistence baseline, with `83.4%` interval coverage. Calibrated FIP MAE is
`0.640` versus `0.975`, with `88.2%` coverage.

Across the pooled 2022–2026 rolling folds, wOBA MAE is `0.0381` versus `0.0775`
and FIP MAE is `0.797` versus `1.016`; coverage is `82.8%` and `85.1%`,
respectively. These are internal adjacent AAA → MLB holdouts over players with
observed MLB playing time. The target-season sample is partial, the correction
is estimated from a small historical cohort, and FIP remains more variable than
wOBA. The result is evidence that the calibrated translation improves this
backtest, not a guarantee for new players or an apples-to-apples ranking against
MLB-only projection systems.

This is a retrospective evaluation, not an untouched model-selection holdout:
the 2026 fold was inspected while comparing candidate model and interval rules.
The per-fold `prior_affine` correction still fits only target seasons earlier
than the scored season, but the 2026 inspection limits how strongly this result
should be generalized.

## External preseason comparison

The closest external benchmarks are FanGraphs' 2026 preseason Steamer and ZiPS
tables, with FanGraphs Depth Charts as a playing-time-adjusted composite. The
FanGraphs projection database also exposes ATC, THE BAT, and OOPSY tables, and
its player tables include wOBA for hitters and FIP for pitchers. These systems
are useful comparison targets, but they answer a different question: they are
primarily MLB projection systems, while Rosetta translates a foreign-league or
AAA line before an MLB track record is established.

The demo therefore links the comparison set without publishing a fake
cross-system scorecard. A defensible MAE/coverage comparison requires one dated
preseason snapshot for every system, a player-ID crosswalk, the same
AAA-to-MLB cohort, and an explicit playing-time rule. FanGraphs makes tables
viewable but keeps bulk export behind membership and does not provide a
supported public projections API. PECOTA's public materials use BP-native
metrics and its 2026 spreadsheet is subscriber-gated; Marcel is public but
does not expose native wOBA/FIP. Until those artifacts are captured and
normalized, current FanGraphs/PECOTA values should be treated as methodology
context, not as apples-to-apples outcome scores.

Sources: [FanGraphs 2026 projection database](https://www.fangraphs.com/projections),
[FanGraphs system descriptions](https://blogs.fangraphs.com/all-the-2026-projections-are-in/),
[PECOTA projections](https://www.baseballprospectus.com/pecota-projections/), and
[Marcel 2026 projections](https://www.baseball-reference.com/leagues/majors/2026-projections.shtml).

Regenerate the bundle with:

```bash
make historical-backtest
make historical-rolling
make showcase-data
```
