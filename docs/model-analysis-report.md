# Model analysis report

The leaderboard’s `/analysis` page is a compact, human-readable error report
for the real-data rolling backtest. It connects sabermetric questions (“is the
translation useful for this player?”) to model diagnostics (“how large was the
residual, did the interval contain the outcome, and did the model beat the
prior AAA value?”).

## What the page reports

- 2026 wOBA and FIP scorecards against the prior AAA value.
- Player-level signed error, absolute error, baseline improvement, interval
  status, and stat-specific error buckets.
- Rolling 2022–2026 summaries, with the baseline shown beside each model MAE.
- The largest current-season and historical wOBA/FIP misses.
- Guardrails distinguishing observed error patterns from causal explanations.

`analysis.json` is exported by `scripts/export_demo.py` from the player-level
`historical-rolling.json` detail. It is real-data-only and carries
`schema_version: "analysis.v1"`.

## Interpretation rules

`signed_error = prediction - actual`; positive means the model overpredicted.
`improvement = abs(prior - actual) - abs(prediction - actual)`; positive means
Rosetta beat the persistence baseline. Error buckets are scale-aware: wOBA
uses `.025/.050/.075`, while FIP uses `.5/1.0/1.5`.

The page deliberately avoids claiming that a residual identifies a cause.
Role changes, parks, injuries, playing time, and sample noise require human
review and are not inferable from the backtest alone. The 80% band is a cohort
calibration signal, not an individualized probability of success.

Regenerate the bundle with:

```bash
make historical-backtest
make historical-rolling
make showcase-data
```
