# Showcase page spec — "Rosetta: predicting MLB performance" (2026 edition)

Agreed spec for a portfolio showcase page. Nothing is built yet; this document
is the contract for a future build, whether in this repo or a portfolio repo.

**Goal:** one static page that proves the model works, readable by
non-technical visitors, rigorous enough for technical ones. Leads with the
real 2026 backtest table.

**Data caveat (show on page):** 2026 MLB season-to-date through ~Sep 21,
2026 — outcomes are partial-season, not final. Backtest = 2025 AAA inputs →
2026 MLB-to-date outcomes (151 batters, 93 pitchers, thresholds 80 PA /
30 IP both sides).

## Data contract (from `scripts/export_demo.py` bundle)

Three JSON files, copied verbatim into the static site. No live server, no
fetch at view time:

- `meta.json` — `data_mode: "real"`, `generated_at`, `rosetta_git_commit`,
  estimator (`ratio_of_means`), era floor (`{AAA: 2019}`), source counts,
  input SHA-256, limitations list.
- `leaderboard.json` — 2026 season, `data_mode: real`, top-50
  batter/pitcher boards (MLE wOBA/FIP + 80% bands, `from_league`, sample
  PA/IP).
- `backtest.json` — `target_season_2026.metrics` (11 role/stat rows: n, mae,
  baseline_mae, rmse, coverage_80), rolling 2022–2026 (per-season +
  pooled), `player_predictions` (wOBA + FIP rows sorted by name).

**Integration note for the build:** `export_demo.py` currently hardcodes
2025 paths/keys (`historical-backtest/`, `target_season_2025`,
`rolling_2022_2025`, leaderboard season 2025). Generalize to a
`--target-season 2026` parameter: read `historical-backtest-2026/`, emit
`target_season_2026` / `rolling_2022_2026`, export the leaderboard with
`--season 2026`. No rounding/field changes needed.

## Page structure (single scroll, plain language first)

1. **Hero + hook (non-technical):** "We turned 2025 minor-league numbers
   into 2026 MLB predictions — and beat the naive guess 11 times out of
   11." One-line sub: naive guess = "just carry last year's AAA rate
   forward."
2. **2026 backtest headlines:** 2-row table (Batter wOBA: 0.0368 vs 0.0670,
   n=151, 86.1% coverage · Pitcher FIP: 0.9184 vs 0.9746, n=93, 84.9%)
   + side-by-side CSS/SVG bar chart (model MAE vs baseline MAE, lower
   wins). Numbers from `backtest.json → target_season_2026`.
3. **Expandable full sweep:** `<details>` "All 11 stats" with the complete
   table (babip/bb/hr/iso/k/woba × era/fip/hr_fb/bb/k + coverage). This is
   the 11/11 proof for experts.
4. **How to read it (inline, no glossary page):** one-line definitions
   where terms first appear + `<abbr title>` tooltips — MAE ("average miss
   size, lower is better"), baseline, 80% coverage ("how often the true
   number landed inside our uncertainty band"), wOBA ("overall hitting
   production"), FIP ("pitching skill stripped of fielding luck").
5. **Trend strip:** mini 2022–2026 rolling table (wOBA/FIP MAE vs baseline
   per year) showing wOBA wins 5/5, FIP 4/5 — with the 2024 FIP miss (1.25
   vs 1.10) visible, not hidden.
6. **"What it can't do" box (always visible):** 2024 FIP loss, pooled FIP
   still ~tied (1.027 vs 1.016 over 2022–26), AAA→MLB only (KBO/NPB
   unvalidated), selects players who actually got MLB time (not a
   promotion model), no playing-time/injury forecast, intervals approximate
   (rows not player-clusters), 2026 outcomes partial-season.
7. **Payoff — 2026 MLE top-5s:** batters (Sato .414, Kondoh .412,
   Morishita .409, Maki .400, Kurihara .390, NPB) and pitchers (Kwak 2.49
   KBO, Takahashi 2.50, Saiki 2.54, Taira 2.67, Sumida 2.74) with 80%
   bands + "full board lives in the app" link.
8. **How it works (collapsed `<details>`):** ratio-of-means link factors,
   AAA 2019 era floor (livelier ball), 250 bootstraps, train
   `to_season < 2026`, real-rows-only.
9. **Reproduce footer:** `make historical-backtest` / `historical-rolling`
   / `leaderboard` commands, `meta.json` commit + checksums +
   generated_at.

## Presentation rules

- Static HTML + inline CSS/SVG bars; no JS framework, no chart
  dependency. Responsive, `prefers-reduced-motion` respected, semantic
  table markup.
- Styling belongs to the portfolio repo — this spec fixes content order
  and data bindings, not visual tokens.
- Acceptance: page renders from the three JSON files alone; `data_mode`
  must read `"real"` or the page shows an error state instead of numbers;
  all 11/11 values match `backtest.json` exactly as exported.
