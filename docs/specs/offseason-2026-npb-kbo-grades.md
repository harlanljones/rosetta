# Spec: 2026-27 offseason update, NPB and KBO grades

Status: approved plan (Harlan, 2026-11-01), spec v1. Plan of record:
"grade players like Teruaki Sato" for the 2026-27 NPB/KBO posting class.

## Problem (measured on commit 3fc5c91)

- Sato 2026 NPB (559 PA, normalized wOBA .419) translates to .414. The NPB link is
  effectively identity.
- `transfers.csv` NPB/KBO movers: batters NPB->MLB 5, NPB->AAA 2, AAA->NPB 8,
  MLB->NPB 8, KBO->MLB 2, KBO->AAA 1, any->KBO 0. Pitchers NPB->MLB 9, NPB->AAA 5,
  AAA->NPB 3, MLB->NPB 4, any KBO 0.
- Shrinkage uses `prior_n` 25-50 pseudo-movers toward 1.0; 5 movers gives
  shrink ~0.1. Bootstrap is skipped below 5 rows (low == high == factor).
- `LeagueFactorModel.path()` prefers a fitted direct link, so the thin NPB->MLB
  link wins over any chain. `get_factor()` only inverts a reverse link when the
  forward link is missing; directions are never pooled.
- Every NPB/KBO row has age 27.0: `make live-data` passes `--no-with-ages`, and
  the ingests fall back to 27.0 silently.
- KBO names are romanized from Hangul, so foreign imports (e.g. a transliterated
  English surname) never match Chadwick. That is why KBO has zero imports.
- NPB/KBO home/road columns duplicate the overall line, so park handling is a no-op.

## Decisions (from the grill)

1. Targets: NPB and KBO, batters and pitchers.
2. Grades: wRC+ (batters) and FIP- (pitchers) with 80% bands; a 20-80 grade
   mapped from them; projected WAR and dollar value.
3. WAR inputs: scraped NPB/KBO fielding data.
4. Links: pool imports and exports in both directions; backfill to 2005;
   direction term reported.
5. Park: NPB/KBO park factors from game logs, home vs road runs, 3 years, regressed.
6. Validation: rolling refit by arrival year, scored on exports' first MLB season.
7. Ship: one release.

## Defaults chosen in this spec (reversible, flag in PR)

- D1. AAA->MLB production behavior is unchanged. Pooling and the new shrink
  target apply only to links with an NPB/KBO/CPBL/CUBA endpoint. The validated
  AAA->MLB backtest must reproduce bit-for-bit.
- D2. Shrink target for a thin international link is the chained product
  through AAA (e.g. NPB->AAA x AAA->MLB) when that chain is fitted, else 1.0.
- D3. 20-80 grade: 50 = MLB median of qualified players in the reference
  season, 10 points per standard deviation, clipped to 20-80, rounded to 5.
- D4. Playing time for WAR: 600 PA batters, 170 IP starters, 65 IP relievers.
- D5. Reference constants (wOBA scale, lg wOBA, R/PA, runs per win, lg FIP,
  FIP constant, $/WAR) live in a checked-in table
  `data/reference/mlb_constants.csv` with a `source` column. lg wOBA and lg FIP
  are computed from the MLB snapshots when the table lacks a season. Pitcher replacement level (`pitcher_repl_r9`) and the $/WAR
  value are NOT filled by this spec; Harlan picks it. Until then dollars render as
  "n/a" and the code path is tested with fixture constants.

## Work packages

Each package lists files it owns. Packages run on separate branches/worktrees
and merge in order. Tests first, all offline, `make test && make lint` green.

### WP1 Ages (Haiku)
Owns: `src/rosetta/ingest/npb.py`, `src/rosetta/ingest/kbo.py`, `Makefile`,
`src/rosetta/models/aging.py`.
- Add `age_source` to NPB/KBO rows: `birth_year` or `imputed`. Imputed rows keep
  age 27.0 for schema compatibility. (Column is added to `schema.py` in WP0.)
- `age_adjust_rate`: when either endpoint's age is imputed, return the rate
  unadjusted (no fake aging). Thread `age_source` through `cohort/builder.py`
  via the pre/post rows (coordinate with WP3: WP1 adds `pre_age_source` /
  `post_age_source` columns only).
- `make live-data`: drop `--no-with-ages`; seasons become `2005..2026` for NPB,
  KBO, AAA, MLB (comma list generated in the Makefile).
- New `rosetta check-snapshots`: fail if any real NPB/KBO row with pa >= 100 or
  ip >= 30 has `age_source == imputed`, and print the count by league-season.
- Tests: fixture-based birth-year parse already exists (`npb_player.html`); add
  imputed-flag tests and an aging no-op test.

### WP0 Schema (Opus, done before others)
Owns: `src/rosetta/schema.py`. Adds optional columns: `age_source`, `position`,
`def_games`, `def_po`, `def_a`, `def_e`, `def_inn`, and the transfers column
`direction`. Defaults keep old snapshots loadable.

### WP2 Backfill readiness (Haiku)
Owns: `src/rosetta/ingest/statsapi.py` (season-range only), `tests/` for it.
- Confirm NPB legacy page parser covers pre-2018 seasons (legacy fixture
  exists); add a guard that raises on an unparsed table instead of writing zero
  rows.
- `fetch-statsapi` accepts `--seasons 2005-2026` range syntax.
- Era floor stays: AAA endpoints before 2019 are still dropped for AAA links.
  NPB<->MLB and KBO<->MLB pairs from 2005+ are allowed.

### WP3 Movers both directions (Sonnet)
Owns: `src/rosetta/ingest/transfers.py`, `src/rosetta/ingest/chadwick.py`,
`src/rosetta/cohort/builder.py`, new `data/snapshots/kbo_foreign_map.csv`
(header only plus a documented format; rows are filled locally).
- Every pair gets `direction`: `export` (from NPB/KBO to AAA/MLB), `import`
  (AAA/MLB to NPB/KBO), `lateral` otherwise.
- KBO foreign-player crosswalk: `kbo_foreign_map.csv` maps KBO player id ->
  mlbam id. `generate-transfers` uses it before Chadwick. Document that the
  local refresh fills it from the English KBO site's roster pages.
- Holdout logic unchanged: most recent to_season held out only when earlier
  to-seasons exist.

### WP4 Pooled international links (Sonnet, after WP3)
Owns: `src/rosetta/models/factors.py`.
- For unordered pair {A, B} with A earlier in `LEAGUE_CHAIN`, when either is
  NPB/KBO/CPBL/CUBA: build one pooled sample in the A->B orientation. Forward
  rows as-is; reverse rows inverted (pre<->post, age_from<->age_to, weights
  unchanged). Estimate ratio-of-means on the pooled sample.
- Direction term: also estimate export-only and import-only factors; store
  `factor_export`, `factor_import`, `n_export`, `n_import`, and
  `direction_gap = log(factor_export / factor_import)` in the link node. Production
  factor is the pooled estimate.
- Shrink target per D2: `factor = target * (raw / target) ** shrink` in log
  space, where `target` is the chained AAA product or 1.0.
- Store both A->B and B->A (the latter as reciprocal, intervals reciprocal and
  swapped).
- Bootstrap resamples pooled rows; below 5 rows, interval comes from
  `resid_sd` only and a `thin` flag is set.
- `path()` keeps preferring direct links, which now carry pooled data.
- Tests: synthetic cohort where import-only and export-only factors are known;
  check pooled estimate, inversion, reciprocity, shrink-to-chain, and that an
  AAA->MLB-only cohort gives identical output to the old code.

### WP5 Park factors from game logs (Sonnet)
Owns: `src/rosetta/models/park.py`, new
`data/snapshots/game_logs_{npb,kbo}.csv` schema (header only).
- Game-log schema: `league, season, date, home_team, away_team, home_runs,
  away_runs, park_id, source, is_synthetic`.
- `estimate_runs_park_factors(logs, window=3, k=...)`: per park, runs per game
  at home (both teams) over runs per game on the road, rolling 3 seasons,
  regressed to 1.0; output `league, team, season, park_factor, games`.
- Translation applies half the park effect (`(1 + pf) / 2`) to wOBA, ISO, hr_pct
  for batters and FIP/ERA/hr_fb for pitchers, by team-season, before league
  factors. Applied identically in cohort building for NPB/KBO endpoints.
- Tests with in-test synthetic logs only.

### WP6 Grades (Sonnet)
Owns: new `src/rosetta/models/grades.py`, `src/rosetta/translate/api.py`,
new `data/reference/mlb_constants.csv`, `src/rosetta/schema.py`
(TranslationResult fields only; after WP0).
- `wrc_plus(woba, season)`: `100 * (((woba - lg_woba) / woba_scale) + lg_r_pa) / lg_r_pa`.
- `fip_minus(fip, season)`: `100 * fip / lg_fip`.
- `grade_20_80(value, dist)`: per D3, inverted for FIP-.
- WAR batter: `(batting_runs + pos_adj + def_runs + repl) / runs_per_win` at
  D4 playing time; `pos_adj` per 162 games: C +12.5, SS +7.5, 2B/3B/CF +2.5,
  LF/RF -7.5, 1B -12.5, DH -17.5, scaled by PA/700; `repl` 20 runs per 600 PA.
  `def_runs` from WP-L2 fielding (0 when missing, flagged).
- WAR pitcher: FIP-based, `((lg_fip - fip) / 9 * ip + repl_p) / runs_per_win`.
- Bands: carry the 80% wOBA/FIP band through each transform (monotone), so
  wRC+, FIP-, grade and WAR all get low/high.
- `TranslationResult` gains `wrc_plus*`, `fip_minus*`, `grade*`, `war*`,
  `dollars*` (each with low/high) and `position`.
- Tests with fixture constants.

### WP7 International validation (Sonnet, after WP4)
Owns: `src/rosetta/backtest/historical.py`, `src/rosetta/cli.py`
(new command only), new `docs/international-backtest-results.md` (generated).
- `rosetta international-rolling --league NPB|KBO --targets 2010-2026`:
  for each target season T, fit on all pooled pairs with to_season < T; score
  exports whose first MLB season is T (from-season T-1, allow AAA->MLB inside T
  as MLB only). Baseline = carry-forward foreign rate.
- Report per league x role x stat: n, MAE, baseline MAE, 80% coverage, plus
  import vs export factor gap per link.
- Ship gate (printed PASS/FAIL): exports per league beat baseline on wOBA
  (batters) and FIP (pitchers) MAE; coverage in [0.70, 0.90].

### WP8 Product (Haiku, after WP6)
Owns: `apps/leaderboard/**`, `scripts/export_demo.py`, `tests/test_showcase_app.py`,
`tests/test_export_demo.py`.
- Leaderboard columns: wRC+/FIP- with band, grade, WAR with band, $.
- Showcase: "2026-27 posting class" panel (players flagged in a small
  `data/reference/posting_class_2026.csv`, filled by Harlan), Sato first.
- "What it can't do" box: international validation numbers once WP7 runs.

## Local packages (need live sites; run on diocletian)

The cloud build session cannot reach npb.jp, koreabaseball.com or
statsapi.mlb.com. These run where the sites are reachable:

- L1 Game-log scrapers for NPB and KBO into the WP5 schema, with HTML fixtures.
- L2 Fielding scrapers for NPB and KBO (position, games, PO, A, E, innings)
  into the WP0 columns; a range-based `def_runs` (plays made per 9 vs league at
  position, times run value) with fixtures.
- L3 `rosetta fetch-chadwick` (the register is gitignored; without it the cohort has
  no NPB/KBO pairs at all), `make live-data` (2005-2026, ages on), fill `kbo_foreign_map.csv`,
  `rosetta generate-transfers`, `rosetta check-snapshots`, `make factors`.
- L4 `rosetta international-rolling` for NPB and KBO; commit the generated
  results doc; rebuild showcase data.

## Done when

- `make test && make lint` pass offline.
- AAA->MLB historical backtest output is unchanged.
- After L3/L4: no imputed ages on qualified NPB/KBO rows; ship gate PASS or the
  failing rows are named in the PR; Sato's card shows wRC+, FIP-n/a, grade, WAR
  with bands.
