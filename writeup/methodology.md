# Rosetta: Translating League Performance to MLB

**1,600 words · methodology, validation, limitations**

Rosetta is a public Major League Equivalency (MLE) engine. It answers a single front-office question: *if this NPB / KBO / CPBL / AAA line happened in MLB, what would the rates look like — and how sure are we?*

The short version of the method: find players who actually moved between leagues, age-adjust their component rates, estimate shrunken difficulty factors from those movers, chain the factors along CPBL → KBO → NPB → AAA → MLB, and report bootstrap uncertainty bands instead of fake precision.

---

## Why component rates, not counting stats

Counting stats (HR, R, ERA totals) confound playing time, role, and environment. Rosetta translates **rates first**:

| Hitters | Pitchers |
|---------|----------|
| BB%, K%, ISO, BABIP, HR%, wOBA | K%, BB%, HR/FB, FIP, ERA |

Strikeout rate is the most portable skill signal across leagues; BABIP and ERA are the least. We still translate the noisy ones — but shrinkage and wide bands keep them honest. After translation we recombine components into summary wOBA / FIP so a reader gets one headline number with a band.

We never multiply a 40-homer KBO season by a single “league factor.” That is how you invent prospect mythology.

---

## The transferred-player cohort (the Rosetta Stone)

Every league-strength factor comes from **matched movers**: same player, meaningful samples on both sides, adjacent seasons (or same-year promotions for AAA↔MLB).

For each mover and each component rate \(r\):

1. Park-aware seasonal lines enter a normalized schema (identical columns in every league).
2. Age-adjust the pre-move rate to the post-move age with a simple delta-method curve peaking near 27.
3. Form the ratio \(r^{\text{post}} / r^{\text{pre, age-adj}}\).
4. Weight by \(\min(\text{PA}_\text{pre}, \text{PA}_\text{post})\) (or IP for pitchers).
5. Average across movers → raw link factor.
6. **Shrink toward 1.0** with prior strength that is larger for noisy stats (BABIP, ERA) than for K%.

\[
\hat f = 1 + \frac{n}{n + n_0}(\bar f_{\text{raw}} - 1)
\]

Small international samples do not get to yell. That is the whole point of shrinkage.

### The chain

Direct NPB→MLB and KBO→MLB movers exist but are selected and sparse. AAA→MLB is the high-N anchor. Rosetta estimates adjacent links and multiplies:

\[
f_{\text{CPBL}\to\text{MLB}} = f_{\text{CPBL}\to\text{KBO}}\cdot f_{\text{KBO}\to\text{NPB}}\cdot f_{\text{NPB}\to\text{AAA}}\cdot f_{\text{AAA}\to\text{MLB}}
\]

When a direct link is observed with decent \(n\), we prefer it over a long product. Uncertainty compounds in log space along the path — long chains *look* precise as point estimates and fall apart as intervals. That is a feature.

---

## Aging and park

A 30-year-old KBO MVP line is not comparable raw to a 24-year-old’s. Before estimating factors we shift pre-move rates by a transparent quadratic aging delta. The curve is deliberately simple (peak 27, published slopes per component) so an interviewer can rewrite it on a whiteboard. Swap in Tango or Davenport empirical curves without changing the API.

Park factors are derived from home/road splits when published factors are missing, then regressed to 1.0. NPB’s Tokyo Dome vs. Koshien problem is real; league-average translation without park adjustment is a known bias in public work.

---

## Uncertainty as a product feature

Every MLE is a **distribution**. We bootstrap the mover cohort for each link, then combine link intervals along the path. Cuban and CPBL translations get very wide bands. That is honesty, not failure.

Calibration target: 80% intervals should cover ~80% of holdout outcomes. If coverage collapses to 40%, the model is overconfident — usually because shrinkage is too weak or the chain is too long.

---

## Validation design

Hold out the most recent transfer seasons (flagged `holdout` in the cohort; default to_season ≥ 2023). Fit factors on the rest. Predict post-move rates from pre-move rates only. Score:

- RMSE / MAE on translated wOBA and FIP/ERA
- Empirical coverage of 80% bands
- Residual slices by age, link, and role

`make backtest` runs this offline against committed snapshots so the number does not drift with the web.

### What “good” looks like on synthetic fixtures

The repo ships **labeled synthetic** snapshots shaped like real environments so CI is reproducible without scraping. On those fixtures the pipeline recovers the direction of true link multipliers (AAA offense comes down toward MLB; K% for hitters rises). Real-data backtests will be noisier — and should be reported as such.

---

## Where translations break

**Selection bias.** Posted players and free agents who clear MLB rosters are not random draws from NPB/KBO. The upper tail transfers; the median does not. MLE factors estimated on movers are slightly too harsh on ordinary players and still too kind on the true stars if scouting value is orthogonal to the rates. Say this out loud in every interview.

**Role changes.** NPB aces become MLB relievers. Rosetta translates rates, then leaves playing-time/role as a separate layer (v1 reports rates + FIP; a role model is phase 2).

**Age extremes.** Teenagers and age-34 “last chance” seasons sit where the quadratic curve is least trustworthy.

**Thin leagues.** Cuba and CPBL produce wide bands by construction. Do not ship a single Cuban wOBA point estimate in a slide deck without the interval.

**Ball and rule regime shifts.** MLB ball changes and NPB foreign-player limits break stationarity. Prefer recent cohorts for current acquisition work; longer cohorts for historical research.

---

## Relation to published work

Rosetta’s v1 is intentionally classical: James → Szymborski-style MLE arithmetic, Davenport’s multi-environment seriousness, Tango’s selection-bias critique, modern component-factor tables (e.g. public MiLB factor sheets). We did **not** start with hierarchical Bayes. Bayes is the right phase-2 upgrade once the data platform and pair quality justify the ceremony. For a hireable public repo, transparency beats elegance.

Sources and formulas are collected in `docs/research/mle-methodology.md`.

---

## What proprietary data would buy

With org-level pitch tracking, medicals, and amateur academy feeds you would:

1. Replace synthetic/BR scrapes with internal seasonal warehouses.
2. Add pitch-level translation (stuff + command) where tracking exists.
3. Model selection into the transfer cohort explicitly (Heckman / Bayesian selection).
4. Fit hierarchical league effects with player random intercepts.
5. Feed MLEs as priors into a rest-of-season / winter projection system.

Rosetta is the open skeleton those systems hang on.

---

## How to run the evidence

```bash
pip install -e ".[dev,web]"
make snapshots   # if data/snapshots empty
make backtest    # factors + holdout metrics
make leaderboard # JSON for the web app
make serve-leaderboard
```

CLI:

```bash
rosetta translate "NPB Hitter 1" NPB --role batter
rosetta cohort-summary
```

Python:

```python
from rosetta import translate
from rosetta.models import LeagueFactorModel

model = LeagueFactorModel.load("data/outputs/factors.json")
# or: translate(player, from_league, factors_path=..., snapshot_dir=...)
```

---

## Bottom line

League translation is the canonical “do you understand baseball analytics” problem because it forces you to confront sample size, selection, aging, parks, and uncertainty at once. Rosetta’s bet is that a **clear, shrunken, component-wise, bootstrap-banded** chain — validated on held-out movers — is more hireable than a black-box model that cannot explain why a CPBL star’s MLE wOBA moved eight points.
