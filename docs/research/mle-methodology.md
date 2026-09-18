# Major League Equivalency (MLE) & Cross-League Baseball Translation

**Research report for a hireable portfolio v1 implementation**  
**Scope:** minor-league levels, NPB, KBO, thin international leagues (CPBL, Cuba/SNB), and chained translations  
**Date:** 2026-03-28  

---

## 1. Executive summary

Major League Equivalencies (MLEs) convert observed performance in one baseball environment into the *statistical shape* that performance would have had in MLB (or another target league). They are **translations of what happened**, not pure forecasts—though good MLEs are a primary input to projections (PECOTA, ZiPS, Marcel-style systems, org models).

Published systems share a common pipeline:

1. Park-neutralize component rates  
2. Adjust for league run environment  
3. Apply a **talent / difficulty** factor estimated from transferred players  
4. Optionally age-adjust and role-adjust (SP→RP)  
5. Shrink noisy rates and report uncertainty  

**Portfolio recommendation (see §9):** **(A) age-adjusted rate ratios + shrinkage + bootstrap**, with multiplicative **league chains** (e.g. CPBL→KBO→NPB→AAA→MLB) and component-specific factors. It is the right v1: implementable in pure Python/pandas, auditable, bootstrap-uncertainty-ready, and competitive with classical James/Szymborski/Davenport practice while incorporating modern Tango-style critiques (selection bias, regression).

---

## 2. What an MLE is (and is not)

| Is | Is not |
|----|--------|
| A contextual re-expression of past stats | A guaranteed future projection |
| Useful as a **baseline** for forecasting | A substitute for scouting / pitch-tracking |
| Comparable across levels when methodology is fixed | Interchangeable across publishers (factors differ) |
| Strongest for stable skill rates (K%, BB%) | Reliable for raw AVG, ERA, RBI |

Bill James introduced batter MLEs in the *1985 Baseball Abstract* and argued that well-built MLEs predicted future MLB batting average about as well as one MLB season predicted the next (~25 points of BA noise). Dan Szymborski later published a practical calculation recipe; Clay Davenport built full minor/international **Davenport Translations (DTs)** feeding PECOTA; modern public work (FanGraphs library, Ogilvie factors, Japan Baseball Lab) uses **per-component rate factors** rather than a single “offense multiplier.”

---

## 3. How published systems estimate league strength

### 3.1 Matched-transfer ratios (classical core)

**Idea:** Find players with meaningful samples in League A and League B (same year or adjacent years). Park-neutralize both sides. Form weighted ratios of rates (or odds). Aggregate with harmonic-mean or min(PA) weights.

\[
f_{A \to B}^{(c)} = \frac{\sum_i w_i \, r_{i,B}^{(c)}}{\sum_i w_i \, r_{i,A}^{(c)}}
\quad\text{or odds-ratio form for event rates}
\]

- **Same-year promotions** minimize aging contamination but suffer selection bias (hot AAA hitters get called up; cold MLB hitters get optioned).  
- **Year-to-year foreign transfers** (NPB→MLB) need explicit **age adjustment** because the second season is one year later.  
- Clay Davenport (2025) documents AAA↔MLB same-year pairs 2001–2025: ~42 points of EqA drop on park/offense-only (“nodif”) translations, with selection bias **inflating** the apparent difficulty gap.

### 3.2 Odds-ratio / log-odds talent model (Tango / Seamheads)

For binary-ish outcomes (K, BB, HR per opportunity), multiplicative **odds ratios** behave better than raw rate ratios near boundaries:

\[
\text{odds}_B = \text{odds}_A \times OR_{A\to B}, \quad
p = \frac{\text{odds}}{1+\text{odds}}
\]

Seamheads’ public MLE spreadsheet used Tango-style odds-ratio batting translations and Kubatko-style pitching neutralization. This is the natural language for chaining leagues without driving rates outside [0,1].

### 3.3 Multiplicative difficulty chains

When direct A→MLB transfers are rare (CPBL, Cuba), estimate adjacent links and multiply:

\[
f_{\text{CPBL}\to\text{MLB}} = f_{\text{CPBL}\to\text{KBO}} \cdot f_{\text{KBO}\to\text{NPB}} \cdot f_{\text{NPB}\to\text{AAA}} \cdot f_{\text{AAA}\to\text{MLB}}
\]

**Variance of a product of independent link estimates** (log space):

\[
\mathrm{Var}(\log \hat f_{\text{path}}) \approx \sum_{\ell \in \text{path}} \mathrm{Var}(\log \hat f_\ell)
\]

Long chains → wide uncertainty bands even if each link looks precise. Prefer the **shortest path with adequate transfer N**, and shrink extreme links toward 1.

### 3.4 Bayesian hierarchical models

Pool component rates across leagues with league-level random effects, player random effects, and park effects:

\[
\begin{aligned}
r_{ij} &\sim \mathrm{BetaBinomial}(\mu_{ij}, \phi) \\
\mathrm{logit}(\mu_{ij}) &= \alpha_{\text{player}} + \beta_{\text{league}} + \gamma_{\text{park}} + \delta_{\text{age}} + \cdots
\end{aligned}
\]

Posterior predictive for “what if this PA were in MLB” is the gold standard for thin leagues—but heavy for a portfolio v1 (Stan/PyMC, careful priors, more data engineering).

### 3.5 Regression with covariates

OLS/GLM of MLB outcome rates on minor/foreign rates plus age, role, handedness, velocity, etc. Flexible and good when you have rich features; easy to overfit sparse international samples and harder to explain to non-stats audiences than transparent factors.

### 3.6 Comparable-player systems (PECOTA)

PECOTA does not *replace* MLEs; it **consumes** Davenport translations of minor/international seasons, then finds historical comps. Translation quality still gates prospect and foreign-player forecasts.

---

## 4. Clay Davenport approach (specifically)

Clay Davenport’s long-running work (Baseball Prospectus DTs; now claydavenport.com) is the most complete public-facing translation tradition.

### 4.1 Design goals

- Put every batter/pitcher season on a **common major-league scale**  
- Historical baseline often described as a **neutral park, mid-1990s NL-like environment** (EqA scale)  
- Support PECOTA and historical “what-if” work (Negro Leagues, Japan, Mexico, fall/winter leagues)

### 4.2 Building blocks

1. **Park and league offense adjustments** (“nodif” translations): strip environment without applying talent discount.  
2. **League quality / difficulty** from multi-year transition studies.  
3. **EqR / EqA**: run-creation and a batting-average-scaled rate that travels across eras better than raw OPS.  
4. **International reviews**: explicit essays on Japanese and Mexican translations; winter/fall league difficulty estimates.  
5. **Validation**: Davenport published DT vs MLE validation work (BP, 1998+) arguing translations track subsequent MLB performance.

### 4.3 Selection bias (Davenport 2025 AAA study)

On 3,631 same-year AAA–MLB pairs (≥50 PA each side, 2001–2025):

| Finding | Detail |
|---------|--------|
| Baseline gap | ~42 EqA points AAA → MLB on nodif stats |
| Time trend | Gap roughly stable since mid-2000s (~39–46 EqA) |
| Age | Slightly smaller drops for the youngest call-ups |
| Bias direction | Overstatement of difficulty: lucky AAA performers promoted; unlucky MLB performers demoted |
| Implication for MLE builders | Naïve transfer ratios need **shrinkage toward less extreme difficulty** and/or explicit selection models |

Davenport also shows catastrophic debut gaps are often recoverable (Zobrist, Tucker, Pedroia-class examples)—reinforcing that MLEs describe **expected translation of true talent**, not destiny from a 80-PA sample.

### 4.4 Practical takeaway from Davenport

Treat DTs as:

- Component-aware environmental + talent adjustments  
- Always paired with **aging** when used for projection  
- Best used as **regressed baselines**, not literal counting-stat destiny  

Exact proprietary factor tables are not fully open; public implementers reconstruct factors from Baseball-Reference/FanGraphs/Statcast promotion cohorts (see Ogilvie 2018–2025 calibration).

---

## 5. Classical James / Szymborski calculation (worked structure)

Szymborski’s public “How to Calculate MLEs” (Baseball Think Factory) operationalizes James:

1. **PL (park/league run context)**  
   Estimate runs/game context of the minor environment vs target MLB league.  
2. **Difficulty retention \(m\)**  
   Classic rule of thumb: batter retains ~**82%** of relative offense AAA→MLB (era-specific; re-estimate).  
   \[
   m = PL \times d_{\text{level}}, \quad M = \sqrt{m}
   \]
   Hits often scaled with \(M\); HR/BB/R/RBI with \(m\); triples extra-penalized; K mildly inflated (~1.05).  
3. **Outs conservation for AB**  
   Keep minor-league outs (AB−H), add translated hits → new AB (prevents BA from being a pure scale of H).  
4. **Target MLB park multipliers** if projecting into a specific stadium.  

**Modern critique:** single \(m\) for “offense” is too coarse. Prefer **separate factors** for BB%, K%, 1B/BIP, XBH/BIP, HR/BIP (Ogilvie; Japan Baseball Lab).

---

## 6. Aging, parks, selection bias, SP→RP

### 6.1 Aging curves

- Peak batting skill roughly mid-to-late 20s; pitchers similar but role- and stuff-dependent.  
- **Do not bake full aging into MLE factors** estimated from young promotion cohorts if you will age again in the projection engine (double-counting).  
- For **cross-season** foreign transfers, age-adjust one side before estimating league factors.  
- Ogilvie uses ~**±1.0–1.5% per year** on offensive components (not K%) vs level-average age.  
- Public KBO tools often use coarse bins: ≤25 +5%, 26–28 neutral, 29–31 −5%, 32+ −10% (rough; replace with continuous curve).  
- SABR Analytics work notes **KBO aging can differ** from MLB/NPB (earlier decline patterns on some metrics)—league-specific curves matter.

### 6.2 Park factors

Order of operations (consensus):

1. Split or season park factors (multi-year regressed; component-specific if possible)  
2. League run environment  
3. Talent/difficulty  
4. Optional destination park  

Minor-league parks are noisy (incomplete home/road history, short seasons). Prefer **regressed multi-year component PF**. NPB parks (Jingu, Yokohama, Kyocera, PayPay) can swing HR/ERA enough to wreck translations if skipped.

### 6.3 Selection bias (Tango’s critique)

Tom Tango’s “Issues with MLEs” remains required reading:

1. **Selective sampling** — only certain profiles are promoted/retained  
2. **Sample-size / weighting** — harmonic PA weights vs equal weight with cutoffs  
3. **Regression to the mean** — both sides of a transition are noisy; which mean?  
4. **No clean control group**  
5. **Biased context** — platoon hiding, feasting on weak pitching, park×profile interactions  

Mitigations for v1:

- Weight pairs by \(\min(PA_A, PA_B)\) or harmonic mean  
- **Regress observed rates before ratio** (Empirical Bayes)  
- Shrink league factors toward 1 with strength ∝ transfer N  
- Stratify by age and rough skill tier when N allows  
- Prefer same-year pairs for MiLB; age-adjust foreign pairs  
- Report **intervals**, not point MLEs alone  

### 6.4 Starter → reliever (and reverse)

Relief innings concentrate leverage, max effort, and platoon advantages:

| Effect | Typical direction |
|--------|-------------------|
| K% | Up as RP |
| BB% | Mixed; often slightly up |
| HR/9 | Often down (shorter stints, timing) |
| ERA/FIP | Improves vs SP baseline for same stuff |
| IP durability | Not interchangeable |

**Rule:** estimate SP and RP translation factors **separately** when possible. If a foreign SP will open in MLB relief, apply a **role delta** after league translation (org-specific; even a simple +1.0 to +1.5 K% pts and −0.1 to −0.3 ERA-equivalent is better than ignoring role). Do not treat KBO/NPB starter ERA as RP ERA.

---

## 7. Which rates translate best?

Synthesizing Japan Baseball Lab NPB work, Ogilvie MiLB factors, KBO open tools, and long-standing DIPS/Marcel intuition:

| Rate / skill | Translation fidelity | Notes |
|--------------|---------------------|-------|
| **BB% (hitters)** | Excellent | Best single offensive carry metric NPB→MLB |
| **K% (hitters)** | Good | Expect **inflation** vs weaker leagues (Ogilvie AAA K% factors ~1.20–1.29) |
| **Pitcher K% / K/9** | Good | Stuff+miss translates; still park/ball effects |
| **Pitcher BB% / BB/9** | Good | Command travels; MLB hitters punish nibbling differently |
| **GB/FB tendencies** | Good | Stable profile; feeds BABIP expectation |
| **ISO (park-adj)** | Moderate | Needs park + power discount; better than raw HR |
| **HR totals / HR%** | Poor–moderate | Parks, ball, and league HR environments dominate |
| **BABIP / AVG** | Poor | Defense, luck, BIP quality mix shift hard |
| **ERA / RA9** | Poor | Use as check only; prefer FIP-like components |
| **RBI, wins, saves** | Very poor | Context stats—do not translate |
| **Breaking-ball shape (NPB)** | Poor | Seam-height ball change; splitters more stable |

**Ogilvie AAA (IL) example multipliers** (MiLB rate × factor → park-neutral MLB-equivalent rate; 2018–2025 promotion cohorts):

| BB% | K% | 1B/BIP | 2B/BIP | HR/BIP |
|-----|-----|--------|--------|--------|
| 0.69 | 1.20 | 0.93 | 0.83 | 0.84 |

**Illustrative KBO→MLB batter factors** (public tool aggregates; re-estimate yourself):

| AVG | OBP | SLG | HR | K rate | BB rate | ISO |
|-----|-----|-----|----|--------|---------|-----|
| 0.86 | 0.87 | 0.87 | 0.79 | 1.10 | 0.90 | 0.88 |

**NPB fantasy-lab rules of thumb:** park-adjust first; HR × ~0.65–0.75; AVG −.020 to −.040; K% +3–5 pts; BB% ~unchanged; NPB ERA +~0.8–1.2 after park (pitch-mix dependent).

**Implementation priority:** translate **K%, BB%, HR/PA or HR/BIP, XBH structure**, then rebuild slash line / ERA—from components—not the reverse.

---

## 8. Uncertainty for thin leagues (CPBL, Cuban SNB, independent ball)

| League tier | Transfer density | Recommended approach |
|-------------|------------------|----------------------|
| AAA / AA | High | Direct empirical factors + light shrink |
| A / Rok | Medium | Direct + stronger shrink; chain via AA/AAA optional |
| NPB | Medium (stars + depth pieces) | Direct NPB↔MLB with age adj; park critical |
| KBO | Low–medium | Direct where possible; partial chain via NPB |
| CPBL | Low | **Chain** + heavy hierarchical shrink; wide CIs |
| Cuba / SNB | Very low / biased | Scouting + sparse pairs; huge priors toward replacement |
| Independent | Low, odd ages | Treat like low minors; distrust power numbers |

**Extra uncertainty sources in thin leagues:**

- Non-random emigrations (only stars or only failures leave)  
- Different balls, mound/enforcement, schedule intensity  
- Incomplete park factors  
- Age fraud / listing error risk historically  
- Tiny PA overlap for multi-link chains  

**Reporting standard:** always publish MLE ± bootstrap/posterior interval and a **confidence tier** (A/B/C) based on path length and transfer N.

---

## 9. Recommendation for hireable portfolio v1

### Choose **(A) age-adjusted rate ratios + shrinkage + bootstrap**

| Criterion | A: ratios + shrink + bootstrap | B: Bayesian hierarchical | C: regression + covariates |
|-----------|--------------------------------|--------------------------|----------------------------|
| Time to working demo | Days | Weeks | 1–2 weeks |
| Explainability in interviews | Excellent | Good if visualized | Medium |
| Thin-league behavior | Good with chains + shrink | Best asymptotically | Overfits easily |
| Uncertainty | Bootstrap/path variance | Full posterior | Bootstrap/sandwich |
| Data needs | Transitions + parks | Same + priors discipline | Needs rich features |
| Python deps | pandas/numpy | PyMC/Stan | sklearn/statsmodels |

**Why A wins for v1**

1. Matches industry-classical MLEs (James, Szymborski, Davenport empirical core, Ogilvie).  
2. Implements Tango’s main fixes (regress rates, shrink factors, weight pairs, show error).  
3. Chain multiply is trivial and honest about CPBL-class uncertainty.  
4. Leaves a clean upgrade path: v2 = hierarchical Bayes on the same design matrix (B), or stuff-augmented regression (C).  

**Do not pick B first** unless the portfolio’s point is Bayesian workflow. **Do not pick C first** without Statcast/Trackman features—covariate models without them underperform simple rates.

---

## 10. Practical Python formulas

### 10.1 Park-neutral rate

```python
def park_neutralize(rate, park_factor, mode="multiplicative"):
    """park_factor: 1.0 = neutral; 1.05 = 5% higher event rate at home context.
    For full-season rates already mixing home/road, use regressed season PF.
    """
    if mode == "multiplicative":
        return rate / park_factor
    raise ValueError(mode)
```

### 10.2 Empirical Bayes shrinkage of a rate (binomial)

```python
def shrink_rate(successes, trials, prior_mean, prior_strength):
    """prior_strength = equivalent trials (e.g. 100–300 PA for K%/BB%)."""
    return (successes + prior_mean * prior_strength) / (trials + prior_strength)
```

Typical prior_strength starting points (tune on backtest):

- K%, BB%: 200–400 PA  
- HR/PA: 300–600 PA  
- BABIP: 500+ BIP  

### 10.3 Age adjustment before estimating cross-season factors

```python
def age_adjust_rate(rate, age, ref_age=27, pct_per_year=0.012, higher_is_better=True):
    """Move observed rate toward ref_age skill before cross-year ratio."""
    years = age - ref_age
    delta = 1.0 - pct_per_year * years  # younger -> slight boost when projecting up
    if not higher_is_better:  # e.g. K% for hitters, ERA
        delta = 1.0 + pct_per_year * years
    return rate * delta
```

### 10.4 Transfer factor with min-PA weights

```python
import numpy as np
import pandas as pd

def estimate_factor(df, col_a, col_b, pa_a, pa_b):
    """df rows = player-seasons paired across leagues A and B."""
    w = np.minimum(df[pa_a], df[pa_b]).astype(float)
    # optional: require min PA
    mask = w >= 40
    w, a, b = w[mask], df.loc[mask, col_a], df.loc[mask, col_b]
    return np.average(b / a.replace(0, np.nan), weights=w)
```

### 10.5 Odds-ratio translation (preferred for bounded rates)

```python
def apply_odds_factor(p, odds_factor):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    new_odds = odds * odds_factor
    return new_odds / (1 + new_odds)
```

### 10.6 Shrink factor toward 1 (or hierarchical prior)

```python
def shrink_factor(f_hat, n_eff, tau=50.0):
    """n_eff ~ total weighted PA/100; tau controls pull to 1.0."""
    alpha = n_eff / (n_eff + tau)
    return alpha * f_hat + (1 - alpha) * 1.0
```

### 10.7 League chain multiply + log-space uncertainty

```python
def chain_factors(factors, se_log=None):
    """factors: list of multiplicative links CPBL->KBO->...->MLB
    se_log: optional list of SEs on log(factor)
    """
    f = np.prod(factors)
    if se_log is None:
        return f, None
    se = np.sqrt(np.sum(np.square(se_log)))
    # 90% CI on factor
    lo, hi = np.exp(np.log(f) + se * np.array([-1.645, 1.645]))
    return f, (lo, hi)

# Example path priors (PLACEHOLDERS — calibrate on data)
# Direction: multiply source_rate by factor to get MLB-equivalent rate for "good" events
# For hitter K%, factors are often > 1 (more Ks in MLB).
CHAIN_ISO = {
    "CPBL->KBO": 0.92,
    "KBO->NPB": 0.94,
    "NPB->AAA": 0.96,
    "AAA->MLB": 0.85,  # illustrative power retention
}
```

### 10.8 Component MLE rebuild (hitter)

```python
def mle_hitter(pa, bb_rate, k_rate, hr_per_pa, bip_1b, bip_xbh, factors, prior):
    """factors: dict of league factors per component; already chained to MLB."""
    bb = shrink_rate(bb_rate * pa, pa, prior["bb"], prior["n"]) * factors["bb"]
    k  = shrink_rate(k_rate * pa, pa, prior["k"], prior["n"]) * factors["k"]
    # re-normalize if bb+k > 0.95 etc.
    hr = shrink_rate(hr_per_pa * pa, pa, prior["hr"], prior["n_hr"]) * factors["hr"]
    # apply BIP factors to non-HR BIP structure ...
    return dict(bb_rate=bb, k_rate=k, hr_per_pa=hr)
```

### 10.9 Bootstrap CI for a league factor

```python
def bootstrap_factor(df, col_a, col_b, pa_a, pa_b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(df)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(estimate_factor(df.iloc[idx], col_a, col_b, pa_a, pa_b))
    return np.mean(stats), np.quantile(stats, [0.05, 0.95])
```

### 10.10 Full v1 pipeline (pseudocode)

```text
for each player-season in source league:
  1. parse component rates (BB%, K%, HBP%, HR/PA, 1B/BIP, 2B/BIP, 3B/BIP)
  2. park-neutralize with regressed component PF
  3. shrink rates toward source-league age-peer means
  4. multiply by chained (or direct) component factors → MLB-neutral rates
  5. optional: age to projection season; optional SP/RP role delta
  6. rebuild AVG/OBP/SLG or FIP from components + league-average BIP outcomes
  7. attach bootstrap CI from factor uncertainty ⊕ sampling uncertainty
```

### 10.11 Marcel-style use after MLE

Once you have MLB-equivalent seasonal components, feed them like MLB history:

\[
\text{proj} = \frac{5 r_{y-1} + 4 r_{y-2} + 3 r_{y-3} + r_{\text{prior}} n_0}{w + n_0}
\]

then age-adjust (Tango Marcel). Chris Mitchell / BP-style prospect work historically layered translations under projection systems this way; PECOTA used DTs + comps instead of pure Marcel weights.

---

## 11. Public datasets & open source

| Resource | Use |
|----------|-----|
| [Chadwick Bureau Register](https://github.com/chadwickbureau/register) | Person IDs linking MLB/MiLB/international |
| [Chadwick Bureau](https://www.chadwick-bureau.com/) | Reproducible historical builds |
| Baseball-Reference / Sports Reference register | Minor + foreign seasonal lines |
| [FanGraphs](https://www.fangraphs.com/) + [Sabermetrics Library – League Equivalencies](https://library.fangraphs.com/principles/league-equivalencies/) | Concepts; leaderboards |
| [Baseball Savant Minors](https://baseballsavant.mlb.com/statcast-search-minors) | MiLB Statcast when available |
| [Ogilvie MLE Factors](https://ogilviebaseball.com/mle-factors) | Modern published MiLB component factors |
| [claydavenport.com](http://claydavenport.com/) | Ongoing translation commentary + studies |
| [kbo-to-mlb-projection (GitHub)](https://github.com/genehuh39/kbo-to-mlb-projection) | Example factor JSON + age/role hooks |
| Seamheads MLE discussions / spreadsheets | Odds-ratio batting templates |
| Retrosheet / MLB stats API | MLB side of pairs |
| Japan Baseball Lab guides | NPB park + translation heuristics |
| `baseballr` / pybaseball-class scrapers | Ingestion (respect ToS) |

**Portfolio data build:** Chadwick IDs → pair seasons with ≥40–80 PA both sides → park factors → factor tables → validation on held-out promotions.

---

## 12. Validation checklist (hire bar)

1. **Backtest:** translated year-Y minor rates vs actual MLB year Y or Y+1 residuals by component.  
2. **Calibration:** 90% CI coverage near 90% on held-out transfers.  
3. **Bias audit:** residual vs age, org, handedness, GB%.  
4. **Stability:** factors re-estimated on rolling windows (Davenport-style era splits).  
5. **Null model:** beat “AAA = 0.82 offense” single-factor James baseline on MAE of BB%, K%, ISO.  
6. **Narrative demo:** one NPB pitcher, one KBO hitter, one AA prospect—with chains and CIs.

---

## 13. Assumptions & caveats

- Exact proprietary PECOTA/DT factor tables are not public; methods above reconstruct the **open literature**.  
- Numeric factors from Ogilvie, Japan Baseball Lab, and KBO GitHub tools are **illustrative starting points**—recalibrate on your pairs file.  
- FanGraphs library page is thin in scrape form; MacAree/StatCorner treatment is the conceptual source for “league equivalencies” in that library.  
- Ball changes (MLB 2015–2023, NPB seam differences) break stationarity—prefer recent cohorts for current projections, longer cohorts for historical MLEs.  
- Defense and baserunning translations are intentionally out of scope for v1 offense/pitch-components focus.

---

## 14. Sources (URLs)

1. Clay Davenport — site & AAA transition study: http://claydavenport.com/ · https://claydavenport.com/2025/09/02/players-moving-between-aaa-and-majors/  
2. Dan Szymborski — How to Calculate MLEs: https://baseballthinkfactory.org/btf/scholars/czerny/articles/calculatingMLEs.htm  
3. Tom Tango — Issues with MLEs: https://www.tangotiger.net/hateMLEs.html  
4. FanGraphs Sabermetrics Library — League Equivalencies: https://library.fangraphs.com/principles/league-equivalencies/  
5. Seamheads — Major League Equivalencies overview: https://seamheads.com/blog/2008/01/19/major-league-equivalencies/  
6. Ogilvie — MLE Translation Factors (2018–2025 cohorts): https://ogilviebaseball.com/mle-factors  
7. Japan Baseball Lab — NPB-to-MLB Translation Cheat Sheet: https://japanbaseballlab.com/npb-to-mlb-cheat-sheet-fantasy/  
8. PECOTA / DTs context (Wikipedia summary + citations to Davenport BP articles): https://en.wikipedia.org/wiki/PECOTA  
9. Down on the Farm — Diving Into Major League Equivalencies: https://downonthefarm.substack.com/p/diving-into-major-league-equivalencies  
10. KBO→MLB open projection tool: https://github.com/genehuh39/kbo-to-mlb-projection  
11. Chadwick Baseball Bureau Register: https://github.com/chadwickbureau/register  
12. Baseball Savant — Minors Statcast search: https://baseballsavant.mlb.com/statcast-search-minors  
13. Marcel projection description (Baseball-Reference): https://www.baseball-reference.com/about/marcels.shtml  
14. Tangotiger talent distribution notes: https://tangotiger.net/talent.html  

---

## 15. Bottom line

**MLE v1 worth hiring for:** component-wise, park-then-talent translations; age-handled once; SP/RP aware; **shrunken** factors; **bootstrap** uncertainty; **chained** paths for CPBL-class leagues; validation notebook on real promotion pairs. That is option **(A)**, aligned with James → Szymborski → Davenport → Tango critique → modern Ogilvie/NPB practice, and upgradeable to hierarchical Bayes when the data platform is ready.
