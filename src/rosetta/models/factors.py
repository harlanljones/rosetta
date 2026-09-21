"""League difficulty factors from the transferred-player cohort.

Method (portfolio v1): age-adjusted ratio-of-means link factors (sum of
weighted post-move rates over sum of weighted age-adjusted pre-move rates)
+ sample-size shrinkage to 1.0 + bootstrap percentile intervals. Factors
chain multiplicatively along CUBA → CPBL → KBO → NPB → AAA → MLB.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rosetta.models.aging import age_adjust_rate
from rosetta.schema import LEAGUE_CHAIN

HITTER_STATS = ("bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba")
PITCHER_STATS = ("k_pct", "bb_pct", "hr_fb", "era", "fip")

#: Domain-fixed era floor per league: seasons before this value are a
#: different run environment and must not be pooled with later seasons when
#: fitting transfer-cohort link factors. AAA adopted the MLB (livelier) ball
#: starting in the 2019 season; AAA-2018-and-earlier rates were produced
#: under a different, deader ball and are not comparable to 2019+ AAA or to
#: MLB rates from any era. This is a fixed domain fact, not a tuned
#: lookback window — do not adjust it to improve backtest metrics.
LEAGUE_ERA_FLOOR: dict[str, int] = {"AAA": 2019}


def apply_era_floor(
    cohort: pd.DataFrame, era_floor: dict[str, int] | None = LEAGUE_ERA_FLOOR
) -> pd.DataFrame:
    """Drop cohort rows whose AAA endpoint predates the AAA ball-standardization floor.

    A row is dropped when ``from_league`` is in ``era_floor`` and
    ``from_season`` precedes the floor, or when ``to_league`` is in
    ``era_floor`` and ``to_season`` precedes the floor. International links
    (KBO/NPB/CPBL/CUBA) are unaffected unless one endpoint is a floored
    league (currently only AAA).

    ``era_floor=None`` disables filtering entirely (returns ``cohort``
    unchanged). If the cohort lacks ``from_season``/``to_season`` columns,
    the floor cannot be evaluated and the cohort is returned unchanged —
    this keeps the helper safe to call on cohorts without season columns
    (e.g. small unit-test fixtures) rather than raising.
    """
    if not era_floor:
        return cohort
    if "from_season" not in cohort.columns or "to_season" not in cohort.columns:
        return cohort
    if cohort.empty:
        return cohort

    from_season = pd.to_numeric(cohort["from_season"], errors="coerce")
    to_season = pd.to_numeric(cohort["to_season"], errors="coerce")

    drop = pd.Series(False, index=cohort.index)
    for league, floor_season in era_floor.items():
        drop |= (cohort["from_league"] == league) & (from_season < floor_season)
        drop |= (cohort["to_league"] == league) & (to_season < floor_season)
    return cohort.loc[~drop].copy()

# Prior strength for shrinkage toward 1.0 (units of "pseudo-movers")
DEFAULT_PRIOR_N = {
    "bb_pct": 30.0,
    "k_pct": 25.0,
    "iso": 40.0,
    "babip": 50.0,
    "hr_pct": 45.0,
    "woba": 35.0,
    "hr_fb": 40.0,
    "era": 45.0,
    "fip": 35.0,
}


def _weight(role: str, row: pd.Series) -> float:
    if role == "batter":
        return float(min(row["pre_pa"], row["post_pa"]))
    return float(min(row["pre_ip"], row["post_ip"]))


_ALLOWED_ESTIMATORS = ("mean_ratio", "ratio_of_means")


def _validate_estimator(estimator: str) -> None:
    if estimator not in _ALLOWED_ESTIMATORS:
        raise ValueError(
            f"unknown estimator {estimator!r}; expected one of {_ALLOWED_ESTIMATORS}"
        )


def estimate_link_factor(
    cohort: pd.DataFrame,
    *,
    from_league: str,
    to_league: str,
    stat: str,
    role: str,
    prior_n: float,
    estimator: str = "ratio_of_means",
) -> dict[str, float]:
    """Estimate a shrunken link factor for one link+stat.

    ``estimator="ratio_of_means"`` (default) computes
    ``sum(weight * post) / sum(weight * pre_adj)`` over the cohort rows and
    weights. Selected on 2022-2024 rolling backtest targets (won 33/33
    target x stat MAE comparisons) and confirmed on the 2025 holdout
    (better on all 11 stats).

    ``estimator="mean_ratio"`` is the weighted mean of per-pair age-adjusted
    post/pre ratios; retained for comparison only. It is Jensen-biased
    upward and unstable (can blow up) when a player's pre-move rate is
    near zero.
    """
    _validate_estimator(estimator)
    mask = (
        (cohort["from_league"] == from_league)
        & (cohort["to_league"] == to_league)
        & (cohort["role"] == role)
        & (~cohort.get("holdout", False).astype(bool) if "holdout" in cohort.columns else True)
    )
    # Training only: exclude holdout seasons when flag present
    if "holdout" in cohort.columns:
        mask = (
            (cohort["from_league"] == from_league)
            & (cohort["to_league"] == to_league)
            & (cohort["role"] == role)
            & (~cohort["holdout"].astype(bool))
        )
    sub = cohort.loc[mask]
    pre_col, post_col = f"pre_{stat}", f"post_{stat}"
    if sub.empty or pre_col not in sub.columns:
        return {
            "factor": 1.0,
            "raw_factor": 1.0,
            "n": 0.0,
            "weight": 0.0,
            "shrink": 1.0,
        }

    ratios = []
    weights = []
    posts = []
    pre_adjs = []
    for _, row in sub.iterrows():
        pre = float(row[pre_col])
        post = float(row[post_col])
        pre_adj = age_adjust_rate(pre, stat, float(row["age_from"]), float(row["age_to"]), role=role)
        if pre_adj <= 1e-9:
            continue
        ratios.append(post / pre_adj)
        weights.append(_weight(role, row))
        posts.append(post)
        pre_adjs.append(pre_adj)

    if not ratios:
        return {"factor": 1.0, "raw_factor": 1.0, "n": 0.0, "weight": 0.0, "shrink": 1.0}

    w = np.asarray(weights, dtype=float)
    if estimator == "mean_ratio":
        r = np.asarray(ratios, dtype=float)
        raw = float(np.average(r, weights=w))
    else:  # ratio_of_means
        post_arr = np.asarray(posts, dtype=float)
        pre_adj_arr = np.asarray(pre_adjs, dtype=float)
        raw = float(np.sum(w * post_arr) / np.sum(w * pre_adj_arr))
    n_eff = float(len(ratios))
    shrink = n_eff / (n_eff + prior_n)
    factor = 1.0 + shrink * (raw - 1.0)
    return {
        "factor": float(factor),
        "raw_factor": float(raw),
        "n": n_eff,
        "weight": float(w.sum()),
        "shrink": float(shrink),
    }


def _link_residuals(
    cohort: pd.DataFrame,
    *,
    from_league: str,
    to_league: str,
    stat: str,
    role: str,
    factor: float,
) -> float:
    """RMSE of post − factor * age-adjusted pre; captures individual outcome noise."""
    mask = (
        (cohort["from_league"] == from_league)
        & (cohort["to_league"] == to_league)
        & (cohort["role"] == role)
    )
    if "holdout" in cohort.columns:
        mask &= ~cohort["holdout"].astype(bool)
    sub = cohort.loc[mask]
    pre_col, post_col = f"pre_{stat}", f"post_{stat}"
    if sub.empty or pre_col not in sub.columns:
        return 0.0
    errs = []
    for _, row in sub.iterrows():
        pre_adj = age_adjust_rate(
            float(row[pre_col]), stat, float(row["age_from"]), float(row["age_to"]), role=role
        )
        errs.append(float(row[post_col]) - factor * pre_adj)
    if not errs:
        return 0.0
    return float(np.sqrt(np.mean(np.square(errs))))


def bootstrap_link_factor(
    cohort: pd.DataFrame,
    *,
    from_league: str,
    to_league: str,
    stat: str,
    role: str,
    prior_n: float,
    n_boot: int = 500,
    seed: int = 0,
    estimator: str = "ratio_of_means",
) -> dict[str, float]:
    """Bootstrap percentile interval for a shrunken link factor + residual sigma.

    ``estimator`` defaults to "ratio_of_means" (see `estimate_link_factor`);
    "mean_ratio" is retained for comparison but is Jensen-biased upward and
    unstable for small pre-move rates.
    """
    _validate_estimator(estimator)
    base = estimate_link_factor(
        cohort,
        from_league=from_league,
        to_league=to_league,
        stat=stat,
        role=role,
        prior_n=prior_n,
        estimator=estimator,
    )
    resid = _link_residuals(
        cohort,
        from_league=from_league,
        to_league=to_league,
        stat=stat,
        role=role,
        factor=base["factor"],
    )
    mask = (
        (cohort["from_league"] == from_league)
        & (cohort["to_league"] == to_league)
        & (cohort["role"] == role)
    )
    if "holdout" in cohort.columns:
        mask &= ~cohort["holdout"].astype(bool)
    sub = cohort.loc[mask]
    if len(sub) < 5:
        return {**base, "low": base["factor"], "high": base["factor"], "resid_sd": resid}

    rng = np.random.default_rng(seed)
    samples = []
    idx = np.arange(len(sub))
    for _ in range(n_boot):
        draw = sub.iloc[rng.choice(idx, size=len(sub), replace=True)]
        est = estimate_link_factor(
            draw,
            from_league=from_league,
            to_league=to_league,
            stat=stat,
            role=role,
            prior_n=prior_n,
            estimator=estimator,
        )
        samples.append(est["factor"])
    low, high = np.quantile(samples, [0.10, 0.90])
    return {**base, "low": float(low), "high": float(high), "resid_sd": resid}


@dataclass
class LeagueFactorModel:
    """Fitted component-wise league factors with chain support."""

    links: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    # links[role][f"{a}->{b}"][stat] = factor stats
    chain: tuple[str, ...] = LEAGUE_CHAIN
    prior_n: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIOR_N))
    n_boot: int = 500
    seed: int = 42

    def link_key(self, a: str, b: str) -> str:
        return f"{a}->{b}"

    def get_factor(self, role: str, from_league: str, to_league: str, stat: str) -> float:
        path = self.path(from_league, to_league)
        acc = 1.0
        for a, b in zip(path[:-1], path[1:], strict=False):
            key = self.link_key(a, b)
            node = self.links.get(role, {}).get(key, {}).get(stat)
            if node is None:
                # Try reverse link inversion for limited cases
                rev = self.links.get(role, {}).get(self.link_key(b, a), {}).get(stat)
                if rev and rev.get("factor"):
                    acc *= 1.0 / rev["factor"]
                continue
            acc *= node["factor"]
        return acc

    #: Log-space SD penalty per path link with no fitted factor. An unestimated
    #: link contributes identity (1.0) to the point estimate but honest width to
    #: the band — thin-data translations must never report zero-width intervals.
    MISSING_LINK_SD = 0.20

    def get_interval(
        self, role: str, from_league: str, to_league: str, stat: str
    ) -> tuple[float, float, float, float]:
        """Return (point, low, high, resid_sd) combining link intervals + residual noise."""
        path = self.path(from_league, to_league)
        log_mu = 0.0
        log_lo = 0.0
        log_hi = 0.0
        resid_var = 0.0
        for a, b in zip(path[:-1], path[1:], strict=False):
            key = self.link_key(a, b)
            node = self.links.get(role, {}).get(key, {}).get(stat)
            if not node:
                resid_var += self.MISSING_LINK_SD**2
                continue
            f = max(node["factor"], 1e-6)
            lo = max(node.get("low", f), 1e-6)
            hi = max(node.get("high", f), 1e-6)
            log_mu += np.log(f)
            log_lo += np.log(lo)
            log_hi += np.log(hi)
            resid_var += float(node.get("resid_sd", 0.0)) ** 2
        return (
            float(np.exp(log_mu)),
            float(np.exp(log_lo)),
            float(np.exp(log_hi)),
            float(np.sqrt(resid_var)),
        )

    def path(self, from_league: str, to_league: str) -> list[str]:
        if from_league == to_league:
            return [from_league]
        # Direct link preferred when fitted
        for role_links in self.links.values():
            if self.link_key(from_league, to_league) in role_links:
                return [from_league, to_league]
        if from_league in self.chain and to_league in self.chain:
            i = self.chain.index(from_league)
            j = self.chain.index(to_league)
            if i < j:
                return list(self.chain[i : j + 1])
            if i > j:
                return list(reversed(self.chain[j : i + 1]))
        return [from_league, to_league]

    def to_dict(self) -> dict:
        return {
            "links": self.links,
            "chain": list(self.chain),
            "prior_n": self.prior_n,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LeagueFactorModel:
        return cls(
            links=data.get("links", {}),
            chain=tuple(data.get("chain", LEAGUE_CHAIN)),
            prior_n=data.get("prior_n", dict(DEFAULT_PRIOR_N)),
            n_boot=int(data.get("n_boot", 500)),
            seed=int(data.get("seed", 42)),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> LeagueFactorModel:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_factor_model(
    cohort: pd.DataFrame,
    *,
    n_boot: int = 400,
    seed: int = 42,
    prior_n: dict[str, float] | None = None,
    estimator: str = "ratio_of_means",
    era_floor: dict[str, int] | None = LEAGUE_ERA_FLOOR,
) -> LeagueFactorModel:
    """Estimate all observed directed links from the cohort.

    ``estimator`` defaults to "ratio_of_means" (see `estimate_link_factor`);
    "mean_ratio" is retained for comparison but is Jensen-biased upward and
    unstable for small pre-move rates.

    ``era_floor`` defaults to `LEAGUE_ERA_FLOOR` and drops cohort rows with a
    pre-2019 AAA endpoint before fitting (see `apply_era_floor`): AAA
    switched to the MLB ball in 2019, so earlier AAA seasons are a different
    run environment and would bias fitted factors. Pass ``None`` to disable.
    """
    _validate_estimator(estimator)
    cohort = apply_era_floor(cohort, era_floor)
    priors = dict(DEFAULT_PRIOR_N if prior_n is None else prior_n)
    model = LeagueFactorModel(prior_n=priors, n_boot=n_boot, seed=seed)

    pairs = (
        cohort[["from_league", "to_league", "role"]]
        .drop_duplicates()
        .sort_values(["from_league", "to_league", "role"])
        .itertuples(index=False, name=None)
    )
    for from_lg, to_lg, role in pairs:
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        key = model.link_key(from_lg, to_lg)
        model.links.setdefault(role, {}).setdefault(key, {})
        for i, stat in enumerate(stats):
            # Python's hash() is intentionally salted per process.  A model
            # written with a seed based on it was therefore not reproducible
            # across workers (or even two CLI invocations).  Derive a stable
            # integer from the link identity instead.
            identity = f"{from_lg}\x1f{to_lg}\x1f{role}\x1f{stat}".encode()
            stable_offset = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % 10000
            est = bootstrap_link_factor(
                cohort,
                from_league=from_lg,
                to_league=to_lg,
                stat=stat,
                role=role,
                prior_n=priors.get(stat, 35.0),
                n_boot=n_boot,
                seed=seed + i * 17 + stable_offset,
                estimator=estimator,
            )
            model.links[role][key][stat] = est
    return model
