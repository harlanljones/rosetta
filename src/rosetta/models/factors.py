"""League difficulty factors from the transferred-player cohort.

Method (portfolio v1): age-adjusted rate ratios + sample-size shrinkage to 1.0
+ bootstrap percentile intervals. Factors chain multiplicatively along
CUBA → CPBL → KBO → NPB → AAA → MLB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rosetta.models.aging import age_adjust_rate
from rosetta.schema import LEAGUE_CHAIN

HITTER_STATS = ("bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba")
PITCHER_STATS = ("k_pct", "bb_pct", "hr_fb", "era", "fip")

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


def estimate_link_factor(
    cohort: pd.DataFrame,
    *,
    from_league: str,
    to_league: str,
    stat: str,
    role: str,
    prior_n: float,
) -> dict[str, float]:
    """Weighted mean of age-adjusted post/pre ratios for one link+stat."""
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
    for _, row in sub.iterrows():
        pre = float(row[pre_col])
        post = float(row[post_col])
        pre_adj = age_adjust_rate(pre, stat, float(row["age_from"]), float(row["age_to"]), role=role)
        if pre_adj <= 1e-9:
            continue
        ratios.append(post / pre_adj)
        weights.append(_weight(role, row))

    if not ratios:
        return {"factor": 1.0, "raw_factor": 1.0, "n": 0.0, "weight": 0.0, "shrink": 1.0}

    w = np.asarray(weights, dtype=float)
    r = np.asarray(ratios, dtype=float)
    raw = float(np.average(r, weights=w))
    n_eff = float(len(r))
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
) -> dict[str, float]:
    """Bootstrap percentile interval for a shrunken link factor + residual sigma."""
    base = estimate_link_factor(
        cohort,
        from_league=from_league,
        to_league=to_league,
        stat=stat,
        role=role,
        prior_n=prior_n,
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
) -> LeagueFactorModel:
    """Estimate all observed directed links from the cohort."""
    priors = dict(DEFAULT_PRIOR_N if prior_n is None else prior_n)
    model = LeagueFactorModel(prior_n=priors, n_boot=n_boot, seed=seed)

    pairs = (
        cohort[["from_league", "to_league", "role"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for from_lg, to_lg, role in pairs:
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        key = model.link_key(from_lg, to_lg)
        model.links.setdefault(role, {}).setdefault(key, {})
        for i, stat in enumerate(stats):
            est = bootstrap_link_factor(
                cohort,
                from_league=from_lg,
                to_league=to_lg,
                stat=stat,
                role=role,
                prior_n=priors.get(stat, 35.0),
                n_boot=n_boot,
                seed=seed + i * 17 + hash((from_lg, to_lg, role)) % 10000,
            )
            model.links[role][key][stat] = est
    return model
