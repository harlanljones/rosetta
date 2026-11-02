"""League difficulty factors from the transferred-player cohort.

Method (portfolio v1): age-adjusted ratio-of-means link factors (sum of
weighted post-move rates over sum of weighted age-adjusted pre-move rates)
+ sample-size shrinkage to 1.0 + bootstrap percentile intervals. Factors
chain multiplicatively along CUBA → CPBL → KBO → NPB → AAA → MLB.

International links (an NPB/KBO/CPBL/CUBA endpoint) are different: both
directions are pooled into one sample (reverse rows inverted), the factor
shrinks in log space toward the chained product through AAA when that chain is
fitted (else 1.0), and an export/import direction term is reported. AAA↔MLB is
fitted per direction exactly as before.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
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


#: Leagues whose links are pooled across directions (D1). Links with neither
#: endpoint here (AAA<->MLB and anything else) are fitted per direction,
#: exactly as before pooling existed.
INTERNATIONAL_LEAGUES = frozenset({"NPB", "KBO", "CPBL", "CUBA"})

#: z for a central 80% interval (matches the 10th/90th bootstrap percentiles).
Z80 = 1.2815515655446004


@lru_cache(maxsize=8)
def _supports_skip(fn: object) -> bool:
    """True when ``fn`` accepts a ``skip`` keyword (``age_adjust_rate`` gains it in WP1)."""
    try:
        return "skip" in inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _is_imputed(pre_source: object, post_source: object) -> bool:
    return pre_source == "imputed" or post_source == "imputed"


def _age_adjust(
    pre: float,
    stat: str,
    age_from: float,
    age_to: float,
    role: str,
    *,
    imputed: bool = False,
) -> float:
    """Age-adjust ``pre``; imputed ages skip the adjustment when aging supports ``skip``.

    ``age_adjust_rate`` is looked up at call time. If it has no ``skip``
    keyword (older aging.py) the call falls back to the plain signature.
    """
    if imputed and _supports_skip(age_adjust_rate):
        return age_adjust_rate(pre, stat, age_from, age_to, role=role, skip=True)  # type: ignore[call-arg]
    return age_adjust_rate(pre, stat, age_from, age_to, role=role)


def _row_imputed(row: pd.Series) -> bool:
    return _is_imputed(row.get("pre_age_source"), row.get("post_age_source"))


def _weight(role: str, row: pd.Series) -> float:
    if role == "batter":
        return float(min(row["pre_pa"], row["post_pa"]))
    return float(min(row["pre_ip"], row["post_ip"]))


_ALLOWED_ESTIMATORS = ("mean_ratio", "ratio_of_means", "odds_ratio")

#: Stats that are proportions in [0, 1] — the only domain where the
#: odds-ratio estimator is meaningful (odds = p / (1 - p)).
ODDS_RATIO_STATS = frozenset({"bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba", "hr_fb"})

#: Odds clipping epsilon: rates at 0 or 1 would give infinite odds.
ODDS_EPS = 0.01


def _validate_estimator(estimator: str) -> None:
    if estimator not in _ALLOWED_ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {_ALLOWED_ESTIMATORS}")


def _to_odds(p: float) -> float:
    p = min(max(p, ODDS_EPS), 1.0 - ODDS_EPS)
    return p / (1.0 - p)


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

    ``estimator="odds_ratio"`` is the weighted mean of per-pair *odds*
    ratios, ``odds(post) / odds(age-adjusted pre)`` — the odds-space analog
    of the classic odds-ratio league translation. It is only defined for
    proportion stats (see ``ODDS_RATIO_STATS``); era/fip raise ValueError.
    Rates are clipped to ``ODDS_EPS`` from 0/1 before the odds transform.
    """
    _validate_estimator(estimator)
    if estimator == "odds_ratio" and stat not in ODDS_RATIO_STATS:
        raise ValueError(
            f"odds_ratio estimator requires a proportion stat (one of {sorted(ODDS_RATIO_STATS)}); got {stat!r}"
        )
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
        pre_adj = _age_adjust(
            pre,
            stat,
            float(row["age_from"]),
            float(row["age_to"]),
            role,
            imputed=_row_imputed(row),
        )
        if pre_adj <= 1e-9:
            continue
        if estimator == "odds_ratio":
            ratios.append(_to_odds(post) / _to_odds(pre_adj))
        else:
            ratios.append(post / pre_adj)
        weights.append(_weight(role, row))
        posts.append(post)
        pre_adjs.append(pre_adj)

    if not ratios:
        return {"factor": 1.0, "raw_factor": 1.0, "n": 0.0, "weight": 0.0, "shrink": 1.0}

    w = np.asarray(weights, dtype=float)
    if estimator in ("mean_ratio", "odds_ratio"):
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
        pre_adj = _age_adjust(
            float(row[pre_col]),
            stat,
            float(row["age_from"]),
            float(row["age_to"]),
            role,
            imputed=_row_imputed(row),
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


# --------------------------------------------------------------------------
# International pooling (D1/D2)
# --------------------------------------------------------------------------


def is_international_pair(a: str, b: str, chain: tuple[str, ...] = LEAGUE_CHAIN) -> bool:
    """True when the link is pooled: an NPB/KBO/CPBL/CUBA endpoint, both in ``chain``."""
    return (
        a != b
        and a in chain
        and b in chain
        and (a in INTERNATIONAL_LEAGUES or b in INTERNATIONAL_LEAGUES)
    )


def invert_transfer_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Re-express B->A transfer rows in the A->B orientation.

    Every ``pre_X`` / ``post_X`` column pair is swapped (stats, ``pa``, ``ip``,
    ``age_source``), as are ``age_from``/``age_to``, ``from_*``/``to_*``
    league and season, and the ``direction`` labels. Weights are
    ``min(pre, post)`` so they are unchanged by construction.
    """
    out = rows.copy()
    for col in list(rows.columns):
        if col.startswith("pre_"):
            other = "post_" + col[4:]
            if other in rows.columns:
                out[col] = rows[other].to_numpy()
                out[other] = rows[col].to_numpy()
    for left, right in (
        ("age_from", "age_to"),
        ("from_league", "to_league"),
        ("from_season", "to_season"),
    ):
        if left in rows.columns and right in rows.columns:
            out[left] = rows[right].to_numpy()
            out[right] = rows[left].to_numpy()
    if "direction" in rows.columns:
        out["direction"] = (
            rows["direction"]
            .map({"export": "import", "import": "export"})
            .fillna(rows["direction"])
        )
    return out


def pool_pair_sample(cohort: pd.DataFrame, a: str, b: str, role: str) -> pd.DataFrame:
    """One A->B sample for the unordered pair {a, b}: forward rows plus inverted reverse rows.

    Holdout rows are dropped first (training only). The ``_forward`` column
    marks rows that were observed in the A->B direction.
    """
    base = cohort["role"] == role
    if "holdout" in cohort.columns:
        base &= ~cohort["holdout"].astype(bool)
    fwd = cohort.loc[base & (cohort["from_league"] == a) & (cohort["to_league"] == b)].copy()
    rev = cohort.loc[base & (cohort["from_league"] == b) & (cohort["to_league"] == a)]
    fwd["_forward"] = True
    rev = invert_transfer_rows(rev)
    rev["_forward"] = False
    return pd.concat([fwd, rev], ignore_index=True)


def _stat_terms(sample: pd.DataFrame, stat: str, role: str) -> dict[str, np.ndarray]:
    """Per-row (post, age-adjusted pre, weight, forward flag) for the usable rows of ``sample``.

    Rows whose age-adjusted pre rate is <= 1e-9 are dropped, as in
    `estimate_link_factor`. Also returns ``all_post``/``all_pre_adj`` over every
    row for residuals.
    """
    pre_col, post_col = f"pre_{stat}", f"post_{stat}"
    empty = np.zeros(0)
    if sample.empty or pre_col not in sample.columns:
        return {
            "post": empty, "pre_adj": empty, "w": empty,
            "forward": np.zeros(0, dtype=bool), "all_post": empty, "all_pre_adj": empty,
        }  # fmt: skip
    posts, pre_adjs, weights, fwd = [], [], [], []
    for _, row in sample.iterrows():
        pre_adj = _age_adjust(
            float(row[pre_col]),
            stat,
            float(row["age_from"]),
            float(row["age_to"]),
            role,
            imputed=_row_imputed(row),
        )
        posts.append(float(row[post_col]))
        pre_adjs.append(pre_adj)
        weights.append(_weight(role, row))
        fwd.append(bool(row["_forward"]))
    post = np.asarray(posts)
    pre_adj = np.asarray(pre_adjs)
    keep = pre_adj > 1e-9
    return {
        "post": post[keep],
        "pre_adj": pre_adj[keep],
        "w": np.asarray(weights)[keep],
        "forward": np.asarray(fwd, dtype=bool)[keep],
        "all_post": post,
        "all_pre_adj": pre_adj,
    }


def _raw_from_terms(post: np.ndarray, pre_adj: np.ndarray, w: np.ndarray, estimator: str) -> float:
    """Unshrunk link factor from usable-row arrays (same arithmetic as `estimate_link_factor`)."""
    if estimator == "ratio_of_means":
        return float(np.sum(w * post) / np.sum(w * pre_adj))
    if estimator == "odds_ratio":
        ratios = np.array([_to_odds(p) / _to_odds(q) for p, q in zip(post, pre_adj, strict=True)])
    else:  # mean_ratio
        ratios = post / pre_adj
    return float(np.average(ratios, weights=w))


def _shrink_to_target(raw: float, target: float, shrink: float) -> float:
    """Shrink ``raw`` toward ``target`` in log space: exp(log t + s * (log raw - log t))."""
    log_t = math.log(max(target, 1e-9))
    log_r = math.log(max(raw, 1e-9))
    return float(math.exp(log_t + shrink * (log_r - log_t)))


def _thin_band(
    factor: float, resid_sd: float, post: np.ndarray, w: np.ndarray, n: float
) -> tuple[float, float]:
    """80% band for a link with < 5 pooled rows, from the residual sd alone.

    ``resid_sd`` is in rate units, so it is made relative to the weighted mean
    post-move rate, which approximates the log-scale sd of one mover's
    translation ratio. The standard error of a link built from ``n`` movers is
    that sd over sqrt(n) (n floored at 1), and the band is the point factor
    times exp(-/+ z80 * se). No credit is taken for shrinkage, so the band is
    conservative.
    """
    if n <= 0 or len(post) == 0:
        return factor, factor
    mean_post = float(np.average(post, weights=w)) if float(np.sum(w)) > 0 else float(post.mean())
    sd_rel = resid_sd / max(abs(mean_post), 1e-9)
    half = Z80 * sd_rel / math.sqrt(max(n, 1.0))
    return float(factor * math.exp(-half)), float(factor * math.exp(half))


def chain_target(
    links: dict[str, dict[str, dict[str, float]]],
    chain: tuple[str, ...],
    role: str,
    stat: str,
    a: str,
    b: str,
) -> float:
    """D2 shrink target: product of fitted links along the chain path a -> b, else 1.0.

    Applies only when the path has >= 3 nodes and every adjacent link on it is
    fitted with n > 0 (a link counts via its reciprocal when only the reverse
    node exists). Callers pass links that are already final (adjacent links
    shrink to 1.0), so this is never circular.
    """
    if a not in chain or b not in chain:
        return 1.0
    i, j = chain.index(a), chain.index(b)
    if j - i < 2:
        return 1.0
    role_links = links.get(role, {})
    target = 1.0
    for x, y in zip(chain[i:j], chain[i + 1 : j + 1], strict=True):
        node = role_links.get(f"{x}->{y}", {}).get(stat)
        if node and node.get("n", 0.0) > 0:
            target *= float(node["factor"])
            continue
        rev = role_links.get(f"{y}->{x}", {}).get(stat)
        if rev and rev.get("n", 0.0) > 0 and rev.get("factor"):
            target /= float(rev["factor"])
            continue
        return 1.0
    return float(target)


def fit_pooled_link(
    cohort: pd.DataFrame,
    *,
    a: str,
    b: str,
    stat: str,
    role: str,
    prior_n: float,
    target: float = 1.0,
    n_boot: int = 500,
    seed: int = 0,
    estimator: str = "ratio_of_means",
    sample: pd.DataFrame | None = None,
) -> tuple[dict, dict] | None:
    """Fit the pooled international link {a, b} for one stat; returns (a->b node, b->a node).

    ``a`` must precede ``b`` in the league chain. Forward (a->b) rows are used
    as is and reverse rows are inverted (`invert_transfer_rows`), then the
    estimator runs on the pooled sample. ``factor`` shrinks toward ``target``
    in log space with ``shrink = n / (n + prior_n)``.

    Node fields beyond the legacy ones: ``shrink_target``, ``thin``,
    ``factor_export`` / ``factor_import`` (unshrunk estimates on the
    forward-only / reverse-only subsets in the a->b orientation, None when that
    side has no usable rows), ``n_export`` / ``n_import`` and
    ``direction_gap = log(factor_export / factor_import)`` (None when either
    side is empty). Bootstrap resamples pooled rows; with fewer than 5 usable
    rows the band comes from `_thin_band` and ``thin`` is True.

    The b->a node is the reciprocal: ``factor``, ``raw_factor``,
    ``shrink_target``, ``factor_export``, ``factor_import`` invert, ``low`` and
    ``high`` invert and swap, ``resid_sd`` divides by the factor (rate units of
    the other league), ``direction_gap`` flips sign, counts are unchanged.
    Returns None when the pair has no usable rows.
    """
    _validate_estimator(estimator)
    if sample is None:
        sample = pool_pair_sample(cohort, a, b, role)
    terms = _stat_terms(sample, stat, role)
    post, pre_adj, w, fwd = terms["post"], terms["pre_adj"], terms["w"], terms["forward"]
    n = float(len(post))
    if n == 0:
        return None

    raw = _raw_from_terms(post, pre_adj, w, estimator)
    shrink = n / (n + prior_n)
    factor = _shrink_to_target(raw, target, shrink)

    n_exp, n_imp = int(fwd.sum()), int((~fwd).sum())

    def _side(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return _raw_from_terms(post[mask], pre_adj[mask], w[mask], estimator)

    f_exp, f_imp = _side(fwd), _side(~fwd)
    gap = math.log(f_exp / f_imp) if f_exp and f_imp and f_exp > 0 and f_imp > 0 else None

    all_post, all_pre = terms["all_post"], terms["all_pre_adj"]
    resid = (
        float(np.sqrt(np.mean(np.square(all_post - factor * all_pre)))) if len(all_post) else 0.0
    )

    thin = n < 5
    if thin:
        low, high = _thin_band(factor, resid, post, w, n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(post))
        samples = []
        for _ in range(n_boot):
            draw = rng.choice(idx, size=len(idx), replace=True)
            raw_b = _raw_from_terms(post[draw], pre_adj[draw], w[draw], estimator)
            samples.append(_shrink_to_target(raw_b, target, shrink))
        lo_q, hi_q = np.quantile(samples, [0.10, 0.90])
        low, high = float(lo_q), float(hi_q)

    forward_node = {
        "factor": float(factor),
        "raw_factor": float(raw),
        "n": n,
        "weight": float(w.sum()),
        "shrink": float(shrink),
        "shrink_target": float(target),
        "factor_export": f_exp,
        "factor_import": f_imp,
        "n_export": n_exp,
        "n_import": n_imp,
        "direction_gap": gap,
        "low": float(low),
        "high": float(high),
        "resid_sd": resid,
        "thin": thin,
    }

    def _inv(x: float | None) -> float | None:
        return None if x is None else 1.0 / x

    reverse_node = {
        **forward_node,
        "factor": 1.0 / forward_node["factor"],
        "raw_factor": 1.0 / forward_node["raw_factor"],
        "shrink_target": 1.0 / forward_node["shrink_target"],
        "factor_export": _inv(f_exp),
        "factor_import": _inv(f_imp),
        "direction_gap": None if gap is None else -gap,
        "low": 1.0 / forward_node["high"],
        "high": 1.0 / forward_node["low"],
        "resid_sd": resid / forward_node["factor"],
    }
    return forward_node, reverse_node


@dataclass
class LeagueFactorModel:
    """Fitted component-wise league factors with chain support."""

    links: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    # links[role][f"{a}->{b}"][stat] = factor stats
    chain: tuple[str, ...] = LEAGUE_CHAIN
    prior_n: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIOR_N))
    n_boot: int = 500
    seed: int = 42
    calibration: dict | None = None

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
        payload = {
            "links": self.links,
            "chain": list(self.chain),
            "prior_n": self.prior_n,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }
        if self.calibration:
            payload["calibration"] = self.calibration
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> LeagueFactorModel:
        return cls(
            links=data.get("links", {}),
            chain=tuple(data.get("chain", LEAGUE_CHAIN)),
            prior_n=data.get("prior_n", dict(DEFAULT_PRIOR_N)),
            n_boot=int(data.get("n_boot", 500)),
            seed=int(data.get("seed", 42)),
            calibration=data.get("calibration"),
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
    """Estimate all observed links from the cohort.

    Links with an NPB/KBO/CPBL/CUBA endpoint are pooled across directions and
    stored under both ``A->B`` and ``B->A`` (see `fit_pooled_link`); every other
    link is a directed fit, unchanged.

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
    legacy: list[tuple[str, str, str]] = []
    pooled: set[tuple[str, str, str]] = set()
    for from_lg, to_lg, role in pairs:
        if is_international_pair(from_lg, to_lg, model.chain):
            a, b = sorted((from_lg, to_lg), key=model.chain.index)
            pooled.add((a, b, role))
        else:
            legacy.append((from_lg, to_lg, role))

    def _effective(stat: str) -> str:
        # odds_ratio is only defined for proportion stats; era/fip fall back
        # to ratio_of_means so a whole-model odds_ratio fit stays usable.
        return (
            "ratio_of_means"
            if estimator == "odds_ratio" and stat not in ODDS_RATIO_STATS
            else estimator
        )

    def _stable_offset(*parts: str) -> int:
        # Python's hash() is intentionally salted per process.  A model
        # written with a seed based on it was therefore not reproducible
        # across workers (or even two CLI invocations).  Derive a stable
        # integer from the link identity instead.
        identity = "\x1f".join(parts).encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % 10000

    # Directed links with no international endpoint (AAA<->MLB, ...) are
    # fitted exactly as before pooling existed (D1).
    for from_lg, to_lg, role in legacy:
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        key = model.link_key(from_lg, to_lg)
        model.links.setdefault(role, {}).setdefault(key, {})
        for i, stat in enumerate(stats):
            stable_offset = _stable_offset(from_lg, to_lg, role, stat)
            est = bootstrap_link_factor(
                cohort,
                from_league=from_lg,
                to_league=to_lg,
                stat=stat,
                role=role,
                prior_n=priors.get(stat, 35.0),
                n_boot=n_boot,
                seed=seed + i * 17 + stable_offset,
                estimator=_effective(stat),
            )
            model.links[role][key][stat] = est

    # International links are pooled over both directions and stored as A->B
    # and B->A. Chain-adjacent pairs go first (their shrink target is 1.0), so
    # non-adjacent pairs can shrink toward the product of fitted adjacent links.
    def _is_adjacent(pair: tuple[str, str, str]) -> bool:
        return model.chain.index(pair[1]) - model.chain.index(pair[0]) == 1

    ordered = sorted(pooled, key=lambda p: (not _is_adjacent(p), p))
    for a, b, role in ordered:
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        sample = pool_pair_sample(cohort, a, b, role)
        for i, stat in enumerate(stats):
            target = chain_target(model.links, model.chain, role, stat, a, b)
            fitted = fit_pooled_link(
                cohort,
                a=a,
                b=b,
                stat=stat,
                role=role,
                prior_n=priors.get(stat, 35.0),
                target=target,
                n_boot=n_boot,
                seed=seed + i * 17 + _stable_offset(a, b, role, stat),
                estimator=_effective(stat),
                sample=sample,
            )
            if fitted is None:
                continue
            fwd_node, rev_node = fitted
            model.links.setdefault(role, {}).setdefault(model.link_key(a, b), {})[stat] = fwd_node
            model.links[role].setdefault(model.link_key(b, a), {})[stat] = rev_node
    return model
