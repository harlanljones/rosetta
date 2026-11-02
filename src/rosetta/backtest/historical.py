"""Leakage-safe historical evaluation of the real AAA-to-MLB link.

This module intentionally does not use the transfer index.  A transfer file is
useful for building the production cohort, but it is not an independent source
of truth for a retrospective evaluation: the evaluator constructs adjacent
season pairs directly from the normalized season snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rosetta.cohort.builder import pair_seasons
from rosetta.ingest.chadwick import build_id_crosswalk, load_register
from rosetta.ingest.loaders import load_all_seasons, load_transfers
from rosetta.models.factors import (
    HITTER_STATS,
    LEAGUE_ERA_FLOOR,
    PITCHER_STATS,
    apply_era_floor,
    fit_factor_model,
)
from rosetta.models.park import load_runs_park_factors, lookup_runs_park_factor
from rosetta.paths import SNAPSHOT_DIR
from rosetta.translate.api import translate_line

_ROLES = ("batter", "pitcher")
_LEAGUES = ("AAA", "MLB")
_CALIBRATION_METHODS = {"none", "prior_affine"}
_CALIBRATION_MIN_ROWS = 20


def _finite(value: Any) -> bool:
    """Return whether a scalar is a finite number (without coercing strings)."""
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _real_mask(series: pd.Series) -> pd.Series:
    """Interpret the normalized boolean column conservatively.

    Only an explicit false value is real.  This keeps missing or malformed
    provenance out of a real-data report rather than accidentally treating it
    as real through pandas truthiness rules.
    """
    return series.map(
        lambda value: (isinstance(value, (bool, np.bool_)) and not bool(value))
        or (isinstance(value, (int, np.integer)) and value == 0)
        or (isinstance(value, (float, np.floating)) and value == 0.0)
        or (isinstance(value, str) and value.strip().lower() in {"false", "0", "no"})
    ).astype(bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_parameters(
    target_season: int, n_boot: int, seed: int, min_pa: float, min_ip: float
) -> None:
    if isinstance(target_season, bool) or not isinstance(target_season, (int, np.integer)):
        raise ValueError("target_season must be an integer season")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _finite(min_pa) or float(min_pa) < 0:
        raise ValueError("min_pa must be a finite non-negative number")
    if not _finite(min_ip) or float(min_ip) < 0:
        raise ValueError("min_ip must be a finite non-negative number")


def _prepare(df: pd.DataFrame, *, league: str, target_season: int, role: str) -> pd.DataFrame:
    """Keep real lines through target season and normalize identity columns."""
    if df.empty:
        return df.copy()
    required = {"player_id", "season", "league", "role", "is_synthetic", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{league} {role} snapshot missing required columns: {missing}")

    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    seasons = pd.to_numeric(out["season"], errors="coerce")
    if seasons.isna().any() or (~np.isfinite(seasons.to_numpy(dtype=float))).any():
        raise ValueError(f"{league} {role} snapshot has non-finite season values")
    if (seasons % 1 != 0).any():
        raise ValueError(f"{league} {role} snapshot has non-integer season values")
    out["season"] = seasons.astype(int)
    # Future rows are deliberately excluded before any value validation.  A
    # malformed 2026 outcome must not change a 2025 fit or prevent it running.
    out = out.loc[(out["season"] <= target_season) & _real_mask(out["is_synthetic"])].copy()
    for field in ("player_id", "source"):
        blank = out[field].isna() | out[field].map(
            lambda value: str(value).strip().lower() in {"", "nan", "none"}
        ).astype(bool)
        if blank.any():
            raise ValueError(f"{league} {role} snapshot has blank {field} on real relevant lines")
    observed_leagues = out["league"].astype(str).str.strip().str.upper()
    if (~observed_leagues.eq(league)).any():
        bad = sorted(observed_leagues.loc[~observed_leagues.eq(league)].unique().tolist())
        raise ValueError(f"{league} snapshot contains wrong league identity: {bad}")
    out["league"] = league
    out["role"] = role

    key = ["player_id", "season", "league", "role"]
    duplicated = out.duplicated(key, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, key].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"ambiguous duplicate season keys in {league} {role} snapshots: {examples}")
    return out


def _pair_rows(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    from_league: str,
    to_league: str,
    target_season: int,
    min_pa: float,
    min_ip: float,
    role: str,
    id_crosswalk: dict[str, str] | None = None,
    include_team: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Build adjacent from_league(s-1)->to_league(s) pairs through target.

    ``include_team`` adds ``pre_team`` (the source line's team) to each row so
    callers can look up a team-season park factor; it is off for the AAA->MLB
    backtest, whose rows and outputs stay exactly as before.
    """
    if source.empty or target.empty:
        return [], 0
    stats = HITTER_STATS if role == "batter" else PITCHER_STATS
    sample = "pa" if role == "batter" else "ip"
    threshold = float(min_pa if role == "batter" else min_ip)
    rows: list[dict[str, Any]] = []
    candidates = 0

    def _index_by_uuid(df: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
        out: dict[tuple[str, int], pd.Series] = {}
        if not id_crosswalk:
            return out
        tmp = df.copy()
        tmp["key_uuid"] = tmp["player_id"].map(id_crosswalk)
        for _, row in tmp.dropna(subset=["key_uuid"]).iterrows():
            uid = str(row["key_uuid"])
            if not uid:
                continue
            out[(uid, int(row["season"]))] = row
        return out

    source_uuid_idx = _index_by_uuid(source)
    source_idx = source.set_index(["player_id", "season"])
    target_idx = target.set_index(["player_id", "season"])

    for (player_id, to_season), post in target_idx.iterrows():
        from_season = int(to_season) - 1
        if int(to_season) > target_season:
            continue
        key = (player_id, from_season)
        pre = source_idx.loc[key] if key in source_idx.index else None
        if pre is None and id_crosswalk:
            uid = id_crosswalk.get(str(player_id))
            if uid:
                pre = source_uuid_idx.get((uid, from_season))
        # Duplicate keys were rejected above, but retain this guard for clear
        # errors if a caller supplies a non-standard DataFrame directly.
        if isinstance(pre, pd.DataFrame) or isinstance(post, pd.DataFrame):
            raise ValueError(
                f"ambiguous {from_league}-to-{to_league} pair for {player_id} {from_season}->{to_season}"
            )
        if pre is None:
            continue
        candidates += 1
        pre_sample = pre.get(sample)
        post_sample = post.get(sample)
        if not _finite(pre_sample) or not _finite(post_sample):
            raise ValueError(
                f"non-finite {sample} for {player_id} {from_league} {from_season} or {to_league} {to_season}"
            )
        if float(pre_sample) < threshold or float(post_sample) < threshold:
            continue

        required = [
            ("age", pre.get("age"), from_league, from_season),
            ("age", post.get("age"), to_league, to_season),
            *[(stat, pre.get(stat), from_league, from_season) for stat in stats],
            *[(stat, post.get(stat), to_league, to_season) for stat in stats],
        ]
        bad = [f"{field} ({league} {season})" for field, value, league, season in required if not _finite(value)]
        if bad:
            raise ValueError(f"non-finite required values for {player_id}: {', '.join(bad)}")

        row: dict[str, Any] = {
            "player_id": str(player_id),
            "player_name": str(post.get("player_name", pre.get("player_name", player_id))),
            "role": role,
            "from_league": from_league,
            "to_league": to_league,
            "from_season": int(from_season),
            "to_season": int(to_season),
            "age_from": float(pre["age"]),
            "age_to": float(post["age"]),
            "pre_source": str(pre.get("source", "snapshot")),
            "post_source": str(post.get("source", "snapshot")),
            "pre_sample": float(pre_sample),
            "post_sample": float(post_sample),
        }
        if include_team:
            row["pre_team"] = str(pre.get("team", "") or "")
        for stat in stats:
            row[f"pre_{stat}"] = float(pre[stat])
            row[f"post_{stat}"] = float(post[stat])
        rows.append(row)
    rows.sort(key=lambda row: (row["to_season"], row["role"], row["player_id"]))
    return rows, candidates


def _training_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"pre_source", "post_source", "pre_sample", "post_sample"}}
         | {"pre_pa": row["pre_sample"] if row["role"] == "batter" else np.nan,
            "post_pa": row["post_sample"] if row["role"] == "batter" else np.nan,
            "pre_ip": row["pre_sample"] if row["role"] == "pitcher" else np.nan,
            "post_ip": row["post_sample"] if row["role"] == "pitcher" else np.nan}
         for row in rows]
    )


def _score_metrics(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not details:
        return []
    frame = pd.DataFrame(details)
    result: list[dict[str, Any]] = []
    for (role, stat), group in frame.groupby(["role", "stat"], sort=True):
        error = group["error"].to_numpy(dtype=float)
        baseline_error = group["prior"].to_numpy(dtype=float) - group["actual"].to_numpy(dtype=float)
        result.append(
            {
                "role": str(role),
                "stat": str(stat),
                "n": int(len(group)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "baseline_mae": float(np.mean(np.abs(baseline_error))),
                "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline_error)))),
                "coverage_80": float(group["covered_80"].mean()),
            }
        )
    return result


def _fit_prior_affine(details: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Fit out-of-sample affine corrections from completed earlier folds.

    ``details`` must contain predictions for target seasons strictly before the
    fold being calibrated.  Keeping this helper independent from the current
    target makes it difficult for a caller to accidentally tune on the holdout.
    A minimum row count and finite-value checks keep sparse fixture folds on the
    identity correction.  Residual spread is retained for widening calibrated
    prediction intervals without using the current target.
    """
    if not details:
        return {}
    frame = pd.DataFrame(details)
    corrections: dict[tuple[str, str], dict[str, Any]] = {}
    for (role, stat), group in frame.groupby(["role", "stat"], sort=True):
        prediction_column = "uncalibrated_prediction" if "uncalibrated_prediction" in group else "prediction"
        x = pd.to_numeric(group[prediction_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < _CALIBRATION_MIN_ROWS:
            continue
        design = np.column_stack((np.ones(len(x), dtype=float), x))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        if not all(_finite(value) for value in (intercept, slope)):
            continue
        residual = y - (intercept + slope * x)
        residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        residual_abs_q85 = float(np.quantile(np.abs(residual), 0.85)) if len(residual) else 0.0
        corrections[(str(role), str(stat))] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "resid_sd": residual_sd if _finite(residual_sd) else 0.0,
            "resid_abs_q85": residual_abs_q85 if _finite(residual_abs_q85) else 0.0,
            "n": int(len(x)),
        }
    return corrections


def _clip_calibrated(stat: str, value: float) -> float:
    """Keep calibrated rates in the same physical domain as source metrics."""
    if stat in {"bb_pct", "k_pct", "hr_pct", "babip", "iso", "woba", "hr_fb"}:
        return float(np.clip(value, 0.0, 1.0))
    if stat in {"era", "fip"}:
        return float(max(0.5, value))
    return float(value)


def _apply_prior_affine(
    payload: dict[str, Any], corrections: dict[tuple[str, str], dict[str, Any]]
) -> None:
    """Apply earlier-fold corrections and recompute detail metrics in place."""
    for row in payload["detail"]:
        key = (str(row["role"]), str(row["stat"]))
        correction = corrections.get(key)
        if correction is None:
            row["calibration"] = "identity"
            continue
        if "uncalibrated_prediction" not in row:
            row["uncalibrated_prediction"] = float(row["prediction"])
            row["uncalibrated_lower"] = float(row["lower"])
            row["uncalibrated_upper"] = float(row["upper"])
        intercept = float(correction["intercept"])
        slope = float(correction["slope"])
        center = intercept + slope * float(row["uncalibrated_prediction"])
        low = intercept + slope * float(row["uncalibrated_lower"])
        high = intercept + slope * float(row["uncalibrated_upper"])
        low, high = min(low, high), max(low, high)
        residual_half = float(correction.get("resid_abs_q85", 1.28 * float(correction["resid_sd"])))
        half = max((high - low) / 2.0, residual_half)
        center = _clip_calibrated(str(row["stat"]), center)
        low = _clip_calibrated(str(row["stat"]), center - half)
        high = _clip_calibrated(str(row["stat"]), center + half)
        row["prediction"] = center
        row["lower"] = low
        row["upper"] = high
        row["error"] = center - float(row["actual"])
        row["absolute_error"] = abs(row["error"])
        row["covered_80"] = bool(low <= float(row["actual"]) <= high)
        row["calibration"] = "prior_affine"
        row["calibration_n"] = int(correction["n"])
    payload["metrics"] = _score_metrics(payload["detail"])


def run_historical_backtest(
    snapshot_dir: Path | str | None = None,
    *,
    target_season: int = 2025,
    from_league: str = "AAA",
    to_league: str = "MLB",
    n_boot: int = 250,
    seed: int = 42,
    min_pa: float = 80.0,
    min_ip: float = 30.0,
    estimator: str = "ratio_of_means",
    era_floor: dict[str, int] | None = LEAGUE_ERA_FLOOR,
) -> dict[str, Any]:
    """Fit on pre-target real pairs and score the immediately following season.

    Only exact player IDs and adjacent AAA-to-MLB seasons are considered.  The
    function is pure with respect to snapshots and does not load or write a
    factors, transfer, leaderboard, or other output artifact.

    ``era_floor`` defaults to `LEAGUE_ERA_FLOOR` ({"AAA": 2019}): training
    pairs with an AAA endpoint season before the floor are excluded before
    fitting, because AAA adopted the MLB ball in 2019 and earlier AAA
    seasons are a different run environment. Evaluation pairs are never
    filtered by the floor (targets scored here are 2020+ anyway). Pass
    ``None`` to disable the floor for comparison runs.
    """
    _validate_parameters(target_season, n_boot, seed, min_pa, min_ip)
    target_season = int(target_season)
    n_boot = int(n_boot)
    seed = int(seed)
    min_pa = float(min_pa)
    min_ip = float(min_ip)
    root = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    source_season = target_season - 1
    from_league = str(from_league).strip().upper()
    to_league = str(to_league).strip().upper()
    link_key = f"{from_league}->{to_league}"

    crosswalk: dict[str, str] | None = None
    chadwick_path = root / "chadwick_people.csv"
    if chadwick_path.exists() and from_league != to_league:
        crosswalk = build_id_crosswalk(load_register(chadwick_path))

    by_role: dict[str, dict[str, pd.DataFrame]] = {}
    checksums: dict[str, str] = {}
    source_counts: dict[str, int] = {from_league: 0, to_league: 0}
    for role in _ROLES:
        by_role[role] = {}
        for league in (from_league, to_league):
            path = root / f"{league.lower()}_{role}s.csv"
            if path.exists():
                checksums[path.name] = _sha256(path)
            loaded = load_all_seasons(role=role, snapshot_dir=root, leagues=[league])
            prepared = _prepare(loaded, league=league, target_season=target_season, role=role)
            by_role[role][league] = prepared
            source_counts[league] += int(len(prepared))

    all_rows: list[dict[str, Any]] = []
    candidates = 0
    for role in _ROLES:
        rows, count = _pair_rows(
            by_role[role][from_league],
            by_role[role][to_league],
            from_league=from_league,
            to_league=to_league,
            target_season=target_season,
            min_pa=float(min_pa),
            min_ip=float(min_ip),
            role=role,
            id_crosswalk=crosswalk,
        )
        all_rows.extend(rows)
        candidates += count
    training_all = [row for row in all_rows if row["to_season"] < target_season]
    evaluation = [row for row in all_rows if row["from_season"] == source_season and row["to_season"] == target_season]
    if not training_all:
        raise ValueError(
            f"insufficient real {link_key} training pairs before {target_season}; "
            "need at least one adjacent pair with both playing-time thresholds met"
        )
    if not evaluation:
        raise ValueError(
            f"insufficient real {link_key} evaluation pairs for {source_season}->{target_season}; "
            "need at least one adjacent pair with both playing-time thresholds met"
        )

    train_frame_all = _training_frame(training_all)
    train_frame = (
        apply_era_floor(train_frame_all, era_floor)
        if from_league == "AAA" or to_league == "AAA"
        else train_frame_all
    )
    excluded_era_floor = len(training_all) - len(train_frame)
    if train_frame.empty:
        raise ValueError(
            f"insufficient real {link_key} training pairs before {target_season} after the AAA "
            f"era floor ({era_floor}) excluded {excluded_era_floor} pair(s); need at least one "
            "adjacent pair with both playing-time thresholds met and a post-floor AAA season"
        )
    kept_keys = set(
        zip(
            train_frame["player_id"], train_frame["from_season"], train_frame["to_season"],
            train_frame["role"], strict=False,
        )
    )
    training = [
        row for row in training_all
        if (row["player_id"], row["from_season"], row["to_season"], row["role"]) in kept_keys
    ]
    model = fit_factor_model(
        train_frame, n_boot=int(n_boot), seed=int(seed), estimator=estimator, era_floor=None,
    )
    # An identity fallback is useful for exploratory translations, but would
    # invalidate this evaluation.  Require every stat that will be scored to
    # have a real fitted count in the fresh model.
    for role in sorted({row["role"] for row in evaluation}):
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        node = model.links.get(role, {}).get(link_key, {})
        missing = [stat for stat in stats if float(node.get(stat, {}).get("n", 0.0)) <= 0]
        if missing:
            raise ValueError(
                f"unfitted real factor(s) for {role} {link_key}: {', '.join(missing)}; "
                "all scored statistics require positive training counts"
            )
    for role, role_links in model.links.items():
        for link_key, link in role_links.items():
            for stat, values in link.items():
                for field in ("factor", "raw_factor", "low", "high", "resid_sd"):
                    if field in values and not _finite(values[field]):
                        raise ValueError(
                            f"non-finite fitted factor for {role} {link_key} {stat} ({field})"
                        )

    details: list[dict[str, Any]] = []
    for row in evaluation:
        stats = HITTER_STATS if row["role"] == "batter" else PITCHER_STATS
        pre = {
            "player_id": row["player_id"], "player_name": row["player_name"],
            "season": row["from_season"], "league": from_league, "role": row["role"],
            "age": row["age_from"], "pa": row["pre_sample"] if row["role"] == "batter" else 0.0,
            "ip": row["pre_sample"] if row["role"] == "pitcher" else 0.0,
        }
        pre.update({stat: row[f"pre_{stat}"] for stat in stats})
        translated = translate_line(pre, model=model, from_league=from_league, to_league=to_league)
        for stat in stats:
            prediction = float(translated.rates[stat])
            actual = float(row[f"post_{stat}"])
            lower = float(translated.lower[stat])
            upper = float(translated.upper[stat])
            if not all(_finite(value) for value in (prediction, actual, lower, upper)):
                raise ValueError(
                    f"non-finite prediction/interval for {row['player_id']} {row['role']} {stat}"
                )
            details.append(
                {
                    "player_id": row["player_id"], "player_name": row["player_name"],
                    "role": row["role"], "from_season": row["from_season"],
                    "to_season": row["to_season"], "from_league": from_league, "to_league": to_league,
                    "pre_sample": row["pre_sample"], "post_sample": row["post_sample"],
                    "pre_source": row["pre_source"], "post_source": row["post_source"],
                    "stat": stat, "prior": row[f"pre_{stat}"], "prediction": prediction,
                    "actual": actual, "error": prediction - actual,
                    "absolute_error": abs(prediction - actual), "lower": lower, "upper": upper,
                    "covered_80": bool(lower <= actual <= upper),
                }
            )

    training_pairs = [
        {
            "player_id": row["player_id"], "player_name": row["player_name"], "role": row["role"],
            "from_season": row["from_season"], "to_season": row["to_season"],
            "from_league": from_league, "to_league": to_league, "pre_source": row["pre_source"],
            "post_source": row["post_source"], "pre_sample": row["pre_sample"],
            "post_sample": row["post_sample"], "cutoff_compliant": bool(row["to_season"] < target_season),
        }
        for row in training
    ]
    limitations = [
        "Evaluation is limited to players with observed MLB playing time above both thresholds; it is not a promotion or playing-time model.",
        "Normalized wOBA retains upstream Stats API event-weight approximations (2024-scale 0.69 UBB, 0.72 HBP, 0.89/1.27/1.62/2.10 hit weights).",
        "Normalized FIP uses the upstream fixed 3.10 constant; HR/FB is approximated as HR/(HR + air outs) when batted-ball splits are unavailable.",
        "Ages are normalized snapshot ages (or upstream inferred/default ages); no park or pitcher-role adjustment is applied and no target-season tuning is performed.",
        f"Only the real adjacent {link_key} link is evaluated; other league links are not validated in this run. A player may contribute to multiple pre-target training seasons and to the target holdout.",
        "The source rate is translated as observed; no additional target-age projection is applied.",
        "Snapshots are retrospective inputs rather than archived as-of feeds, and bootstrap resamples rows rather than player clusters.",
        "An AAA era floor excludes training pairs with an AAA endpoint season before "
        f"{era_floor}: AAA adopted the MLB (livelier) ball in 2019, so earlier AAA seasons are a "
        "different run environment (materially lower weighted AAA FIP/higher HR rates) and would "
        "bias fitted factors if pooled with post-2019 seasons. This is a domain-fixed floor, not a "
        "tuned lookback window; it can be disabled via era_floor=None for comparison." if era_floor
        else "The AAA era floor was explicitly disabled for this run (era_floor=None); training "
        "may include pre-2019 AAA seasons from before AAA's 2019 ball standardization.",
    ]
    metadata = {
        "data_mode": "real",
        "target_season": target_season,
        "source_season": source_season,
        "from_league": from_league,
        "to_league": to_league,
        "link": link_key,
        "training_cutoff": {"destination_season_lt": target_season},
        "counts": {
            "real_lines": source_counts, "candidate_adjacent_pairs": candidates,
            "training_pairs": len(training), "evaluation_pairs": len(evaluation),
            "detail_rows": len(details),
            "training_pairs_excluded_era_floor": int(excluded_era_floor),
        },
        "parameters": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "min_pa": float(min_pa),
            "min_ip": float(min_ip),
            "estimator": str(estimator),
            "era_floor": dict(era_floor) if era_floor else None,
        },
        "input_sha256": checksums, "source_counts": source_counts,
        "latest_training_to_season": max(row["to_season"] for row in training),
        "error_definition": "prediction minus actual",
        "limitations": limitations,
    }
    return {
        "metadata": metadata,
        "metrics": _score_metrics(details),
        "detail": details,
        "training_pairs": training_pairs,
        "model": model.to_dict(),
    }


def run_rolling_backtest(
    snapshot_dir: Path | str | None = None,
    *,
    target_seasons: list[int] | tuple[int, ...],
    n_boot: int = 250,
    seed: int = 42,
    min_pa: float = 80.0,
    min_ip: float = 30.0,
    estimator: str = "ratio_of_means",
    era_floor: dict[str, int] | None = LEAGUE_ERA_FLOOR,
    calibration: str = "prior_affine",
) -> dict[str, Any]:
    """Run `run_historical_backtest` independently per target season.

    Each target is fit and scored in isolation (its own to_season < target
    cutoff); no state or model is shared across targets. A target that raises
    `ValueError` for insufficient data is recorded in `metadata["skipped"]`
    rather than aborting the whole run. If every target is skipped this
    raises `ValueError`.

    ``era_floor`` defaults to `LEAGUE_ERA_FLOOR` and is passed through to
    every per-target `run_historical_backtest` call; pass ``None`` to
    disable the AAA ball-standardization floor for a comparison run.

    ``calibration="prior_affine"`` fits a per-stat linear correction from
    completed earlier target folds and applies it to the current fold.  The
    current target is never included in its correction, so this is a
    leakage-safe calibration step.  Pass ``calibration="none"`` for the raw
    factor model comparison.
    """
    seasons = [int(season) for season in target_seasons]
    if not seasons:
        raise ValueError("target_seasons must be non-empty")
    calibration = str(calibration).strip().lower()
    if calibration not in _CALIBRATION_METHODS:
        raise ValueError(
            f"unknown calibration {calibration!r}; expected one of {sorted(_CALIBRATION_METHODS)}"
        )

    raw_by_target: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []

    for season in seasons:
        try:
            payload = run_historical_backtest(
                snapshot_dir,
                target_season=season,
                n_boot=n_boot,
                seed=seed,
                min_pa=min_pa,
                min_ip=min_ip,
                estimator=estimator,
                era_floor=era_floor,
            )
        except ValueError as exc:
            skipped.append({"target_season": season, "reason": str(exc)})
            continue
        raw_by_target[str(season)] = payload

    if not raw_by_target:
        raise ValueError(
            "all requested target seasons were skipped for insufficient real data: "
            f"{skipped}"
        )

    by_target: dict[str, Any] = {}
    all_detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    calibration_fits: dict[str, dict[str, dict[str, Any]]] = {}
    if calibration == "prior_affine":
        # Preserve the raw out-of-fold predictions before any target payload is
        # calibrated.  Later folds must fit from the same raw factor output,
        # rather than recursively fitting on an already corrected fold.
        for payload in raw_by_target.values():
            for row in payload["detail"]:
                row["uncalibrated_prediction"] = float(row["prediction"])
                row["uncalibrated_lower"] = float(row["lower"])
                row["uncalibrated_upper"] = float(row["upper"])
    for season in sorted(int(key) for key in raw_by_target):
        payload = raw_by_target[str(season)]
        prior_details = [
            {**row, "target_season": prior_season}
            for prior_season, prior_payload in raw_by_target.items()
            if int(prior_season) < season
            for row in prior_payload["detail"]
        ]
        corrections = _fit_prior_affine(prior_details) if calibration == "prior_affine" else {}
        if calibration == "prior_affine":
            _apply_prior_affine(payload, corrections)
            calibration_fits[str(season)] = {
                f"{role}:{stat}": values for (role, stat), values in corrections.items()
            }
        payload["metadata"]["parameters"]["calibration"] = calibration
        payload["metadata"]["calibration"] = {
            "method": calibration,
            "fit_target_seasons": sorted({int(row["target_season"]) for row in prior_details}),
            "fits": calibration_fits.get(str(season), {}),
        }
        by_target[str(season)] = payload
        for row in payload["detail"]:
            all_detail.append({"target_season": season, **row})
        for row in payload["metrics"]:
            summary.append({"target_season": season, **row})

    excluded_era_floor_total = sum(
        int(payload["metadata"]["counts"].get("training_pairs_excluded_era_floor", 0))
        for payload in by_target.values()
    )

    pooled = _score_metrics(all_detail)

    metadata = {
        "data_mode": "real",
        "target_seasons": seasons,
        "parameters": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "min_pa": float(min_pa),
            "min_ip": float(min_ip),
            "estimator": str(estimator),
            "era_floor": dict(era_floor) if era_floor else None,
            "calibration": calibration,
        },
        "counts": {
            "training_pairs_excluded_era_floor": excluded_era_floor_total,
        },
        "input_sha256": next(iter(by_target.values()))["metadata"]["input_sha256"] if by_target else {},
        "skipped": skipped,
    }
    return {
        "metadata": metadata,
        "by_target": by_target,
        "summary": summary,
        "pooled": pooled,
    }


# --------------------------------------------------------------------------
# International (NPB / KBO -> MLB) rolling validation
# --------------------------------------------------------------------------

INTERNATIONAL_EVAL_LEAGUES = ("NPB", "KBO")
#: Headline stats the ship gate is judged on: (role, stat).
GATE_STATS: tuple[tuple[str, str], ...] = (("batter", "woba"), ("pitcher", "fip"))
GATE_MIN_N = 10
GATE_COVERAGE_RANGE = (0.70, 0.90)
_LINK_STATS = {"batter": "woba", "pitcher": "fip"}
_ROUND_LINK_FIELDS = ("factor", "low", "high", "raw_factor", "direction_gap")


_CROSSWALK_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}


def _cached_crosswalk(root: Path) -> dict[str, str] | None:
    """Chadwick id crosswalk for ``root`` (None without a register), memoized per file state.

    Building it walks the whole register row by row (minutes on the full file),
    so the international run builds it once and shares it between cohort
    assembly and evaluation pairing.
    """
    path = root / "chadwick_people.csv"
    if not path.exists():
        return None
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key not in _CROSSWALK_CACHE:
        _CROSSWALK_CACHE.clear()
        _CROSSWALK_CACHE[key] = build_id_crosswalk(
            load_register(path), kbo_foreign_map_path=root / "kbo_foreign_map.csv"
        )
    return _CROSSWALK_CACHE[key]


def _transfer_cohort(
    root: Path, crosswalk: dict[str, str] | None, park_factors: pd.DataFrame,
    *, min_pa: float, min_ip: float,
) -> pd.DataFrame:
    """The `build_cohort` cohort (same pairing, park handling and thresholds) with a shared crosswalk."""
    paired = pair_seasons(
        load_transfers(root),
        load_all_seasons(role="batter", snapshot_dir=root),
        load_all_seasons(role="pitcher", snapshot_dir=root),
        id_crosswalk=crosswalk,
        park_factors=park_factors,
    )
    if paired.empty:
        return paired
    keep = pd.Series(False, index=paired.index)
    batter = paired["role"] == "batter"
    keep |= batter & (paired["pre_pa"] >= min_pa) & (paired["post_pa"] >= min_pa)
    if "pre_ip" in paired.columns and "post_ip" in paired.columns:
        keep |= (~batter) & (paired["pre_ip"] >= min_ip) & (paired["post_ip"] >= min_ip)
    return paired.loc[keep].reset_index(drop=True)


def international_ship_gate(pooled: list[dict[str, Any]]) -> dict[str, Any]:
    """Ship gate on the pooled international result.

    PASS: for batter wOBA and pitcher FIP, model MAE < baseline MAE and 80%
    coverage inside ``GATE_COVERAGE_RANGE``. FAIL lists the failing rows.
    INSUFFICIENT when any gated stat has fewer than ``GATE_MIN_N`` scored rows
    (or none): too few exports to judge, which takes precedence over FAIL.
    """
    lo, hi = GATE_COVERAGE_RANGE
    rows: list[dict[str, Any]] = []
    for role, stat in GATE_STATS:
        match = next((r for r in pooled if r["role"] == role and r["stat"] == stat), None)
        n = int(match["n"]) if match else 0
        row: dict[str, Any] = {"role": role, "stat": stat, "n": n}
        if match is None or n < GATE_MIN_N:
            row.update(status="INSUFFICIENT", reasons=[f"n={n} < {GATE_MIN_N}"])
            rows.append(row)
            continue
        mae, base, cov = float(match["mae"]), float(match["baseline_mae"]), float(match["coverage_80"])
        reasons: list[str] = []
        if not mae < base:
            reasons.append(f"MAE {mae:.4f} does not beat baseline {base:.4f}")
        if not lo <= cov <= hi:
            reasons.append(f"80% coverage {cov:.3f} outside [{lo:.2f}, {hi:.2f}]")
        row.update(
            mae=mae, baseline_mae=base, coverage_80=cov,
            status="FAIL" if reasons else "PASS", reasons=reasons,
        )
        rows.append(row)
    statuses = {r["status"] for r in rows}
    verdict = "INSUFFICIENT" if "INSUFFICIENT" in statuses else "FAIL" if "FAIL" in statuses else "PASS"
    return {
        "verdict": verdict,
        "criteria": {"min_n": GATE_MIN_N, "coverage_range": [lo, hi], "stats": [list(x) for x in GATE_STATS]},
        "rows": rows,
        "failing": [r for r in rows if r["status"] != "PASS"],
    }


def _link_report(
    model: Any, *, league: str, target_season: int
) -> list[dict[str, Any]]:
    """Pooled factor, counts and direction gap for league->MLB and league->AAA."""
    out: list[dict[str, Any]] = []
    for dest in ("MLB", "AAA"):
        key = f"{league}->{dest}"
        for role, stat in _LINK_STATS.items():
            node = model.links.get(role, {}).get(key, {}).get(stat) or {}
            row: dict[str, Any] = {
                "target_season": target_season, "link": key, "role": role, "stat": stat,
                "fitted": bool(node), "n": int(node.get("n", 0) or 0),
                "n_export": int(node.get("n_export", 0) or 0),
                "n_import": int(node.get("n_import", 0) or 0),
                "thin": bool(node.get("thin", False)),
            }
            for field in _ROUND_LINK_FIELDS:
                value = node.get(field)
                row[field] = float(value) if value is not None and _finite(value) else None
            out.append(row)
    return out


def _path_fitted(model: Any, role: str, league: str, stats: tuple[str, ...]) -> str | None:
    """Reason string when a scored stat has an unfitted link on the league->MLB path."""
    path = model.path(league, "MLB")
    for a, b in zip(path[:-1], path[1:], strict=False):
        node = model.links.get(role, {}).get(model.link_key(a, b), {})
        missing = [s for s in stats if float(node.get(s, {}).get("n", 0.0) or 0.0) <= 0]
        if missing:
            return f"{role}: no fitted {a}->{b} link for {', '.join(missing)}"
    return None


def run_international_rolling(
    snapshot_dir: Path | str | None = None,
    *,
    league: str,
    target_seasons: list[int] | tuple[int, ...],
    n_boot: int = 250,
    seed: int = 42,
    min_pa: float = 80.0,
    min_ip: float = 30.0,
    estimator: str = "ratio_of_means",
) -> dict[str, Any]:
    """Rolling validation of NPB/KBO -> MLB translation on exports.

    For each target season T the factor model is refit on the full transferred
    cohort (both directions, every league, built by `build_cohort`) restricted
    to ``to_season < T``; international links pool across directions inside
    `fit_factor_model`.  Evaluation rows are exports: players with a real
    ``league`` line in T-1 and a real MLB line in T, both above the playing-time
    thresholds.  The MLB-T line is scored against ``translate_line`` of the T-1
    line (with runs park factors when game logs exist); the baseline carries the
    foreign rate forward untouched.  A target with no usable exports is recorded
    in ``metadata["skipped"]`` rather than raised.
    """
    league = str(league).strip().upper()
    if league not in INTERNATIONAL_EVAL_LEAGUES:
        raise ValueError(f"league must be one of {INTERNATIONAL_EVAL_LEAGUES}, got {league!r}")
    seasons = sorted({int(season) for season in target_seasons})
    if not seasons:
        raise ValueError("target_seasons must be non-empty")
    _validate_parameters(seasons[-1], n_boot, seed, min_pa, min_ip)
    n_boot, seed = int(n_boot), int(seed)
    min_pa, min_ip = float(min_pa), float(min_ip)
    root = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    last = seasons[-1]

    crosswalk = _cached_crosswalk(root)
    park_factors = load_runs_park_factors(root)
    cohort = _transfer_cohort(root, crosswalk, park_factors, min_pa=min_pa, min_ip=min_ip)

    checksums: dict[str, str] = {}
    by_role: dict[str, dict[str, pd.DataFrame]] = {}
    source_counts: dict[str, int] = {league: 0, "MLB": 0}
    for role in _ROLES:
        by_role[role] = {}
        for lg in (league, "MLB"):
            path = root / f"{lg.lower()}_{role}s.csv"
            if path.exists():
                checksums[path.name] = _sha256(path)
            loaded = load_all_seasons(role=role, snapshot_dir=root, leagues=[lg])
            prepared = _prepare(loaded, league=lg, target_season=last, role=role)
            by_role[role][lg] = prepared
            source_counts[lg] += int(len(prepared))

    by_target: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    all_detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for season in seasons:
        if cohort.empty:
            skipped.append({"target_season": season, "role": None, "reason": "no transferred-player cohort"})
            continue
        train = cohort.loc[cohort["to_season"] < season].copy()
        train["holdout"] = False
        if train.empty:
            skipped.append(
                {"target_season": season, "role": None, "reason": f"no training pairs with to_season < {season}"}
            )
            continue

        eval_rows: list[dict[str, Any]] = []
        candidates = 0
        for role in _ROLES:
            source = by_role[role][league]
            target = by_role[role]["MLB"]
            rows, count = _pair_rows(
                source.loc[source["season"] == season - 1],
                target.loc[target["season"] == season],
                from_league=league, to_league="MLB", target_season=season,
                min_pa=min_pa, min_ip=min_ip, role=role, id_crosswalk=crosswalk,
                include_team=True,
            )
            eval_rows.extend(rows)
            candidates += count
        if not eval_rows:
            skipped.append(
                {
                    "target_season": season, "role": None,
                    "reason": f"no {league} {season - 1} -> MLB {season} exports above the playing-time thresholds",
                }
            )
            continue

        model = fit_factor_model(train, n_boot=n_boot, seed=seed, estimator=estimator)
        links = _link_report(model, league=league, target_season=season)

        details: list[dict[str, Any]] = []
        park_applied = 0
        scored_roles: set[str] = set()
        for role in _ROLES:
            role_rows = [row for row in eval_rows if row["role"] == role]
            if not role_rows:
                continue
            stats = HITTER_STATS if role == "batter" else PITCHER_STATS
            reason = _path_fitted(model, role, league, stats)
            if reason is not None:
                skipped.append({"target_season": season, "role": role, "reason": reason})
                continue
            scored_roles.add(role)
            for row in role_rows:
                pre = {
                    "player_id": row["player_id"], "player_name": row["player_name"],
                    "season": row["from_season"], "league": league, "role": role,
                    "team": row["pre_team"], "age": row["age_from"],
                    "pa": row["pre_sample"] if role == "batter" else 0.0,
                    "ip": row["pre_sample"] if role == "pitcher" else 0.0,
                }
                pre.update({stat: row[f"pre_{stat}"] for stat in stats})
                if lookup_runs_park_factor(park_factors, league, row["pre_team"], row["from_season"]) is not None:
                    park_applied += 1
                translated = translate_line(
                    pre, model=model, from_league=league, to_league="MLB", park_factors=park_factors
                )
                for stat in stats:
                    prediction = float(translated.rates[stat])
                    actual = float(row[f"post_{stat}"])
                    lower, upper = float(translated.lower[stat]), float(translated.upper[stat])
                    if not all(_finite(value) for value in (prediction, actual, lower, upper)):
                        raise ValueError(
                            f"non-finite prediction/interval for {row['player_id']} {role} {stat}"
                        )
                    details.append(
                        {
                            "league": league, "target_season": season,
                            "player_id": row["player_id"], "player_name": row["player_name"],
                            "role": role, "from_season": row["from_season"], "to_season": row["to_season"],
                            "pre_team": row["pre_team"],
                            "pre_sample": row["pre_sample"], "post_sample": row["post_sample"],
                            "stat": stat, "prior": row[f"pre_{stat}"], "prediction": prediction,
                            "actual": actual, "error": prediction - actual,
                            "absolute_error": abs(prediction - actual),
                            "lower": lower, "upper": upper,
                            "covered_80": bool(lower <= actual <= upper),
                        }
                    )
        if not details:
            continue
        metrics = _score_metrics(details)
        by_target[str(season)] = {
            "counts": {
                "training_pairs": int(len(train)),
                "candidate_adjacent_pairs": candidates,
                "evaluation_pairs": len(eval_rows),
                "detail_rows": len(details),
                "park_adjusted_pairs": park_applied,
                "scored_roles": sorted(scored_roles),
            },
            "latest_training_to_season": int(train["to_season"].max()),
            "metrics": metrics,
            "links": links,
            "detail": details,
        }
        all_detail.extend(details)
        summary.extend({"target_season": season, **row} for row in metrics)
        link_rows.extend(links)

    pooled = [{"league": league, **row} for row in _score_metrics(all_detail)]
    gate = international_ship_gate(pooled)
    metadata = {
        "data_mode": "real",
        "league": league,
        "link": f"{league}->MLB",
        "target_seasons": seasons,
        "training_cutoff": {"destination_season_lt": "target season"},
        "parameters": {
            "n_boot": n_boot, "seed": seed, "min_pa": min_pa, "min_ip": min_ip,
            "estimator": str(estimator),
        },
        "counts": {
            "real_lines": source_counts,
            "targets_scored": len(by_target),
            "targets_skipped": len({item["target_season"] for item in skipped} - {int(k) for k in by_target}),
            "detail_rows": len(all_detail),
            "park_adjusted_pairs": sum(t["counts"]["park_adjusted_pairs"] for t in by_target.values()),
            "cohort_pairs": int(len(cohort)),
        },
        "input_sha256": checksums,
        "skipped": skipped,
        "crosswalk": "chadwick" if crosswalk else "exact ids only",
        "error_definition": "prediction minus actual",
        "baseline": "carry-forward foreign rate",
    }
    return {
        "metadata": metadata,
        "gate": gate,
        "pooled": pooled,
        "summary": summary,
        "links": link_rows,
        "by_target": by_target,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if value is not None and _finite(value) else "n/a"


def _metric_table(rows: list[dict[str, Any]], *, with_target: bool = False) -> list[str]:
    columns = (["target"] if with_target else []) + [
        "role", "stat", "n", "MAE", "baseline MAE", "80% coverage",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        digits = 3 if row["stat"] in {"fip", "era"} else 4
        cells = ([str(row["target_season"])] if with_target else []) + [
            str(row["role"]), str(row["stat"]), str(row["n"]),
            _fmt(row["mae"], digits), _fmt(row["baseline_mae"], digits),
            f"{float(row['coverage_80']) * 100:.1f}%",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _link_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | link | role | stat | pooled factor | n | n_export | n_import | direction_gap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_season']} | {row['link']} | {row['role']} | {row['stat']} | "
            f"{_fmt(row['factor'], 3)} | {row['n']} | {row['n_export']} | {row['n_import']} | "
            f"{_fmt(row['direction_gap'], 3)} |"
        )
    return lines


def render_international_markdown(payload: dict[str, Any]) -> str:
    """Readable report for one `run_international_rolling` payload."""
    meta = payload["metadata"]
    gate = payload["gate"]
    league = meta["league"]
    lines = [
        f"# {league} -> MLB international rolling validation",
        "",
        f"Targets: {', '.join(str(s) for s in meta['target_seasons'])}. Baseline: {meta['baseline']}. "
        f"Estimator {meta['parameters']['estimator']}, n_boot {meta['parameters']['n_boot']}, "
        f"seed {meta['parameters']['seed']}, min PA {meta['parameters']['min_pa']:g}, "
        f"min IP {meta['parameters']['min_ip']:g}. Error = prediction minus actual.",
        "",
        f"## Ship gate: {gate['verdict']}",
        "",
        "PASS needs model MAE < baseline MAE and 80% coverage in "
        f"[{gate['criteria']['coverage_range'][0]:.2f}, {gate['criteria']['coverage_range'][1]:.2f}] "
        f"for batter wOBA and pitcher FIP; INSUFFICIENT when n < {gate['criteria']['min_n']}.",
        "",
    ]
    for row in gate["rows"]:
        detail = "; ".join(row["reasons"]) if row["reasons"] else "model beats baseline, coverage in range"
        lines.append(f"- {row['role']} {row['stat']} (n={row['n']}): {row['status']} - {detail}")
    lines += ["", "## Pooled metrics", ""]
    lines += _metric_table(payload["pooled"]) if payload["pooled"] else ["No scored exports."]
    lines += ["", "## Per-target metrics", ""]
    lines += _metric_table(payload["summary"], with_target=True) if payload["summary"] else ["None."]
    lines += ["", "## Pooled link factors and direction gap", ""]
    lines += _link_table(payload["links"]) if payload["links"] else ["None."]
    if meta["skipped"]:
        lines += ["", "## Skipped", ""]
        for item in meta["skipped"]:
            role = f" ({item['role']})" if item.get("role") else ""
            lines.append(f"- {item['target_season']}{role}: {item['reason']}")
    lines.append("")
    return "\n".join(lines)


def write_international_report(payload: dict[str, Any], out_dir: Path | str) -> dict[str, Path]:
    """Write ``international-rolling-<league>.json`` and ``.md`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"international-rolling-{payload['metadata']['league'].lower()}"
    json_path, md_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_international_markdown(payload))
    return {"json": json_path, "markdown": md_path}


def render_international_doc(payloads: list[dict[str, Any]]) -> str:
    """Concise results doc for one or more `run_international_rolling` payloads.

    Every number comes from the payloads; only the limitations prose is fixed
    text (with counts interpolated where the payload has them).
    """
    if not payloads:
        raise ValueError("at least one international-rolling payload is required")
    payloads = sorted(payloads, key=lambda p: p["metadata"]["league"])
    first = payloads[0]["metadata"]
    params = first["parameters"]
    seasons = first["target_seasons"]
    lines = [
        "# International backtest results (NPB and KBO to MLB)",
        "",
        "Generated by `rosetta international-rolling`; the numbers below are copied from its JSON "
        "output, do not edit them by hand. Regenerate after any snapshot, factor-model or "
        "park-factor change.",
        "",
        "## What was run",
        "",
        f"- Command: `rosetta international-rolling --league <NPB|KBO> --targets {seasons[0]}-{seasons[-1]} "
        f"--boot {params['n_boot']}` (seed {params['seed']}, estimator {params['estimator']}, "
        f"minimum {params['min_pa']:g} PA / {params['min_ip']:g} IP on both sides).",
        "- For each target season T the factor model is refit on every transferred-player pair "
        "(both directions, all leagues) with `to_season < T`. NPB and KBO links are pooled across "
        "directions with a direction term and shrink toward the chained AAA product.",
        "- Scored set: exports, meaning players with a real league season in T-1 and a real MLB "
        "season in T. The MLB-T line is compared with `translate_line` of the T-1 line.",
        "- Baseline: the T-1 foreign rate carried forward unchanged. Error is prediction minus actual. "
        "Coverage is the share of actual outcomes inside the model's 80% band.",
        f"- Crosswalk: {first['crosswalk']}. Targets with no exports are recorded as skipped.",
        "",
        "## Ship gate",
        "",
        "PASS needs model MAE below baseline MAE and 80% coverage in "
        f"[{GATE_COVERAGE_RANGE[0]:.2f}, {GATE_COVERAGE_RANGE[1]:.2f}] for batter wOBA and pitcher FIP "
        f"on the pooled result. INSUFFICIENT when a gated stat has fewer than {GATE_MIN_N} scored rows.",
        "",
        "| league | verdict | batter wOBA (n) | pitcher FIP (n) | failing or insufficient |",
        "|---|---|---|---|---|",
    ]
    for payload in payloads:
        gate = payload["gate"]
        by_stat = {(r["role"], r["stat"]): r for r in gate["rows"]}
        bat, pit = by_stat[("batter", "woba")], by_stat[("pitcher", "fip")]
        why = "; ".join(f"{r['role']} {r['stat']}: {', '.join(r['reasons'])}" for r in gate["failing"]) or "none"
        lines.append(
            f"| {payload['metadata']['league']} | {gate['verdict']} | {bat['status']} ({bat['n']}) | "
            f"{pit['status']} ({pit['n']}) | {why} |"
        )

    for payload in payloads:
        meta = payload["metadata"]
        league = meta["league"]
        lines += ["", f"## {league}", "", f"### Pooled metrics, {league} exports", ""]
        lines += _metric_table(payload["pooled"]) if payload["pooled"] else ["No scored exports."]
        headline = [r for r in payload["summary"] if r["stat"] in {"woba", "fip"}]
        if headline:
            lines += ["", f"### Per-target wOBA and FIP, {league}", ""]
            lines += _metric_table(headline, with_target=True)
        if payload["links"]:
            lines += ["", f"### Pooled link factors and direction gap, {league}", ""]
            lines += [
                "`direction_gap` is log(export factor / import factor) in the "
                f"{league}->destination orientation; blank when one side has no usable rows.",
                "",
            ]
            lines += _link_table(payload["links"])
        if meta["skipped"]:
            lines += ["", f"### Skipped, {league}", ""]
            lines += [
                f"- {i['target_season']}{' (' + i['role'] + ')' if i.get('role') else ''}: {i['reason']}"
                for i in meta["skipped"]
            ]

    park_pairs = sum(p["metadata"]["counts"]["park_adjusted_pairs"] for p in payloads)
    lines += [
        "",
        "## Limitations",
        "",
        "- Thin samples. Exports with both-side playing time are few, and KBO movers are rarer "
        "still; read the n column before the MAE column. A small pooled n makes both the MAE "
        "comparison and the coverage figure noisy, and the gate marks it INSUFFICIENT rather than "
        "guessing.",
        "- Snapshot coverage is partial. Real lines in the snapshots (batters plus pitchers, all "
        "seasons): "
        + ", ".join(
            f"{p['metadata']['league']} {p['metadata']['counts']['real_lines'][p['metadata']['league']]}"
            for p in payloads
        )
        + f", MLB {payloads[0]['metadata']['counts']['real_lines']['MLB']}. NPB and KBO seasons "
        "hold only the players the ingest kept, so exports missing from the snapshots never "
        "reach the scored set.",
        "- History is short. The committed snapshots only go back to 2018, so early targets train "
        "on very little. The local backfill to 2005 has not run; rerun this after it does.",
        "- Ages are imputed (27.0) for NPB and KBO rows until the age refresh runs, so age "
        "adjustment is a no-op for those endpoints and aging differences are absorbed into the "
        "factor and the residual spread.",
        f"- Park factors are empty until NPB and KBO game logs are scraped: {park_pairs} scored "
        "pair(s) had a park adjustment. Park effects are therefore inside the error here.",
        "- No fielding or defense data is used here; only rate stats (wOBA, FIP and components) are "
        "scored. Grades, WAR and dollars are not validated by this run.",
        "- Selection: only players who reached enough MLB playing time appear. This is not a "
        "promotion, injury or playing-time model, and survivors tend to be the better performers.",
        "- Leakage control is by destination season (`to_season < T`). The snapshots are "
        "retrospective, not as-of archives, and bootstrap intervals resample rows rather than "
        "player clusters (a player can appear in several training seasons).",
        "- Pooled n counts each exported player-season once per stat; a player with several "
        "MLB-adjacent seasons can appear in more than one target.",
        "",
    ]
    return "\n".join(lines)
