"""Leakage-safe historical evaluation of the real AAA-to-MLB link.

This module intentionally does not use the transfer index.  A transfer file is
useful for building the production cohort, but it is not an independent source
of truth for a retrospective evaluation: the evaluator constructs adjacent
season pairs directly from the normalized season snapshots.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.factors import (
    HITTER_STATS,
    LEAGUE_ERA_FLOOR,
    PITCHER_STATS,
    apply_era_floor,
    fit_factor_model,
)
from rosetta.paths import SNAPSHOT_DIR
from rosetta.translate.api import translate_line

_ROLES = ("batter", "pitcher")
_LEAGUES = ("AAA", "MLB")


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
        )
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
    aaa: pd.DataFrame,
    mlb: pd.DataFrame,
    *,
    target_season: int,
    min_pa: float,
    min_ip: float,
    role: str,
) -> tuple[list[dict[str, Any]], int]:
    """Build all exact adjacent AAA(s-1)->MLB(s) pairs through target."""
    if aaa.empty or mlb.empty:
        return [], 0
    stats = HITTER_STATS if role == "batter" else PITCHER_STATS
    sample = "pa" if role == "batter" else "ip"
    threshold = float(min_pa if role == "batter" else min_ip)
    aaa_idx = aaa.set_index(["player_id", "season"])
    mlb_idx = mlb.set_index(["player_id", "season"])
    rows: list[dict[str, Any]] = []
    candidates = 0

    for (player_id, to_season), post in mlb_idx.iterrows():
        from_season = int(to_season) - 1
        if int(to_season) > target_season:
            continue
        key = (player_id, from_season)
        if key not in aaa_idx.index:
            continue
        pre = aaa_idx.loc[key]
        # Duplicate keys were rejected above, but retain this guard for clear
        # errors if a caller supplies a non-standard DataFrame directly.
        if isinstance(pre, pd.DataFrame) or isinstance(post, pd.DataFrame):
            raise ValueError(f"ambiguous AAA-to-MLB pair for {player_id} {from_season}->{to_season}")
        candidates += 1
        pre_sample = pre.get(sample)
        post_sample = post.get(sample)
        if not _finite(pre_sample) or not _finite(post_sample):
            raise ValueError(
                f"non-finite {sample} for {player_id} AAA {from_season} or MLB {to_season}"
            )
        if float(pre_sample) < threshold or float(post_sample) < threshold:
            continue

        required = [
            ("age", pre.get("age"), "AAA", from_season),
            ("age", post.get("age"), "MLB", to_season),
            *[(stat, pre.get(stat), "AAA", from_season) for stat in stats],
            *[(stat, post.get(stat), "MLB", to_season) for stat in stats],
        ]
        bad = [f"{field} ({league} {season})" for field, value, league, season in required if not _finite(value)]
        if bad:
            raise ValueError(f"non-finite required values for {player_id}: {', '.join(bad)}")

        row: dict[str, Any] = {
            "player_id": str(player_id),
            "player_name": str(post.get("player_name", pre.get("player_name", player_id))),
            "role": role,
            "from_league": "AAA",
            "to_league": "MLB",
            "from_season": int(from_season),
            "to_season": int(to_season),
            "age_from": float(pre["age"]),
            "age_to": float(post["age"]),
            "pre_source": str(pre.get("source", "snapshot")),
            "post_source": str(post.get("source", "snapshot")),
            "pre_sample": float(pre_sample),
            "post_sample": float(post_sample),
        }
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


def run_historical_backtest(
    snapshot_dir: Path | str | None = None,
    *,
    target_season: int = 2025,
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

    by_role: dict[str, dict[str, pd.DataFrame]] = {}
    checksums: dict[str, str] = {}
    source_counts: dict[str, int] = {"AAA": 0, "MLB": 0}
    for role in _ROLES:
        by_role[role] = {}
        for league in _LEAGUES:
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
            by_role[role]["AAA"], by_role[role]["MLB"], target_season=target_season,
            min_pa=float(min_pa), min_ip=float(min_ip), role=role,
        )
        all_rows.extend(rows)
        candidates += count
    training_all = [row for row in all_rows if row["to_season"] < target_season]
    evaluation = [row for row in all_rows if row["from_season"] == source_season and row["to_season"] == target_season]
    if not training_all:
        raise ValueError(
            f"insufficient real AAA-to-MLB training pairs before {target_season}; "
            "need at least one adjacent pair with both playing-time thresholds met"
        )
    if not evaluation:
        raise ValueError(
            f"insufficient real AAA-to-MLB evaluation pairs for {source_season}->{target_season}; "
            "need at least one adjacent pair with both playing-time thresholds met"
        )

    train_frame_all = _training_frame(training_all)
    train_frame = apply_era_floor(train_frame_all, era_floor)
    excluded_era_floor = len(training_all) - len(train_frame)
    if train_frame.empty:
        raise ValueError(
            f"insufficient real AAA-to-MLB training pairs before {target_season} after the AAA "
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
        node = model.links.get(role, {}).get("AAA->MLB", {})
        missing = [stat for stat in stats if float(node.get(stat, {}).get("n", 0.0)) <= 0]
        if missing:
            raise ValueError(
                f"unfitted real factor(s) for {role} AAA->MLB: {', '.join(missing)}; "
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
            "season": row["from_season"], "league": "AAA", "role": row["role"],
            "age": row["age_from"], "pa": row["pre_sample"] if row["role"] == "batter" else 0.0,
            "ip": row["pre_sample"] if row["role"] == "pitcher" else 0.0,
        }
        pre.update({stat: row[f"pre_{stat}"] for stat in stats})
        translated = translate_line(pre, model=model, from_league="AAA", to_league="MLB")
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
                    "to_season": row["to_season"], "from_league": "AAA", "to_league": "MLB",
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
            "from_league": "AAA", "to_league": "MLB", "pre_source": row["pre_source"],
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
        "Only the real adjacent AAA-to-MLB link is evaluated; international links are not validated here. A player may contribute to multiple pre-target training seasons and to the target holdout.",
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
        "data_mode": "real", "target_season": target_season, "source_season": source_season,
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
    """
    seasons = [int(season) for season in target_seasons]
    if not seasons:
        raise ValueError("target_seasons must be non-empty")

    by_target: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    all_detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    excluded_era_floor_total = 0

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
        by_target[str(season)] = payload
        excluded_era_floor_total += int(
            payload["metadata"]["counts"].get("training_pairs_excluded_era_floor", 0)
        )
        for row in payload["detail"]:
            all_detail.append({"target_season": season, **row})
        for row in payload["metrics"]:
            summary.append({"target_season": season, **row})

    if not by_target:
        raise ValueError(
            "all requested target seasons were skipped for insufficient real data: "
            f"{skipped}"
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
