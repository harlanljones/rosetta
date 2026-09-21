"""Export a compact, deterministic JSON snapshot for the static portfolio demo.

Reads already-generated artifacts (``data/outputs/leaderboard.json``,
``data/outputs/historical-backtest-{target}/historical-backtest.json``,
``data/outputs/historical-rolling/historical-rolling.json``) and writes a
small, self-describing bundle (``meta.json``, ``leaderboard.json``,
``backtest.json``, ``analysis.json``) intended to be copied verbatim into a separate static
site. This script never touches the network and never regenerates data with
different parameters than what is already on disk — it only shells out to
``make`` targets (offline) when an expected artifact is missing.

The public leaderboard invariant from AGENTS.md applies here too: only real
(non-synthetic) rows are exported, and the script fails loudly rather than
silently falling back to synthetic fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = ROOT / "data" / "outputs" / "leaderboard.json"
ROLLING_PATH = ROOT / "data" / "outputs" / "historical-rolling" / "historical-rolling.json"

# Rounding precision by field-name suffix/prefix, applied to every float in
# the leaderboard/backtest payloads to keep output small and readable.
RATE_FIELDS = {
    "woba",
    "bb_pct",
    "k_pct",
    "iso",
    "babip",
    "hr_pct",
    "hr_fb",
    "mae",
    "rmse",
    "baseline_mae",
    "baseline_rmse",
    "coverage_80",
}
ERA_LIKE_FIELDS = {"era", "fip"}


def _round_value(key: str, value: Any) -> Any:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    base_key = key
    for prefix in ("mle_", "_low", "_high"):
        base_key = base_key.replace(prefix, "")
    if base_key in ERA_LIKE_FIELDS or key.startswith(("era", "fip")) or "era" in key or "fip" in key:
        return round(float(value), 3)
    if any(base_key == f or base_key.endswith(f) for f in RATE_FIELDS):
        return round(float(value), 4)
    if isinstance(value, float) and value.is_integer():
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def _round_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _round_value(k, v) for k, v in sorted(record.items())}


def _round_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    """Round a backtest/rolling metrics row using its own ``stat`` field.

    Metric rows use generic field names (``mae``, ``rmse``, ...) that don't
    encode the underlying stat the way leaderboard columns do (``mle_fip``),
    so precision is chosen from ``record["stat"]`` instead of the key name.
    """
    is_era_like = record.get("stat") in ERA_LIKE_FIELDS
    out: dict[str, Any] = {}
    for key, value in sorted(record.items()):
        if key in ("mae", "rmse", "baseline_mae", "baseline_rmse"):
            out[key] = round(float(value), 3 if is_era_like else 4)
        elif key == "coverage_80":
            out[key] = round(float(value), 3)
        else:
            out[key] = value
    return out


def _round_player_row(row: dict[str, Any], precision: int) -> dict[str, Any]:
    """Round a player-prediction row's numeric fields to a fixed precision."""
    numeric_fields = {"prior_2024_aaa", "predicted", "actual", "lower", "upper", "absolute_error"}
    out: dict[str, Any] = {}
    for key, value in sorted(row.items()):
        if key in numeric_fields and isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = round(float(value), precision)
        else:
            out[key] = value
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_artifacts_exist(backtest_path: Path) -> None:
    regen_targets = (
        (LEADERBOARD_PATH, "leaderboard"),
        (backtest_path, "historical-backtest"),
        (ROLLING_PATH, "historical-rolling"),
    )
    missing = [(p, t) for p, t in regen_targets if not p.exists()]
    if not missing:
        return
    for path, target in missing:
        print(f"[export_demo] {path} missing, running `make {target}` (offline)", file=sys.stderr)
        subprocess.run(["make", target], cwd=ROOT, check=True)
    still_missing = [str(p) for p, _ in regen_targets if not p.exists()]
    if still_missing:
        raise SystemExit(f"Required artifacts still missing after regeneration: {still_missing}")


def git_short_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_leaderboard(raw: dict[str, Any]) -> dict[str, Any]:
    data_mode = raw.get("data_mode")
    if data_mode != "real":
        raise SystemExit(
            f"Refusing to export leaderboard: data_mode={data_mode!r}, expected 'real'. "
            "Never ship a synthetic-backed demo board."
        )
    boards_out: dict[str, list[dict[str, Any]]] = {}
    kept_fields = (
        "player_id",
        "player_name",
        "from_league",
        "to_league",
        "role",
        "season",
        "team",
        "path",
        "age",
        "source",
        "sample_pa",
        "sample_ip",
        "n_movers_path",
        "notes",
        "woba",
        "woba_low",
        "woba_high",
        "fip",
        "fip_low",
        "fip_high",
        "mle_bb_pct",
        "mle_bb_pct_low",
        "mle_bb_pct_high",
        "mle_k_pct",
        "mle_k_pct_low",
        "mle_k_pct_high",
        "mle_iso",
        "mle_iso_low",
        "mle_iso_high",
        "mle_babip",
        "mle_babip_low",
        "mle_babip_high",
        "mle_hr_pct",
        "mle_hr_pct_low",
        "mle_hr_pct_high",
        "mle_hr_fb",
        "mle_hr_fb_low",
        "mle_hr_fb_high",
        "mle_era",
        "mle_era_low",
        "mle_era_high",
        "mle_fip",
        "mle_fip_low",
        "mle_fip_high",
    )
    zero_synthetic_checked = 0
    for role, rows in sorted(raw.get("boards", {}).items()):
        if not isinstance(rows, list):
            continue
        synthetic_rows = [r for r in rows if r.get("is_synthetic")]
        if synthetic_rows:
            raise SystemExit(
                f"Refusing to export leaderboard: {len(synthetic_rows)} synthetic rows found "
                f"in board {role!r}. Public/demo boards are real-data-only."
            )
        zero_synthetic_checked += len(rows)
        trimmed = []
        for row in rows:
            trimmed_row = {k: row.get(k) for k in kept_fields if k in row}
            trimmed.append(_round_record(trimmed_row))
        # Stable ordering: role's natural sort order (already ranked), then
        # break ties on player_id for determinism.
        boards_out[role] = trimmed
    if zero_synthetic_checked == 0:
        raise SystemExit("Refusing to export leaderboard: zero rows found across all boards.")
    return {
        "season": raw.get("season"),
        "data_mode": data_mode,
        "source_counts": raw.get("source_counts", {}),
        "boards": boards_out,
    }


def build_backtest(
    backtest_raw: dict[str, Any], rolling_raw: dict[str, Any], target_season: int
) -> dict[str, Any]:
    metadata = backtest_raw.get("metadata", {})
    if metadata.get("data_mode") != "real":
        raise SystemExit("Refusing to export backtest: metadata.data_mode is not 'real'.")

    metrics = sorted(
        (_round_metric_record(m) for m in backtest_raw.get("metrics", [])),
        key=lambda m: (m["role"], m["stat"]),
    )
    if len(metrics) != 11:
        raise SystemExit(
            f"Expected 11 role/stat metric rows for the {target_season} backtest, found {len(metrics)}."
        )

    rolling_summary = sorted(
        (_round_metric_record(m) for m in rolling_raw.get("summary", [])),
        key=lambda m: (m["target_season"], m["role"], m["stat"]),
    )
    rolling_pooled = sorted(
        (_round_metric_record(m) for m in rolling_raw.get("pooled", [])),
        key=lambda m: (m["role"], m["stat"]),
    )

    detail = backtest_raw.get("detail", [])
    player_rows: dict[str, list[dict[str, Any]]] = {"batter_woba": [], "pitcher_fip": []}
    for row in detail:
        stat = row.get("stat")
        role = row.get("role")
        key = None
        if role == "batter" and stat == "woba":
            key = "batter_woba"
        elif role == "pitcher" and stat == "fip":
            key = "pitcher_fip"
        if key is None:
            continue
        precision = 3 if key == "pitcher_fip" else 4
        player_rows[key].append(
            _round_player_row(
                {
                    "player_id": row.get("player_id"),
                    "player_name": row.get("player_name"),
                    "prior_2024_aaa": row.get("prior"),
                    "predicted": row.get("prediction"),
                    "actual": row.get("actual"),
                    "lower": row.get("lower"),
                    "upper": row.get("upper"),
                    "covered_80": row.get("covered_80"),
                    "absolute_error": row.get("absolute_error"),
                },
                precision,
            )
        )
    for key in player_rows:
        player_rows[key] = sorted(player_rows[key], key=lambda r: (r["player_name"], r["player_id"]))

    return {
        f"target_season_{target_season}": {
            "metrics": metrics,
        },
        f"rolling_2022_{target_season}": {
            "target_seasons": sorted(rolling_raw.get("metadata", {}).get("target_seasons", [])),
            "summary_by_season": rolling_summary,
            "pooled": rolling_pooled,
        },
        "player_predictions": player_rows,
    }


def _analysis_row(row: dict[str, Any], target_season: int) -> dict[str, Any]:
    """Make one player error legible without hiding the model's uncertainty."""
    prediction = row.get("prediction")
    actual = row.get("actual")
    prior = row.get("prior")
    lower = row.get("lower")
    upper = row.get("upper")
    signed_error = prediction - actual if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)) else None
    baseline_signed_error = prior - actual if isinstance(prior, (int, float)) and isinstance(actual, (int, float)) else None
    baseline_absolute_error = abs(baseline_signed_error) if baseline_signed_error is not None else None
    absolute_error = row.get("absolute_error")
    improvement = baseline_absolute_error - absolute_error if baseline_absolute_error is not None and absolute_error is not None else None
    interval_width = upper - lower if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) else None
    position = (actual - lower) / interval_width if interval_width and isinstance(actual, (int, float)) else None
    role = row.get("role")
    stat = row.get("stat")
    precision = 3 if stat in ERA_LIKE_FIELDS else 4
    bucket_limits = (0.5, 1.0, 1.5) if stat in ERA_LIKE_FIELDS else (0.025, 0.05, 0.075)
    error_bucket = (
        "small" if absolute_error is not None and absolute_error < bucket_limits[0]
        else "moderate" if absolute_error is not None and absolute_error < bucket_limits[1]
        else "large" if absolute_error is not None and absolute_error < bucket_limits[2]
        else "extreme" if absolute_error is not None else None
    )
    rounded = _round_record(
        {
            "season": target_season,
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "role": role,
            "stat": stat,
            "from_league": row.get("from_league"),
            "predicted": prediction,
            "actual": actual,
            "prior": prior,
            "lower": lower,
            "upper": upper,
            "signed_error": signed_error,
            "absolute_error": absolute_error,
            "baseline_signed_error": baseline_signed_error,
            "baseline_absolute_error": baseline_absolute_error,
            "improvement": improvement,
            "error_bucket": error_bucket,
            "direction": "over" if signed_error and signed_error > 0 else "under" if signed_error and signed_error < 0 else "even",
            "covered_80": row.get("covered_80"),
            "interval_miss": row.get("covered_80") is False,
            "interval_midpoint": (lower + upper) / 2 if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) else None,
            "interval_width": interval_width,
            "interval_position": position,
            "sample": row.get("post_sample"),
        }
    )
    for key in ("signed_error", "absolute_error", "interval_width", "interval_position"):
        if isinstance(rounded.get(key), (int, float)):
            rounded[key] = round(float(rounded[key]), precision)
    return rounded


def _analysis_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate player errors by season, role, and stat for the report UI."""
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["season"], row["role"], row["stat"]), []).append(row)
    summary = []
    for (season, role, stat), group in sorted(groups.items()):
        errors = [r["signed_error"] for r in group if r["signed_error"] is not None]
        abs_errors = [r["absolute_error"] for r in group if r["absolute_error"] is not None]
        baseline_errors = [r["baseline_absolute_error"] for r in group if r["baseline_absolute_error"] is not None]
        covered = [r["covered_80"] for r in group if isinstance(r["covered_80"], bool)]
        summary.append(
            _round_metric_record(
                {
                    "target_season": season,
                    "role": role,
                    "stat": stat,
                    "n": len(group),
                    "mae": sum(abs_errors) / len(abs_errors) if abs_errors else None,
                    "baseline_mae": sum(baseline_errors) / len(baseline_errors) if baseline_errors else None,
                    "bias": sum(errors) / len(errors) if errors else None,
                    "overprediction_rate": sum(e > 0 for e in errors) / len(errors) if errors else None,
                    "coverage_80": sum(covered) / len(covered) if covered else None,
                }
            )
        )
    return summary


def build_analysis(rolling_raw: dict[str, Any]) -> dict[str, Any]:
    """Build the player-error layer that bridges metrics and scouting language."""
    metadata = rolling_raw.get("metadata", {})
    if metadata.get("data_mode") != "real":
        raise SystemExit("Refusing to export analysis: rolling metadata.data_mode is not 'real'.")

    rows: list[dict[str, Any]] = []
    for season, payload in sorted(rolling_raw.get("by_target", {}).items(), key=lambda item: int(item[0])):
        rows.extend(_analysis_row(row, int(season)) for row in payload.get("detail", []))
    if not rows:
        raise SystemExit("Refusing to export analysis: rolling backtest has no player-level detail.")

    current_season = max(row["season"] for row in rows)
    featured_stats = {("batter", "woba"), ("pitcher", "fip")}
    featured = [row for row in rows if (row["role"], row["stat"]) in featured_stats]
    current_featured = [row for row in featured if row["season"] == current_season]
    top_current = sorted(current_featured, key=lambda row: (-row["absolute_error"], row["player_name"]))[:12]
    top_historical = sorted(featured, key=lambda row: (-row["absolute_error"], row["player_name"]))[:16]

    role_summary = []
    for role in ("batter", "pitcher"):
        role_rows = [
            row for row in current_featured
            if row["season"] == current_season and row["role"] == role
        ]
        errors = [row["absolute_error"] for row in role_rows]
        signed = [row["signed_error"] for row in role_rows]
        covered = [row["covered_80"] for row in role_rows if isinstance(row["covered_80"], bool)]
        role_summary.append(
            _round_record(
                {
                    "season": current_season,
                    "role": role,
                    "n": len(role_rows),
                    "mae": sum(errors) / len(errors),
                    "bias": sum(signed) / len(signed),
                    "overprediction_rate": sum(error > 0 for error in signed) / len(signed),
                    "coverage_80": sum(covered) / len(covered),
                }
            )
        )

    return {
        "schema_version": "analysis.v1",
        "data_mode": "real",
        "current_season": current_season,
        "featured_stats": ["woba", "fip"],
        "definitions": {
            "signed_error": "prediction - actual; positive means the model overpredicted",
            "coverage_80": "share of observed outcomes inside the model's 80% interval",
            "interpretation": "Observed error patterns are descriptive; possible causes are hypotheses, not causal findings.",
        },
        "scope": {
            "seasons": sorted({row["season"] for row in rows}),
            "player_stat_rows": len(rows),
            "featured_player_stat_rows": len(featured),
            "source": "historical-rolling.json detail",
        },
        "summary_by_season": _analysis_summary(rows),
        "current_metrics": sorted(
            (
                _round_metric_record(metric)
                for metric in rolling_raw.get("by_target", {}).get(str(current_season), {}).get("metrics", [])
            ),
            key=lambda metric: (metric["role"], metric["stat"]),
        ),
        "current_role_summary": role_summary,
        "current_featured_misses": top_current,
        "historical_featured_misses": top_historical,
    }


def build_meta(
    leaderboard_raw: dict[str, Any],
    backtest_raw: dict[str, Any],
    rolling_raw: dict[str, Any],
    input_paths: tuple[Path, ...],
) -> dict[str, Any]:
    bt_meta = backtest_raw.get("metadata", {})
    roll_meta = rolling_raw.get("metadata", {})
    params = bt_meta.get("parameters", {})

    # Deterministic timestamp: derived from the source artifacts' own
    # filesystem mtimes (max across inputs), never wall-clock "now". Given a
    # fixed set of input files this is stable across repeated runs.
    generated_at = datetime.fromtimestamp(
        max(p.stat().st_mtime for p in input_paths), tz=UTC
    ).isoformat()

    checksums = dict(sorted(bt_meta.get("input_sha256", {}).items()))

    limitations = [
        "Evaluation is limited to players with observed MLB playing time above "
        "threshold; this is not a promotion or playing-time model.",
        "Only the real adjacent AAA-to-MLB link is backtested here; international "
        "links (KBO/NPB/CPBL/Cuba) are shown on the leaderboard but not "
        "held-out-validated against MLB outcomes in this demo.",
        "No park, role, or target-season aging adjustment is applied.",
        "An AAA era floor (2019+) excludes pre-livelier-ball AAA seasons from "
        "training to avoid mixing run environments.",
        "Bootstrap resamples rows, not player clusters; intervals are "
        "approximate for players with few comparable transferred pairs.",
        "This is a static, point-in-time export — not a live/hosted runtime.",
    ]

    return {
        "data_mode": "real",
        "generated_at": generated_at,
        "rosetta_git_commit": git_short_sha(),
        "estimator": params.get("estimator"),
        "era_floor": params.get("era_floor"),
        "source_counts": leaderboard_raw.get("source_counts", {}),
        "backtest_source_season": bt_meta.get("source_season"),
        "backtest_target_season": bt_meta.get("target_season"),
        "backtest_training_cutoff": bt_meta.get("training_cutoff"),
        "backtest_latest_training_to_season": bt_meta.get("latest_training_to_season"),
        "rolling_target_seasons": sorted(roll_meta.get("target_seasons", [])),
        "input_sha256": checksums,
        "limitations": limitations,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-season",
        type=int,
        default=2026,
        help="Target season represented by the historical backtest (default: 2026).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "outputs" / "demo",
        help="Output directory for the exported demo bundle.",
    )
    args = parser.parse_args()

    backtest_path = ROOT / "data" / "outputs" / f"historical-backtest-{args.target_season}" / "historical-backtest.json"
    ensure_artifacts_exist(backtest_path)

    leaderboard_raw = load_json(LEADERBOARD_PATH)
    backtest_raw = load_json(backtest_path)
    rolling_raw = load_json(ROLLING_PATH)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_out = build_leaderboard(leaderboard_raw)
    backtest_out = build_backtest(backtest_raw, rolling_raw, args.target_season)
    analysis_out = build_analysis(rolling_raw)
    meta_out = build_meta(
        leaderboard_raw,
        backtest_raw,
        rolling_raw,
        (LEADERBOARD_PATH, backtest_path, ROLLING_PATH),
    )

    write_json(out_dir / "leaderboard.json", leaderboard_out)
    write_json(out_dir / "backtest.json", backtest_out)
    write_json(out_dir / "analysis.json", analysis_out)
    write_json(out_dir / "meta.json", meta_out)

    total_bytes = sum((out_dir / name).stat().st_size for name in ("leaderboard.json", "backtest.json", "analysis.json", "meta.json"))
    n_batter = len(leaderboard_out["boards"].get("batter", []))
    n_pitcher = len(leaderboard_out["boards"].get("pitcher", []))
    n_woba_players = len(backtest_out["player_predictions"]["batter_woba"])
    n_fip_players = len(backtest_out["player_predictions"]["pitcher_fip"])
    print(
        f"[export_demo] wrote {out_dir} "
        f"({total_bytes:,} bytes total; leaderboard {n_batter}B/{n_pitcher}P rows; "
        f"backtest players {n_woba_players} wOBA / {n_fip_players} FIP; "
        f"analysis rows {analysis_out['scope']['player_stat_rows']})"
    )


if __name__ == "__main__":
    main()
