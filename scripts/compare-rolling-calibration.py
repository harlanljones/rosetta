"""Compare raw and prior-fold-calibrated historical rolling backtests.

This is an offline, reproducible experiment.  Both candidates use the same
real snapshots, target seasons, factor estimator, and bootstrap seed; the
only changed parameter is ``calibration``.  The script emits JSON so model
reports can preserve the exact comparison rather than copying table values by
hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rosetta.backtest.historical import run_rolling_backtest


def _metric_map(payload: dict[str, Any], field: str = "pooled") -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["role"]), str(row["stat"])): row for row in payload[field]}


def _compare(raw: dict[str, Any], calibrated: dict[str, Any]) -> list[dict[str, Any]]:
    raw_metrics = _metric_map(raw)
    calibrated_metrics = _metric_map(calibrated)
    rows: list[dict[str, Any]] = []
    for role, stat in sorted(raw_metrics):
        before = raw_metrics[(role, stat)]
        after = calibrated_metrics[(role, stat)]
        rows.append(
            {
                "role": role,
                "stat": stat,
                "n": int(after["n"]),
                "raw_mae": float(before["mae"]),
                "calibrated_mae": float(after["mae"]),
                "delta_mae": float(after["mae"] - before["mae"]),
                "raw_rmse": float(before["rmse"]),
                "calibrated_rmse": float(after["rmse"]),
                "delta_rmse": float(after["rmse"] - before["rmse"]),
                "raw_coverage_80": float(before["coverage_80"]),
                "calibrated_coverage_80": float(after["coverage_80"]),
                "delta_coverage_80": float(after["coverage_80"] - before["coverage_80"]),
                "persistence_baseline_mae": float(after["baseline_mae"]),
                "persistence_baseline_rmse": float(after["baseline_rmse"]),
            }
        )
    return rows


def _target_rows(raw: dict[str, Any], calibrated: dict[str, Any], target: int) -> list[dict[str, Any]]:
    raw_metrics = _metric_map(raw["by_target"][str(target)], field="metrics")
    calibrated_metrics = _metric_map(calibrated["by_target"][str(target)], field="metrics")
    rows: list[dict[str, Any]] = []
    for role, stat in sorted(raw_metrics):
        before = raw_metrics[(role, stat)]
        after = calibrated_metrics[(role, stat)]
        rows.append(
            {
                "target_season": target,
                "role": role,
                "stat": stat,
                "n": int(after["n"]),
                "raw_mae": float(before["mae"]),
                "calibrated_mae": float(after["mae"]),
                "delta_mae": float(after["mae"] - before["mae"]),
                "raw_rmse": float(before["rmse"]),
                "calibrated_rmse": float(after["rmse"]),
                "delta_rmse": float(after["rmse"] - before["rmse"]),
                "raw_coverage_80": float(before["coverage_80"]),
                "calibrated_coverage_80": float(after["coverage_80"]),
                "delta_coverage_80": float(after["coverage_80"] - before["coverage_80"]),
                "persistence_baseline_mae": float(after["baseline_mae"]),
                "persistence_baseline_rmse": float(after["baseline_rmse"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--target-seasons", default="2022,2023,2024,2025,2026")
    parser.add_argument("--boot", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pa", type=float, default=80.0)
    parser.add_argument("--min-ip", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    seasons = [int(part.strip()) for part in args.target_seasons.split(",") if part.strip()]
    common = {
        "snapshot_dir": str(args.snapshots),
        "target_seasons": seasons,
        "n_boot": args.boot,
        "seed": args.seed,
        "min_pa": args.min_pa,
        "min_ip": args.min_ip,
    }
    raw = run_rolling_backtest(**common, calibration="none")
    calibrated = run_rolling_backtest(**common, calibration="prior_affine")
    result = {
        "experiment": "raw_vs_prior_affine",
        "configuration": {
            **common,
            "estimator": "ratio_of_means",
            "era_floor": {"AAA": 2019},
            "calibration_selection": "Each target uses only strictly earlier target folds; current target excluded.",
        },
        "pooled": _compare(raw, calibrated),
        "by_target": [
            row
            for target in seasons
            if str(target) in raw["by_target"] and str(target) in calibrated["by_target"]
            for row in _target_rows(raw, calibrated, target)
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
