"""Fit production calibration artifacts from historical AAA→MLB folds."""

from __future__ import annotations

from typing import Any

from rosetta.backtest.historical import run_historical_backtest
from rosetta.models.calibration import fit_prior_affine


def fit_production_calibration(
    snapshot_dir: Any,
    *,
    fit_target_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026),
    n_boot: int = 250,
    seed: int = 42,
    min_pa: float = 80.0,
    min_ip: float = 30.0,
) -> dict[str, Any]:
    """Collect raw historical folds and fit one prior_affine map for production."""
    prior_details: list[dict[str, Any]] = []
    fit_through = 0
    for season in sorted(fit_target_seasons):
        try:
            payload = run_historical_backtest(
                snapshot_dir,
                target_season=season,
                n_boot=n_boot,
                seed=seed,
                min_pa=min_pa,
                min_ip=min_ip,
            )
        except ValueError:
            continue
        fit_through = max(fit_through, season)
        for row in payload["detail"]:
            prior_details.append(
                {
                    **row,
                    "uncalibrated_prediction": float(row["prediction"]),
                    "uncalibrated_lower": float(row["lower"]),
                    "uncalibrated_upper": float(row["upper"]),
                }
            )
    fits = fit_prior_affine(prior_details)
    return {
        "method": "prior_affine",
        "fit_target_seasons": list(fit_target_seasons),
        "fit_through_season": fit_through,
        "fits": fits,
    }
