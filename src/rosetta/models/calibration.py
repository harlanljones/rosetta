"""Leakage-safe prior-fold affine calibration for AAA→MLB translations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CALIBRATION_MIN_ROWS = 20


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def fit_prior_affine(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit per role/stat affine maps from completed earlier-fold detail rows."""
    if not details:
        return {}
    frame = pd.DataFrame(details)
    corrections: dict[str, dict[str, Any]] = {}
    for (role, stat), group in frame.groupby(["role", "stat"], sort=True):
        prediction_column = "uncalibrated_prediction" if "uncalibrated_prediction" in group else "prediction"
        x = pd.to_numeric(group[prediction_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < CALIBRATION_MIN_ROWS:
            continue
        design = np.column_stack((np.ones(len(x), dtype=float), x))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        if not all(_finite(value) for value in (intercept, slope)):
            continue
        residual = y - (intercept + slope * x)
        residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        residual_abs_q85 = float(np.quantile(np.abs(residual), 0.85)) if len(residual) else 0.0
        key = f"{role}:{stat}"
        corrections[key] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "resid_sd": residual_sd if _finite(residual_sd) else 0.0,
            "resid_abs_q85": residual_abs_q85 if _finite(residual_abs_q85) else 0.0,
            "n": int(len(x)),
        }
    return corrections


def _clip_calibrated(stat: str, value: float) -> float:
    if stat in {"bb_pct", "k_pct", "hr_pct", "babip", "iso", "woba", "hr_fb"}:
        return float(np.clip(value, 0.0, 1.0))
    if stat in {"era", "fip"}:
        return float(max(0.5, value))
    return float(value)


def apply_affine_stat(
    stat: str,
    center: float,
    low: float,
    high: float,
    correction: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Apply one stat's affine correction to point estimate and interval."""
    if correction is None:
        return center, low, high
    intercept = float(correction["intercept"])
    slope = float(correction["slope"])
    new_center = intercept + slope * center
    new_low = intercept + slope * low
    new_high = intercept + slope * high
    new_low, new_high = min(new_low, new_high), max(new_low, new_high)
    residual_half = float(
        correction.get("resid_abs_q85", 1.28 * float(correction.get("resid_sd", 0.0)))
    )
    half = max((new_high - new_low) / 2.0, residual_half)
    new_center = _clip_calibrated(stat, new_center)
    new_low = _clip_calibrated(stat, new_center - half)
    new_high = _clip_calibrated(stat, new_center + half)
    return new_center, new_low, new_high


def calibration_applies(path: list[str], to_league: str) -> bool:
    """Production calibration is fit on the AAA→MLB link only."""
    return to_league == "MLB" and "AAA" in path
