"""Delta-method aging curves for cross-season comparisons.

Simple quadratic curves peaking near 27 — good enough for v1 and fully
transparent. Replace with empirical Tango/Clay curves without API changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgingCurve:
    # Annual additive rate change per year of aging (sign: higher-is-better stats decline).
    batter_better_high: dict[str, float]
    pitcher_better_high: dict[str, float]
    peak_age: float = 27.0


DEFAULT_CURVE = AgingCurve(
    peak_age=27.0,
    # approx change in rate per year of aging past peak (negative = decline)
    batter_better_high={
        "bb_pct": -0.0015,
        "k_pct": 0.0020,  # K% rises (worse) with age; higher is worse for hitters
        "iso": -0.0040,
        "babip": -0.0025,
        "hr_pct": -0.0010,
        "woba": -0.0035,
    },
    pitcher_better_high={
        "k_pct": -0.0030,  # K% falls with age
        "bb_pct": 0.0015,  # BB% rises
        "hr_fb": 0.0020,
        "era": 0.08,
        "fip": 0.06,
    },
)


def aging_multiplier(stat: str, age_from: float, age_to: float, role: str = "batter") -> float:
    """Return additive aging delta to apply to `stat` when moving age_from → age_to.

    We use additive deltas on rates (delta method), not multiplicative, so a
    one-year age gap is removed before estimating league factors.
    """
    curve = DEFAULT_CURVE
    table = curve.batter_better_high if role == "batter" else curve.pitcher_better_high
    slope = table.get(stat, 0.0)
    # Distance from peak matters; approximate with average slope × Δage
    return slope * (age_to - age_from)


def age_adjust_rate(
    rate: float,
    stat: str,
    age_from: float,
    age_to: float,
    role: str = "batter",
) -> float:
    """Shift a pre-move rate forward to the post-move age before comparing."""
    return rate + aging_multiplier(stat, age_from, age_to, role=role)
