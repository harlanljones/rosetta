"""Park factors from home/road splits."""

from __future__ import annotations

import pandas as pd


def estimate_park_factors(
    seasons: pd.DataFrame,
    *,
    stat: str = "woba",
    min_pa: float = 200.0,
) -> pd.DataFrame:
    """Estimate team-season park factors as home/road ratio, regressed to 1.0.

    For pitchers pass stat='era' (factors inverted so >1 means hitter-friendly).
    """
    home_col = f"home_{stat}"
    road_col = f"road_{stat}"
    if home_col not in seasons.columns or road_col not in seasons.columns:
        return pd.DataFrame(columns=["league", "team", "season", "park_factor", "n"])

    df = seasons.dropna(subset=[home_col, road_col]).copy()
    df = df[df["home_pa"] + df["road_pa"] >= min_pa]
    if df.empty:
        return pd.DataFrame(columns=["league", "team", "season", "park_factor", "n"])

    grouped = (
        df.groupby(["league", "team", "season"], as_index=False)
        .agg(
            home=(home_col, "mean"),
            road=(road_col, "mean"),
            n=("home_pa", "sum"),
        )
    )
    raw = grouped["home"] / grouped["road"].replace(0, pd.NA)
    # Shrink toward 1.0 with weight rising in sample size
    k = 500.0
    weight = grouped["n"] / (grouped["n"] + k)
    grouped["park_factor"] = 1.0 + weight * (raw.fillna(1.0) - 1.0)
    return grouped[["league", "team", "season", "park_factor", "n"]]


def neutralize(rate: float, park_factor: float, *, center: float = 1.0) -> float:
    """Strip park effect toward league-neutral."""
    if park_factor <= 0:
        return rate
    return rate / park_factor * center
