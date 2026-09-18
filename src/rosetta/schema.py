"""Normalized seasonal schema shared by every league."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

LEAGUE_CHAIN: tuple[str, ...] = ("CPBL", "KBO", "NPB", "AAA", "MLB")

HITTER_RATE_COLS = ("bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba")
PITCHER_RATE_COLS = ("k_pct", "bb_pct", "hr_fb", "era", "fip")


class League(StrEnum):
    MLB = "MLB"
    AAA = "AAA"
    NPB = "NPB"
    KBO = "KBO"
    CPBL = "CPBL"
    CUBA = "CUBA"


class Role(StrEnum):
    BATTER = "batter"
    PITCHER = "pitcher"


class SeasonLine(BaseModel):
    """One player-season in the normalized schema."""

    player_id: str
    player_name: str
    season: int
    league: League
    team: str = ""
    role: Role
    age: float
    # Playing time
    pa: float = 0.0
    ip: float = 0.0
    g: float = 0.0
    gs: float = 0.0
    # Hitter rates
    bb_pct: float | None = None
    k_pct: float | None = None
    iso: float | None = None
    babip: float | None = None
    hr_pct: float | None = None
    avg: float | None = None
    obp: float | None = None
    slg: float | None = None
    woba: float | None = None
    # Pitcher rates
    hr_fb: float | None = None
    era: float | None = None
    fip: float | None = None
    # Splits for park factors
    home_pa: float = 0.0
    road_pa: float = 0.0
    home_woba: float | None = None
    road_woba: float | None = None
    home_era: float | None = None
    road_era: float | None = None
    # Metadata
    park_id: str = ""
    source: str = "snapshot"
    is_synthetic: bool = False


HITTER_COLUMNS: list[str] = [
    "player_id",
    "player_name",
    "season",
    "league",
    "team",
    "role",
    "age",
    "pa",
    "g",
    "bb_pct",
    "k_pct",
    "iso",
    "babip",
    "hr_pct",
    "avg",
    "obp",
    "slg",
    "woba",
    "home_pa",
    "road_pa",
    "home_woba",
    "road_woba",
    "park_id",
    "source",
    "is_synthetic",
]

PITCHER_COLUMNS: list[str] = [
    "player_id",
    "player_name",
    "season",
    "league",
    "team",
    "role",
    "age",
    "ip",
    "g",
    "gs",
    "k_pct",
    "bb_pct",
    "hr_fb",
    "era",
    "fip",
    "home_pa",
    "road_pa",
    "home_era",
    "road_era",
    "park_id",
    "source",
    "is_synthetic",
]


def empty_hitter_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=HITTER_COLUMNS)


def empty_pitcher_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PITCHER_COLUMNS)


def validate_hitter_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in HITTER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"hitter frame missing columns: {missing}")
    out = df[HITTER_COLUMNS].copy()
    out["role"] = "batter"
    return out


def validate_pitcher_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PITCHER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"pitcher frame missing columns: {missing}")
    out = df[PITCHER_COLUMNS].copy()
    out["role"] = "pitcher"
    return out


class TranslationResult(BaseModel):
    player_id: str
    player_name: str
    season: int
    from_league: League
    to_league: League = League.MLB
    role: Role
    age: float
    path: list[str]
    # Point estimates (MLB-equivalent rates)
    rates: dict[str, float]
    # Uncertainty bands from bootstrap of mover cohort
    lower: dict[str, float] = Field(default_factory=dict)
    upper: dict[str, float] = Field(default_factory=dict)
    # Recombined summary stats
    woba: float | None = None
    woba_low: float | None = None
    woba_high: float | None = None
    fip: float | None = None
    fip_low: float | None = None
    fip_high: float | None = None
    sample_pa: float = 0.0
    sample_ip: float = 0.0
    n_movers_path: int = 0
    notes: list[str] = Field(default_factory=list)


Side = Literal["pre", "post"]
