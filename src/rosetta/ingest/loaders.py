"""Load normalized season lines from committed snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR
from rosetta.schema import (
    empty_hitter_frame,
    empty_pitcher_frame,
    validate_hitter_frame,
    validate_pitcher_frame,
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"snapshot missing: {path}")
    return pd.read_csv(path)


def load_league_seasons(
    league: str,
    *,
    role: str = "batter",
    snapshot_dir: Path | None = None,
) -> pd.DataFrame:
    """Load one league's seasonal lines for batters or pitchers."""
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    path = root / f"{league.lower()}_{role}s.csv"
    df = _read_csv(path)
    if role == "batter":
        return validate_hitter_frame(df)
    if role == "pitcher":
        return validate_pitcher_frame(df)
    raise ValueError(f"unknown role: {role}")


def load_all_seasons(
    *,
    role: str = "batter",
    snapshot_dir: Path | None = None,
    leagues: list[str] | None = None,
) -> pd.DataFrame:
    """Stack every available league snapshot for a role."""
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    wanted = leagues or ["MLB", "AAA", "NPB", "KBO", "CPBL", "CUBA"]
    frames: list[pd.DataFrame] = []
    for league in wanted:
        path = root / f"{league.lower()}_{role}s.csv"
        if not path.exists():
            continue
        frames.append(load_league_seasons(league, role=role, snapshot_dir=root))
    if not frames:
        return empty_hitter_frame() if role == "batter" else empty_pitcher_frame()
    return pd.concat(frames, ignore_index=True)


def load_transfers(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Load the curated transferred-player pair table."""
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    path = root / "transfers.csv"
    df = _read_csv(path)
    required = {
        "player_id",
        "player_name",
        "role",
        "from_league",
        "to_league",
        "from_season",
        "to_season",
        "age_from",
        "age_to",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"transfers.csv missing columns: {sorted(missing)}")
    return df
