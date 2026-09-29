"""Merge Savant-derived home/road splits into AAA snapshot rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.savant import aggregate_batters, load_pitches
from rosetta.paths import DATA_DIR, SNAPSHOT_DIR


def enrich_aaa_batter_splits(
    *,
    snapshot_dir: Path | None = None,
    raw_dir: Path | None = None,
    season: int = 2024,
) -> int:
    """Update ``aaa_batters.csv`` split columns from cached Savant minors CSVs."""
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    raw = Path(raw_dir) if raw_dir else DATA_DIR / "raw" / "savant"
    paths = sorted(raw.glob(f"minors_{season}-*.csv"))
    if not paths:
        paths = sorted(raw.glob("minors_*.csv"))
    pitches = load_pitches(paths)
    if pitches.empty:
        return 0
    savant = aggregate_batters(pitches, season, "AAA")
    if savant.empty:
        return 0
    path = root / "aaa_batters.csv"
    if not path.exists():
        return 0
    aaa = pd.read_csv(path)
    split_cols = ["home_pa", "road_pa", "home_woba", "road_woba"]
    savant_idx = savant.set_index("player_id")
    updated = 0
    for idx, row in aaa.iterrows():
        pid = str(row.get("player_id", ""))
        if pid not in savant_idx.index:
            continue
        src = savant_idx.loc[pid]
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        for col in split_cols:
            if col in src and pd.notna(src[col]):
                aaa.at[idx, col] = src[col]
        updated += 1
    aaa.to_csv(path, index=False)
    return updated
