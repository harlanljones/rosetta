from __future__ import annotations

from pathlib import Path

from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons, load_transfers
from rosetta.schema import (
    HITTER_COLUMNS,
    PITCHER_COLUMNS,
    validate_hitter_frame,
    validate_pitcher_frame,
)


def test_build_and_load_snapshots(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path, seed=1)
    assert (root / "manifest.json").exists()
    assert (root / "transfers.csv").exists()
    bat = load_all_seasons(role="batter", snapshot_dir=root)
    pit = load_all_seasons(role="pitcher", snapshot_dir=root)
    assert len(bat) > 100
    assert len(pit) > 50
    validate_hitter_frame(bat)
    validate_pitcher_frame(pit)
    for col in HITTER_COLUMNS:
        assert col in bat.columns
    for col in PITCHER_COLUMNS:
        assert col in pit.columns
    transfers = load_transfers(root)
    assert set(["AAA", "MLB"]).issubset(set(transfers["to_league"]) | set(transfers["from_league"]))
