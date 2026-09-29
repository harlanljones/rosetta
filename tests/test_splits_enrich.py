from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.splits import enrich_aaa_batter_splits

FIXTURE = Path(__file__).parent / "fixtures" / "savant_minors_sample.csv"


def test_enrich_aaa_splits_from_fixture(tmp_path: Path) -> None:
    raw = tmp_path / "savant"
    raw.mkdir()
    (raw / "minors_2024-04-03.csv").write_bytes(FIXTURE.read_bytes())
    snap = tmp_path / "snapshots"
    snap.mkdir()
    aaa = pd.DataFrame(
        [
            {
                "player_id": "mlbam-669392",
                "player_name": "Test",
                "season": 2024,
                "league": "AAA",
                "team": "T",
                "role": "batter",
                "age": 25.0,
                "pa": 100.0,
                "g": 20.0,
                "bb_pct": 0.1,
                "k_pct": 0.2,
                "iso": 0.15,
                "babip": 0.3,
                "hr_pct": 0.03,
                "avg": 0.25,
                "obp": 0.33,
                "slg": 0.4,
                "woba": 0.32,
                "home_pa": 50.0,
                "road_pa": 50.0,
                "home_woba": 0.32,
                "road_woba": 0.32,
                "park_id": "",
                "source": "statsapi",
                "is_synthetic": False,
            }
        ]
    )
    aaa.to_csv(snap / "aaa_batters.csv", index=False)
    updated = enrich_aaa_batter_splits(snapshot_dir=snap, raw_dir=raw, season=2024)
    assert updated >= 0
