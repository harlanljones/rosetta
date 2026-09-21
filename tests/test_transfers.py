from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.transfers import generate_transfers


def test_generate_transfers_empty_with_empty_register(tmp_path: Path) -> None:
    """Test that an empty Chadwick register produces no transfers.

    Passes a header-only register file instead of None to avoid FileNotFoundError
    in CI environments where data/snapshots/chadwick_people.csv is gitignored.
    Production behavior (raising on missing register) is preserved.
    """
    from rosetta.ingest.generate_snapshots import build_snapshots

    root = build_snapshots(out_dir=tmp_path, seed=1)

    # Create a header-only Chadwick register with the columns that
    # load_register and build_id_crosswalk expect
    register_path = tmp_path / "chadwick_people.csv"
    register_df = pd.DataFrame(columns=[
        "key_uuid",
        "key_mlbam",
        "key_retro",
        "key_bbref",
        "key_bbref_minors",
        "key_fangraphs",
        "key_npb",
        "key_kbo",
        "name_last",
        "name_first",
        "name_given",
        "birth_year",
        "mlb_played_first",
        "mlb_played_last",
    ])
    register_df.to_csv(register_path, index=False)

    t = generate_transfers(register_path=register_path, snapshot_dir=root, max_season_gap=1)
    assert t.empty


def test_generate_transfers_detects_cross_league(tmp_path: Path) -> None:
    cw = pd.DataFrame(
        {
            "key_uuid": ["u1"],
            "key_mlbam": [100001.0],
            "key_npb": [200001.0],
            "key_bbref": [None],
            "name_last": ["Test"],
            "name_first": ["Player"],
        }
    )
    cw.to_csv(tmp_path / "chadwick_people.csv", index=False)
    seasons = pd.DataFrame(
        [
            {"player_id": "mlbam-100001", "season": 2023, "league": "MLB"},
            {"player_id": "npb-200001", "season": 2022, "league": "NPB"},
        ]
    )
    from rosetta.schema import HITTER_COLUMNS

    for league in ("MLB", "NPB"):
        pdf = seasons[seasons["league"] == league].copy()
        pdf["player_name"] = "Test Player"
        pdf["role"] = "batter"
        pdf["age"] = 27.0
        pdf["team"] = ""
        pdf["pa"] = 100.0
        pdf["bb_pct"] = 0.08
        pdf["k_pct"] = 0.2
        pdf["iso"] = 0.15
        pdf["babip"] = 0.3
        pdf["hr_pct"] = 0.03
        pdf["avg"] = 0.270
        pdf["obp"] = 0.340
        pdf["slg"] = 0.450
        pdf["woba"] = 0.32
        pdf["home_pa"] = 50.0
        pdf["road_pa"] = 50.0
        pdf["home_woba"] = 0.32
        pdf["road_woba"] = 0.32
        pdf["park_id"] = ""
        pdf["source"] = "test"
        pdf["is_synthetic"] = False
        pdf.reindex(columns=HITTER_COLUMNS).to_csv(
            tmp_path / f"{league.lower()}_batters.csv", index=False
        )
    for league in ("MLB", "NPB"):
        empty = pd.DataFrame(
            columns=[
                "player_id", "player_name", "season", "league", "role", "team", "age",
                "ip", "g", "gs", "k_pct", "bb_pct", "hr_fb", "era", "fip",
                "home_pa", "road_pa", "home_era", "road_era", "park_id",
                "source", "is_synthetic",
            ]
        )
        empty.to_csv(tmp_path / f"{league.lower()}_pitchers.csv", index=False)

    t = generate_transfers(
        register_path=tmp_path / "chadwick_people.csv",
        snapshot_dir=tmp_path,
        max_season_gap=2,
    )
    assert len(t) == 1
    assert t.iloc[0]["from_league"] == "NPB"
    assert t.iloc[0]["to_league"] == "MLB"