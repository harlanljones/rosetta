from __future__ import annotations

import json
from pathlib import Path

from rosetta.ingest.loaders import load_league_seasons
from rosetta.ingest.statsapi import (
    age_at_season,
    attach_ages,
    normalize_hitting,
    normalize_pitching,
    parse_ip,
    write_statsapi_snapshots,
)

FIXTURE = Path(__file__).parent / "fixtures" / "statsapi_aaa_2024.json"


def _splits() -> dict:
    return json.loads(FIXTURE.read_text())


def test_parse_ip() -> None:
    assert abs(parse_ip("178.2") - (178 + 2 / 3)) < 1e-9
    assert parse_ip("0.0") == 0.0
    assert parse_ip(None) == 0.0


def test_normalize_hitting_rates() -> None:
    df = normalize_hitting(_splits()["hitting"], 2024, "AAA")
    assert len(df) == 3
    row = df.iloc[0]
    assert row["league"] == "AAA"
    assert row["player_id"].startswith("mlbam-")
    assert row["source"] == "statsapi"
    assert not row["is_synthetic"]
    pa = row["pa"]
    assert 0 < row["bb_pct"] < 0.3
    assert 0 < row["k_pct"] < 0.6
    assert 0 <= row["iso"] < 0.5
    assert 0.15 < row["babip"] < 0.5
    assert 0.2 < row["woba"] < 0.55
    # Internal consistency: rates recombine from counting stats
    assert abs(row["bb_pct"] + row["k_pct"]) < 0.6
    assert pa > 0


def test_normalize_pitching_rates() -> None:
    df = normalize_pitching(_splits()["pitching"], 2024, "AAA")
    assert len(df) == 3
    row = df.iloc[0]
    assert row["league"] == "AAA"
    assert 0 < row["k_pct"] < 0.6
    assert 0 < row["bb_pct"] < 0.3
    assert 0 <= row["hr_fb"] <= 1.0
    assert 0 < row["era"] < 15
    assert 0 < row["fip"] < 15
    assert row["ip"] > 0
    assert row["gs"] <= row["g"]


def test_age_at_season() -> None:
    assert age_at_season("1995-09-20", 2024) == 29.0
    assert age_at_season(None, 2024) is None
    assert age_at_season("unknown", 2024) is None


def test_attach_ages_from_pool() -> None:
    import pandas as pd

    pool = json.loads((Path(__file__).parent / "fixtures" / "statsapi_sport11_players_2024.json").read_text())[
        "people"
    ]
    first = pool[0]
    df = pd.DataFrame(
        [
            {"player_id": f"mlbam-{first['id']}", "age": 27.0},
            {"player_id": "mlbam-0", "age": 27.0},
        ]
    )
    out = attach_ages(df, 2024, 11, people=pool)
    expect = 2024 - int(first["birthDate"][:4])
    assert out.loc[out["player_id"] == f"mlbam-{first['id']}", "age"].iloc[0] == float(expect)
    assert out.loc[out["player_id"] == "mlbam-0", "age"].iloc[0] == 27.0


def test_write_merges_with_synthetic(tmp_path: Path, monkeypatch) -> None:
    import rosetta.ingest.statsapi as api

    fx = _splits()

    def fake_fetch(season: int, sport_id: int, **kwargs):
        label = kwargs.get("league_label") or "AAA"
        return (
            api.normalize_hitting(fx["hitting"], season, label),
            api.normalize_pitching(fx["pitching"], season, label),
        )

    monkeypatch.setattr(api, "fetch_league_season", fake_fetch)
    # The writer's optional age backfill must remain offline-testable; without
    # this mock it would make an incidental network request for the player pool.
    monkeypatch.setattr(api, "fetch_sport_players", lambda season, sport_id: [])
    # Seed a synthetic snapshot to merge against
    from rosetta.ingest.generate_snapshots import build_snapshots

    build_snapshots(out_dir=tmp_path, seed=7)
    before = load_league_seasons("AAA", role="batter", snapshot_dir=tmp_path)
    assert (before["is_synthetic"].astype(str) == "True").all()
    write_statsapi_snapshots([2024], 11, out_dir=tmp_path)
    after = load_league_seasons("AAA", role="batter", snapshot_dir=tmp_path)
    assert len(after) > len(before)
    assert ((after["source"] == "statsapi") & (after["season"] == 2024)).any()
    pit = load_league_seasons("AAA", role="pitcher", snapshot_dir=tmp_path)
    assert ((pit["source"] == "statsapi") & (pit["season"] == 2024)).any()
