from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.npb import (
    fetch_birth_year,
    fetch_hitting,
    fetch_pitching,
    normalize_hitting,
    normalize_pitching,
    parse_table,
)

FIX = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


def test_parse_batting_table() -> None:
    headers, rows = parse_table(_html("npb_bat_c.html"))
    assert headers[:6] == ["Rk", "Player", "Team", "AVG", "G", "PA"]
    assert len(rows) == 18
    assert rows[0]["player_name"] == "Kozono, Kaito"
    assert rows[0]["team"] == "C"
    assert len(rows[0]["values"]) == len(headers)


def test_parse_pitching_table() -> None:
    headers, rows = parse_table(_html("npb_pit_c.html"))
    assert "ERA" in headers and "BF" in headers and "SO" in headers
    assert rows[0]["player_name"] == "Saiki, Hiroto"
    assert len(rows) > 0


def test_parse_id_index() -> None:
    from rosetta.ingest.npb import parse_id_index

    ids = parse_id_index(_html("npb_index_k.html"))
    assert ids["Kadowaki, Makoto"] == "11515157"
    assert all(len(v) == 8 and v.isdigit() for v in ids.values())


def test_parse_birth_year() -> None:
    assert fetch_birth_year(_html("npb_player.html")) == 1990


class _StaticSession:
    """Feeds fixture HTML instead of network."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def get(self, url: str) -> str:
        return self.mapping[url]

    def get_season(self, url: str, season: int) -> str:
        return self.mapping[url]


def _session() -> _StaticSession:
    from rosetta.ingest import npb

    return _StaticSession(
        {
            npb.stat_url(2026, "bat_c"): _html("npb_bat_c.html"),
            npb.stat_url(2026, "bat_p"): _html("npb_bat_c.html"),
            npb.stat_url(2026, "pit_c"): _html("npb_pit_c.html"),
            npb.stat_url(2026, "pit_p"): _html("npb_pit_c.html"),
        }
    )


def test_fetch_hitting() -> None:
    df = fetch_hitting(2026, session=_session(), id_map={"Kozono, Kaito": "81983888"})
    assert len(df) == 36  # Central (18) + Pacific (18) fixture tables
    for col in ("PA", "AB", "H", "HR", "BB", "IBB", "HP", "SO", "OBP", "SLG", "_npb_id"):
        assert col in df.columns, col
    assert (df["_npb_id"] == "81983888").any()


def test_fetch_pitching() -> None:
    df = fetch_pitching(2026, session=_session())
    assert len(df) > 0
    for col in ("ERA", "G", "IP", "BF", "H", "HR", "BB", "SO"):
        assert col in df.columns, col


def test_normalize_hitting_rates() -> None:
    df = fetch_hitting(2026, session=_session(), id_map={"Kozono, Kaito": "81983888"})
    out = normalize_hitting(df, 2026, birth_years={"81983888": 1990})
    assert len(out) == 36
    row = out.iloc[0]
    assert row["league"] == "NPB"
    assert row["player_id"] == "npb-81983888"
    assert row["source"] == "npb-official"
    assert 0 < row["bb_pct"] < 0.3
    assert 0 < row["k_pct"] < 0.6
    assert 0.2 < row["woba"] < 0.55
    assert row["age"] == 36.0
    from rosetta.schema import validate_hitter_frame

    validate_hitter_frame(out)


def test_normalize_pitching_rates() -> None:
    df = fetch_pitching(2026, session=_session())
    out = normalize_pitching(df, 2026)
    assert len(out) > 0
    row = out.iloc[0]
    assert row["league"] == "NPB"
    assert 0 < row["k_pct"] < 0.6
    assert 0 < row["bb_pct"] < 0.3
    assert 0 <= row["hr_fb"] <= 1.0
    assert row["ip"] > 0
    assert row["era"] > 0
    from rosetta.schema import validate_pitcher_frame

    validate_pitcher_frame(out)


def test_write_merges_with_synthetic(tmp_path: Path, monkeypatch) -> None:
    import rosetta.ingest.npb as api
    from rosetta.ingest.generate_snapshots import build_snapshots
    from rosetta.ingest.loaders import load_league_seasons

    sess = _session()

    def fake_fetch(season: int, **kwargs):
        id_map = {"Kozono, Kaito": "81983888"}
        return (
            api.normalize_hitting(api.fetch_hitting(season, session=sess, id_map=id_map), season),
            api.normalize_pitching(api.fetch_pitching(season, session=sess), season),
        )

    monkeypatch.setattr(api, "fetch_npb_season", fake_fetch)
    build_snapshots(out_dir=tmp_path, seed=7)
    before = load_league_seasons("NPB", role="batter", snapshot_dir=tmp_path)
    assert (before["is_synthetic"].astype(str) == "True").all()
    api.write_npb_snapshots([2026], out_dir=tmp_path, with_ages=False)
    after = load_league_seasons("NPB", role="batter", snapshot_dir=tmp_path)
    assert len(after) > len(before)
    assert ((after["source"] == "npb-official") & (after["season"] == 2026)).any()


def test_snapshot_append_filters_synthetic(tmp_path: Path, monkeypatch) -> None:
    """Live rows must never clobber real rows already in the snapshot."""
    import rosetta.ingest.npb as api

    sess = _session()

    def fake_fetch(season: int, **kwargs):
        id_map = {"Kozono, Kaito": "81983888"}
        return (
            api.normalize_hitting(api.fetch_hitting(season, session=sess, id_map=id_map), season),
            api.normalize_pitching(api.fetch_pitching(season, session=sess), season),
        )

    monkeypatch.setattr(api, "fetch_npb_season", fake_fetch)
    api.write_npb_snapshots([2026], out_dir=tmp_path, with_ages=False)
    path = tmp_path / "npb_batters.csv"
    df = pd.read_csv(path)
    assert ((df["source"] == "npb-official") & (df["league"] == "NPB")).all()
    assert (df["is_synthetic"].astype(str) == "False").all()
