from __future__ import annotations

from pathlib import Path

from rosetta.ingest.kbo import (
    fetch_hitting,
    fetch_pitching,
    normalize_hitting,
    normalize_pitching,
    parse_birth_year,
    parse_table,
)

FIX = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


def test_parse_hitter_basic1() -> None:
    headers, rows = parse_table(_html("kbo_hitter_basic1.html"))
    assert headers[:6] == ["순위", "선수명", "팀명", "AVG", "G", "PA"]
    assert len(rows) == 30
    assert rows[0]["player_id"] != ""
    assert len(rows[0]["values"]) == len(headers)


def test_parse_hitter_basic2() -> None:
    headers, rows = parse_table(_html("kbo_hitter_basic2.html"))
    assert "BB" in headers and "SO" in headers and "OPS" in headers
    assert len(rows) == 30


def test_parse_pitcher_tables() -> None:
    h1, r1 = parse_table(_html("kbo_pitcher_basic1.html"))
    h2, r2 = parse_table(_html("kbo_pitcher_basic2.html"))
    assert "SO" in h1 and "IP" in h1
    assert "TBF" in h2
    assert len(r1) > 0 and len(r2) > 0


def test_parse_birth_year() -> None:
    assert parse_birth_year(_html("kbo_hitter_detail.html")) == 1993


class _StaticSession:
    """Feeds fixture HTML instead of network (no postback)."""

    def __init__(self, mapping: dict[str, str], delay: float = 0.0) -> None:
        self.mapping = mapping

    def get_season(self, url: str, season: int) -> str:
        return self.mapping[url]

    def get(self, url: str) -> str:
        return self.mapping[url]


def _session() -> _StaticSession:
    from rosetta.ingest import kbo

    return _StaticSession(
        {
            kbo.HIT_BASIC1: _html("kbo_hitter_basic1.html"),
            kbo.HIT_BASIC2: _html("kbo_hitter_basic2.html"),
            kbo.PIT_BASIC1: _html("kbo_pitcher_basic1.html"),
            kbo.PIT_BASIC2: _html("kbo_pitcher_basic2.html"),
        }
    )


def test_fetch_hitting_merges() -> None:
    df = fetch_hitting(2026, session=_session())
    assert len(df) == 30  # one row per Basic1 entry — no cartesian blowup
    for col in ("PA", "AB", "H", "HR", "BB", "SO", "OBP", "SLG", "_kbo_id"):
        assert col in df.columns, col


def test_normalize_hitting_rates() -> None:
    df = normalize_hitting(fetch_hitting(2026, session=_session()), 2026)
    assert len(df) == 30
    row = df.iloc[0]
    assert row["league"] == "KBO"
    assert row["player_id"].startswith("kbo-")
    assert row["source"] == "kbo-official"
    assert 0 < row["bb_pct"] < 0.3
    assert 0 < row["k_pct"] < 0.6
    assert 0.2 < row["woba"] < 0.55
    assert row["age"] == 27.0  # no birth years supplied
    aged = normalize_hitting(
        fetch_hitting(2026, session=_session()), 2026, birth_years={row["player_id"].split("-")[1]: 1993}
    )
    assert aged.iloc[0]["age"] == 33.0


def test_normalize_pitching_rates() -> None:
    df = normalize_pitching(fetch_pitching(2026, session=_session()), 2026)
    assert len(df) > 0
    row = df.iloc[0]
    assert row["league"] == "KBO"
    assert 0 < row["k_pct"] < 0.6
    assert 0 < row["bb_pct"] < 0.3
    assert 0 <= row["hr_fb"] <= 1.0
    assert row["ip"] > 0


def test_write_merges_with_synthetic(tmp_path: Path, monkeypatch) -> None:
    import rosetta.ingest.kbo as api
    from rosetta.ingest.generate_snapshots import build_snapshots
    from rosetta.ingest.loaders import load_league_seasons

    sess = _session()

    def fake_fetch(season: int, **kwargs):
        return (
            api.normalize_hitting(api.fetch_hitting(season, session=sess), season),
            api.normalize_pitching(api.fetch_pitching(season, session=sess), season),
        )

    monkeypatch.setattr(api, "fetch_kbo_season", fake_fetch)
    build_snapshots(out_dir=tmp_path, seed=7)
    before = load_league_seasons("KBO", role="batter", snapshot_dir=tmp_path)
    assert (before["is_synthetic"].astype(str) == "True").all()
    api.write_kbo_snapshots([2026], out_dir=tmp_path, with_ages=False)
    after = load_league_seasons("KBO", role="batter", snapshot_dir=tmp_path)
    assert len(after) > len(before)
    assert ((after["source"] == "kbo-official") & (after["season"] == 2026)).any()


def test_frames_validate() -> None:
    from rosetta.schema import validate_hitter_frame, validate_pitcher_frame

    validate_hitter_frame(normalize_hitting(fetch_hitting(2026, session=_session()), 2026))
    validate_pitcher_frame(normalize_pitching(fetch_pitching(2026, session=_session()), 2026))
