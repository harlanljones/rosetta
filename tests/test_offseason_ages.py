"""Offseason 2026-27 WP1/WP2 checks: age provenance, season ranges, check-snapshots.

All offline: snapshot files are written into tmp_path, never the committed data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rosetta import cli
from rosetta.models.aging import age_adjust_rate, aging_multiplier

REPO = Path(__file__).resolve().parents[1]

_BATTER_HEADER = (
    "player_id,player_name,season,league,team,role,age,pa,g,bb_pct,k_pct,iso,babip,"
    "hr_pct,avg,obp,slg,woba,home_pa,road_pa,home_woba,road_woba,park_id,source,"
    "is_synthetic,age_source\n"
)
_PITCHER_HEADER = (
    "player_id,player_name,season,league,team,role,age,ip,g,gs,k_pct,bb_pct,hr_fb,era,"
    "fip,home_pa,road_pa,home_era,road_era,park_id,source,is_synthetic,age_source\n"
)


def _bat(pid: str, league: str, season: int, pa: float, age_source: str, synthetic: bool = False) -> str:
    return (
        f"{pid},Name {pid},{season},{league},Team,batter,27.0,{pa},100,0.1,0.2,0.2,0.3,"
        f"0.03,0.25,0.3,0.4,0.35,{pa / 2},{pa / 2},0.35,0.35,T,src,{synthetic},{age_source}\n"
    )


def _pit(pid: str, league: str, season: int, ip: float, age_source: str, synthetic: bool = False) -> str:
    return (
        f"{pid},Name {pid},{season},{league},Team,pitcher,27.0,{ip},20,0,0.2,0.08,0.1,3.5,"
        f"3.6,100,100,3.5,3.5,T,src,{synthetic},{age_source}\n"
    )


def _write(root: Path, name: str, header: str, rows: list[str]) -> None:
    (root / name).write_text(header + "".join(rows))


def _run_check(snapshots: Path):
    return CliRunner().invoke(cli.app, ["check-snapshots", "--snapshots", str(snapshots)])


# --- aging skip ------------------------------------------------------------


def test_age_adjust_rate_skip_returns_rate_unchanged() -> None:
    assert age_adjust_rate(0.200, "iso", 24, 30, role="batter", skip=True) == 0.200
    assert age_adjust_rate(3.10, "fip", 24, 33, role="pitcher", skip=True) == 3.10


def test_age_adjust_rate_default_behavior_unchanged() -> None:
    expected = 0.300 + aging_multiplier("woba", 30, 33, role="batter")
    assert age_adjust_rate(0.300, "woba", 30, 33, role="batter") == pytest.approx(expected)
    assert age_adjust_rate(0.300, "woba", 30, 33, role="batter", skip=False) == pytest.approx(expected)
    assert age_adjust_rate(0.200, "iso", 24, 30, role="batter") < 0.200


# --- season spec -------------------------------------------------------------


def test_parse_season_spec_single_list_and_range() -> None:
    from rosetta.ingest.statsapi import parse_season_spec

    assert parse_season_spec("2024") == [2024]
    assert parse_season_spec("2022,2024") == [2022, 2024]
    assert parse_season_spec("2005-2008") == [2005, 2006, 2007, 2008]
    assert parse_season_spec("2005-2007,2010") == [2005, 2006, 2007, 2010]
    assert parse_season_spec(" 2024 - 2025 , 2030 ") == [2024, 2025, 2030]


def test_parse_season_spec_dedupes_and_sorts() -> None:
    from rosetta.ingest.statsapi import parse_season_spec

    assert parse_season_spec("2025,2024-2025,2024") == [2024, 2025]


def test_parse_season_spec_rejects_bad_input() -> None:
    from rosetta.ingest.statsapi import parse_season_spec

    for bad in ("", " , ", "abc", "2010-2005", "2005-", "2005-x"):
        with pytest.raises(ValueError):
            parse_season_spec(bad)


def test_fetch_commands_accept_range_syntax(tmp_path: Path, monkeypatch) -> None:
    import rosetta.ingest.kbo as kbo
    import rosetta.ingest.npb as npb
    import rosetta.ingest.statsapi as statsapi

    seen: dict[str, list[int]] = {}

    def record(key: str):
        def _fake(years, *args, **kwargs):
            seen[key] = list(years)
            return tmp_path

        return _fake

    monkeypatch.setattr(statsapi, "write_statsapi_snapshots", record("aaa"))
    monkeypatch.setattr(kbo, "write_kbo_snapshots", record("kbo"))
    monkeypatch.setattr(npb, "write_npb_snapshots", record("npb"))
    runner = CliRunner()
    for args in (
        ["fetch-statsapi", "--seasons", "2005-2006", "--sport", "11", "--snapshots", str(tmp_path)],
        ["fetch-kbo", "--seasons", "2005-2006", "--snapshots", str(tmp_path)],
        ["fetch-npb", "--seasons", "2005-2006", "--snapshots", str(tmp_path)],
    ):
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 0, result.output
    assert seen == {"aaa": [2005, 2006], "kbo": [2005, 2006], "npb": [2005, 2006]}


def test_fetch_command_rejects_bad_season_range(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app, ["fetch-kbo", "--seasons", "2026-2005", "--snapshots", str(tmp_path)]
    )
    assert result.exit_code != 0


# --- Makefile -----------------------------------------------------------------


def test_live_data_keeps_ages_on_and_spans_2005_2026() -> None:
    text = (REPO / "Makefile").read_text()
    assert "--no-with-ages" not in text
    assert "seq -s, 2005 2026" in text
    live_lines = [ln for ln in text.splitlines() if ln.startswith("\trosetta fetch-")]
    assert len(live_lines) == 4
    for line in live_lines:
        assert "$(SEASONS)" in line, line


# --- check-snapshots ----------------------------------------------------------


def test_check_snapshots_passes_when_all_qualified_ages_are_birth_year(tmp_path: Path) -> None:
    _write(tmp_path, "npb_batters.csv", _BATTER_HEADER, [_bat("npb-1", "NPB", 2026, 300, "birth_year")])
    _write(tmp_path, "kbo_pitchers.csv", _PITCHER_HEADER, [_pit("kbo-9", "KBO", 2026, 150, "birth_year")])
    result = _run_check(tmp_path)
    assert result.exit_code == 0, result.output


def test_check_snapshots_fails_on_imputed_qualified_batter(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "npb_batters.csv",
        _BATTER_HEADER,
        [
            _bat("npb-1", "NPB", 2026, 300, "imputed"),
            _bat("npb-2", "NPB", 2026, 300, "imputed"),
            _bat("npb-3", "NPB", 2025, 99, "imputed"),
            _bat("npb-4", "NPB", 2025, 300, "birth_year"),
        ],
    )
    result = _run_check(tmp_path)
    assert result.exit_code == 1
    assert "NPB 2026 batter: 2" in result.output
    assert "2025" not in result.output  # 2025 rows are birth_year or below threshold


def test_check_snapshots_fails_on_imputed_pitcher_and_counts_kbo(tmp_path: Path) -> None:
    _write(tmp_path, "kbo_pitchers.csv", _PITCHER_HEADER, [_pit("kbo-1", "KBO", 2025, 31, "imputed")])
    _write(tmp_path, "kbo_batters.csv", _BATTER_HEADER, [_bat("kbo-2", "KBO", 2025, 100, "imputed")])
    result = _run_check(tmp_path)
    assert result.exit_code == 1
    assert "KBO 2025 pitcher: 1" in result.output
    assert "KBO 2025 batter: 1" in result.output


def test_check_snapshots_ignores_low_playing_time_and_synthetic(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "npb_batters.csv",
        _BATTER_HEADER,
        [
            _bat("npb-1", "NPB", 2026, 99, "imputed"),
            _bat("npb-2", "NPB", 2026, 500, "imputed", synthetic=True),
        ],
    )
    _write(tmp_path, "npb_pitchers.csv", _PITCHER_HEADER, [_pit("npb-3", "NPB", 2026, 29.9, "imputed")])
    assert _run_check(tmp_path).exit_code == 0


def test_check_snapshots_treats_empty_age_source_as_failure(tmp_path: Path) -> None:
    _write(tmp_path, "npb_batters.csv", _BATTER_HEADER, [_bat("npb-1", "NPB", 2026, 300, "")])
    result = _run_check(tmp_path)
    assert result.exit_code == 1


def test_check_snapshots_fails_when_age_source_column_missing(tmp_path: Path) -> None:
    legacy_header = _BATTER_HEADER.replace(",age_source", "")
    legacy_row = _bat("npb-1", "NPB", 2026, 300, "").rsplit(",", 1)[0] + "\n"
    (tmp_path / "npb_batters.csv").write_text(legacy_header + legacy_row)
    assert _run_check(tmp_path).exit_code == 1


def test_check_snapshots_with_no_npb_kbo_files_passes(tmp_path: Path) -> None:
    _write(tmp_path, "aaa_batters.csv", _BATTER_HEADER, [_bat("mlbam-1", "AAA", 2026, 300, "")])
    assert _run_check(tmp_path).exit_code == 0
