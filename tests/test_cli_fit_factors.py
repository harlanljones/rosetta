"""CLI smoke: leaderboard/fit-factors must not leak typer OptionInfo defaults."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rosetta import cli

_TRANSFERS_HEADER = (
    "player_id,player_name,role,from_league,to_league,from_season,to_season,"
    "age_from,age_to,holdout\n"
)
_BATTER_HEADER = (
    "player_id,player_name,season,league,team,role,age,pa,g,bb_pct,k_pct,iso,babip,"
    "hr_pct,avg,obp,slg,woba,home_pa,road_pa,home_woba,road_woba,park_id,source,is_synthetic\n"
)
_PITCHER_HEADER = (
    "player_id,player_name,season,league,team,role,age,ip,g,gs,k_pct,bb_pct,hr_fb,era,"
    "fip,home_pa,road_pa,home_era,road_era,park_id,source,is_synthetic\n"
)


def _minimal_snapshot_dir(tmp_path: Path) -> Path:
    """Header-only snapshots: enough to reach cohort assembly, empty cohort."""
    (tmp_path / "transfers.csv").write_text(_TRANSFERS_HEADER)
    (tmp_path / "kbo_batters.csv").write_text(_BATTER_HEADER)
    (tmp_path / "kbo_pitchers.csv").write_text(_PITCHER_HEADER)
    return tmp_path


def test_leaderboard_with_missing_factors_never_leaks_optioninfo(tmp_path: Path) -> None:
    """Regression: leaderboard_cmd calls fit_factors() directly, bypassing typer's
    default resolution, so Path | None options arrived as raw OptionInfo objects."""
    snapshots = _minimal_snapshot_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["leaderboard", "--snapshots", str(snapshots), "--out", str(tmp_path / "lb.json")],
    )
    assert "OptionInfo" not in str(result.exception or "")
    assert "OptionInfo" not in result.output
