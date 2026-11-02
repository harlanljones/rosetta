"""P2: leaderboard player lookup and CSV export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from apps.leaderboard import app as leaderboard_app


def _write_leaderboard(tmp_path: Path) -> Path:
    payload = {
        "season": 2025,
        "data_mode": "real",
        "source_counts": {"batter": {"kbo": 2}},
        "boards": {
            "batter": [
                {
                    "player_name": "Hyun-woo Lee",
                    "from_league": "KBO",
                    "woba": 0.412,
                    "woba_low": 0.35,
                    "woba_high": 0.47,
                },
                {
                    "player_name": "Ji-hwan Park",
                    "from_league": "KBO",
                    "woba": 0.395,
                    "woba_low": 0.33,
                    "woba_high": 0.46,
                },
            ],
            "pitcher": [
                {
                    "player_name": "Won-jun Kim",
                    "from_league": "KBO",
                    "fip": 3.42,
                    "fip_low": 2.9,
                    "fip_high": 4.1,
                },
            ],
        },
    }
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(payload))
    return path


def test_csv_export_contains_all_board_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "OUTPUT", _write_leaderboard(tmp_path))

    response = leaderboard_app.api_leaderboard_csv()

    rows = list(csv.DictReader(io.StringIO(response.body.decode())))
    assert len(rows) == 3
    assert {r["role"] for r in rows} == {"batter", "pitcher"}
    assert rows[0]["player_name"] == "Hyun-woo Lee"


def test_csv_export_matches_json_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "OUTPUT", _write_leaderboard(tmp_path))

    response = leaderboard_app.api_leaderboard_csv()
    rows = list(csv.DictReader(io.StringIO(response.body.decode())))

    batter = next(r for r in rows if r["player_name"] == "Won-jun Kim")
    assert batter["role"] == "pitcher"
    assert float(batter["fip"]) == 3.42


def test_player_lookup_matches_case_insensitive_substring(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "OUTPUT", _write_leaderboard(tmp_path))

    matches = leaderboard_app._find_player("park")

    assert [m["player_name"] for m in matches] == ["Ji-hwan Park"]
    assert matches[0]["role"] == "batter"


def test_player_lookup_spans_both_roles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "OUTPUT", _write_leaderboard(tmp_path))

    matches = leaderboard_app._find_player("Kim")

    assert [m["role"] for m in matches] == ["pitcher"]


def test_player_lookup_no_match_returns_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "OUTPUT", _write_leaderboard(tmp_path))

    assert leaderboard_app._find_player("zzzz") == []
    assert leaderboard_app._find_player("  ") == []


def test_player_template_renders_matches() -> None:
    row = {
        "role": "batter",
        "player_name": "Ji-hwan Park",
        "from_league": "KBO",
        "woba": 0.395,
        "woba_low": 0.33,
        "woba_high": 0.46,
    }
    html = leaderboard_app.TEMPLATES.get_template("player.html").render(
        query="park", searched=True, matches=[row]
    )
    assert "Ji-hwan Park" in html
    assert "KBO" in html
    assert "0.395" in html


def test_player_template_no_match_state() -> None:
    html = leaderboard_app.TEMPLATES.get_template("player.html").render(
        query="zzzz", searched=True, matches=[]
    )
    assert "No matches" in html


def test_player_template_initial_state_prompts_for_query() -> None:
    html = leaderboard_app.TEMPLATES.get_template("player.html").render(
        query="", searched=False, matches=[]
    )
    assert "Search" in html


def test_player_route_wires_query_to_matches(monkeypatch) -> None:
    captured: dict = {}

    def fake_template_response(request, template: str, context: dict):
        captured.update(template=template, context=context)

    monkeypatch.setattr(leaderboard_app, "_find_player", lambda q: [{"role": "batter"}])
    monkeypatch.setattr(leaderboard_app.TEMPLATES, "TemplateResponse", fake_template_response)
    leaderboard_app.player_lookup(None, q="kim")
    assert captured["template"] == "player.html"
    assert captured["context"]["matches"] == [{"role": "batter"}]
    assert captured["context"]["searched"] is True


def test_index_links_to_lookup_and_csv_export() -> None:
    html = leaderboard_app.TEMPLATES.get_template("index.html").render(
        season=2025, data_mode="real", batters=[], pitchers=[], error=None
    )
    assert 'href="/player"' in html
    assert 'href="/api/leaderboard.csv"' in html


# ---- WP8: wRC+/FIP-, 20-80 grade, WAR and $ with bands -------------------


def _graded_batter(**overrides) -> dict:
    row = {
        "player_name": "Ji-hwan Park",
        "from_league": "KBO",
        "position": "RF",
        "woba": 0.395,
        "woba_low": 0.33,
        "woba_high": 0.46,
        "wrc_plus": 131.4,
        "wrc_plus_low": 110.2,
        "wrc_plus_high": 152.9,
        "grade": 65.0,
        "grade_low": 55.0,
        "grade_high": 75.0,
        "war": 3.21,
        "war_low": 1.9,
        "war_high": 4.6,
        "dollars": 25_700_000.0,
        "dollars_low": 15_200_000.0,
        "dollars_high": 36_800_000.0,
    }
    row.update(overrides)
    return row


def _graded_pitcher(**overrides) -> dict:
    row = {
        "player_name": "Won-jun Kim",
        "from_league": "KBO",
        "fip": 3.42,
        "fip_low": 2.9,
        "fip_high": 4.1,
        "fip_minus": 88.1,
        "fip_minus_low": 75.0,
        "fip_minus_high": 102.6,
        "grade": 55.0,
        "grade_low": 45.0,
        "grade_high": 60.0,
        "war": 2.4,
        "war_low": 1.1,
        "war_high": 3.5,
        "dollars": 19_200_000.0,
        "dollars_low": 8_800_000.0,
        "dollars_high": 28_000_000.0,
    }
    row.update(overrides)
    return row


def _render_index(batters: list[dict], pitchers: list[dict] | None = None) -> str:
    return leaderboard_app.TEMPLATES.get_template("index.html").render(
        season=2025, data_mode="real", batters=batters, pitchers=pitchers or [], error=None
    )


def test_index_batter_columns_show_wrc_plus_grade_war_and_dollars_with_bands() -> None:
    html = _render_index([_graded_batter()])

    assert "wRC+" in html
    assert "20-80" in html
    assert "WAR" in html
    assert "131" in html and "110–153" in html
    assert ">65<" in html
    assert "3.2" in html and "1.9–4.6" in html
    assert "$25.7M" in html and "$15.2M–$36.8M" in html
    # Existing wOBA column and band stay.
    assert "0.395" in html and "0.330–0.460" in html


def test_index_pitcher_columns_show_fip_minus_instead_of_wrc_plus() -> None:
    html = _render_index([], [_graded_pitcher()])

    assert "FIP-" in html
    assert "88" in html and "75–103" in html
    assert "$19.2M" in html
    assert "3.42" in html  # existing FIP column stays


def test_index_renders_n_a_for_null_grade_fields_without_crashing() -> None:
    null_batter = _graded_batter(
        wrc_plus=None, wrc_plus_low=None, wrc_plus_high=None,
        grade=None, grade_low=None, grade_high=None,
        war=None, war_low=None, war_high=None,
        dollars=None, dollars_low=None, dollars_high=None,
    )
    html = _render_index([null_batter], [_graded_pitcher(fip_minus=None, fip_minus_low=None, fip_minus_high=None, grade=None, grade_low=None, grade_high=None, war=None, war_low=None, war_high=None, dollars=None, dollars_low=None, dollars_high=None)])

    assert html.count("n/a") >= 8
    assert "None" not in html


def test_index_renders_rows_that_predate_grade_fields() -> None:
    legacy = {"player_name": "Old Row", "from_league": "KBO", "woba": 0.4, "woba_low": 0.35, "woba_high": 0.45}
    html = _render_index([legacy])

    assert "Old Row" in html
    assert "n/a" in html


def test_player_template_shows_grades_and_n_a_for_null_fields() -> None:
    null_batter = _graded_batter(
        wrc_plus=None, wrc_plus_low=None, wrc_plus_high=None, grade=None, grade_low=None,
        grade_high=None, war=None, war_low=None, war_high=None, dollars=None,
        dollars_low=None, dollars_high=None,
    )
    html = leaderboard_app.TEMPLATES.get_template("player.html").render(
        query="park", searched=True,
        matches=[{"role": "batter", **null_batter}, {"role": "pitcher", **_graded_pitcher()}],
    )

    assert "n/a" in html
    assert "wRC+" in html
    assert "FIP-" in html and "88" in html
    assert "None" not in html


def test_format_helpers_render_n_a_for_none_and_nan() -> None:
    fmt_num = leaderboard_app._fmt_num
    fmt_band = leaderboard_app._fmt_band
    fmt_money = leaderboard_app._fmt_money
    fmt_money_band = leaderboard_app._fmt_money_band

    assert fmt_num(None, 0) == "n/a"
    assert fmt_num(float("nan"), 1) == "n/a"
    assert fmt_num(3.14159, 1) == "3.1"
    assert fmt_band(None, 2.0, 0) == "n/a"
    assert fmt_band(1.0, 2.0, 1) == "1.0–2.0"
    assert fmt_money(None) == "n/a"
    assert fmt_money(2_500_000) == "$2.5M"
    assert fmt_money_band(None, 1_000_000) == "n/a"
    assert fmt_money_band(1_000_000, 2_000_000) == "$1.0M–$2.0M"


def test_leaderboard_command_builds_grades_with_constants_and_nulls_without(tmp_path, monkeypatch) -> None:
    """The CLI always passes constants: missing ones must give nulls, present ones values."""
    from rosetta import cli
    from rosetta.models.factors import LeagueFactorModel

    factors = tmp_path / "factors.json"
    LeagueFactorModel(links={}).save(factors)
    batter = {
        "player_id": "b1", "player_name": "Bat Ter", "season": 2024, "league": "NPB",
        "role": "batter", "age": 27, "pa": 500, "woba": 0.360, "bb_pct": 0.09,
        "k_pct": 0.18, "iso": 0.18, "babip": 0.31, "hr_pct": 0.04, "position": "SS",
    }
    pitcher = {
        "player_id": "p1", "player_name": "Pit Cher", "season": 2024, "league": "NPB",
        "role": "pitcher", "age": 27, "ip": 150, "g": 25, "gs": 25, "k_pct": 0.25,
        "bb_pct": 0.07, "hr_fb": 0.10, "era": 3.2, "fip": 3.4,
    }
    frames = {"batter": pd.DataFrame([batter]), "pitcher": pd.DataFrame([pitcher])}
    monkeypatch.setattr(
        cli, "load_all_seasons",
        lambda *, role, snapshot_dir=None, leagues=None: frames[role],
    )

    def run(out: Path) -> dict:
        cli.leaderboard_cmd(
            snapshots=tmp_path / "snap", factors=factors, out=out, season=2024,
            min_pa=0, min_ip=0, include_synthetic=False,
        )
        return json.loads(out.read_text())

    # No reference constants anywhere: every grade field serializes as null.
    empty = run(tmp_path / "empty.json")
    b = empty["boards"]["batter"][0]
    p = empty["boards"]["pitcher"][0]
    for key in ("wrc_plus", "wrc_plus_low", "wrc_plus_high", "grade", "war", "dollars"):
        assert b[key] is None
    for key in ("fip_minus", "fip_minus_low", "fip_minus_high", "grade", "war", "dollars"):
        assert p[key] is None
    assert b["woba"] == pytest.approx(0.360)

    # With fixture constants, the same rows carry values and ordered bands.
    from rosetta.models.grades import MlbConstants

    def fixture_constants(season, snapshot_dir=None, reference_path=None) -> MlbConstants:
        c = MlbConstants(
            season=season, woba_scale=1.25, lg_woba=0.320, lg_r_pa=0.11, runs_per_win=10.0,
            lg_fip=4.00, fip_constant=3.0, dollars_per_war=8.0, pitcher_repl_r9=0.5,
            ref_woba=[0.290, 0.310, 0.320, 0.330, 0.350], ref_fip=[3.2, 3.8, 4.0, 4.2, 4.8],
        )
        return c

    monkeypatch.setattr(cli, "load_constants", fixture_constants)
    graded = run(tmp_path / "graded.json")
    b = graded["boards"]["batter"][0]
    p = graded["boards"]["pitcher"][0]
    assert b["wrc_plus"] is not None and b["wrc_plus_low"] <= b["wrc_plus"] <= b["wrc_plus_high"]
    assert b["grade"] is not None and b["war"] is not None and b["dollars"] is not None
    assert p["fip_minus"] is not None and p["fip_minus_low"] <= p["fip_minus"] <= p["fip_minus_high"]
    assert p["grade"] is not None and p["war"] is not None
