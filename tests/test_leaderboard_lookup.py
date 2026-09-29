"""P2: leaderboard player lookup and CSV export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

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
