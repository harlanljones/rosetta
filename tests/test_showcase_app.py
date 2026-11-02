from __future__ import annotations

import json
from pathlib import Path

from starlette.requests import Request

from apps.leaderboard import app as leaderboard_app


def _write_bundle(root: Path, *, data_mode: str = "real") -> None:
    root.mkdir()
    for name, payload in {
        "meta": {"data_mode": data_mode},
        "leaderboard": {"data_mode": data_mode},
        "backtest": {},
    }.items():
        (root / f"{name}.json").write_text(json.dumps(payload))


def test_showcase_requires_exported_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(leaderboard_app, "SHOWCASE_BUNDLE", tmp_path / "missing")
    result = leaderboard_app._load_showcase()
    assert result["error"].startswith("Showcase data is not exported")


def test_showcase_refuses_non_real_bundle(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "demo"
    _write_bundle(bundle, data_mode="synthetic")
    monkeypatch.setattr(leaderboard_app, "SHOWCASE_BUNDLE", bundle)
    result = leaderboard_app._load_showcase()
    assert result == {"error": "Showcase data is not real-data-only; refusing to render numbers."}


def test_showcase_refuses_synthetic_backtest_even_with_real_board(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "demo"
    _write_bundle(bundle, data_mode="real")
    (bundle / "backtest.json").write_text(json.dumps({"data_mode": "synthetic"}))
    monkeypatch.setattr(leaderboard_app, "SHOWCASE_BUNDLE", bundle)

    result = leaderboard_app._load_showcase()

    assert result == {"error": "Showcase data is not real-data-only; refusing to render numbers."}


def test_showcase_error_state_renders() -> None:
    html = leaderboard_app.TEMPLATES.get_template("showcase.html").render(
        data={"error": "Showcase data is not exported yet."}
    )
    assert "Showcase data is not exported yet." in html


def test_showcase_route_uses_showcase_template(monkeypatch) -> None:
    monkeypatch.setattr(
        leaderboard_app,
        "_load_showcase",
        lambda: {"error": "Showcase data is not exported yet."},
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/showcase",
            "raw_path": b"/showcase",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "http_version": "1.1",
            "client": ("test", 1),
            "server": ("test", 80),
        }
    )
    captured: dict = {}

    def fake_template_response(request, template: str, context: dict):
        captured.update(request=request, template=template, context=context)
        return captured

    monkeypatch.setattr(leaderboard_app.TEMPLATES, "TemplateResponse", fake_template_response)
    response = leaderboard_app.showcase(request)
    assert response["template"] == "showcase.html"
    assert response["context"]["data"]["error"].startswith("Showcase data")


def test_analysis_requires_real_export(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "demo"
    bundle.mkdir()
    (bundle / "analysis.json").write_text(json.dumps({"data_mode": "synthetic"}))
    monkeypatch.setattr(leaderboard_app, "SHOWCASE_BUNDLE", bundle)
    result = leaderboard_app._load_analysis()
    assert result == {"error": "Analysis data is not real-data-only; refusing to render numbers."}


def test_analysis_template_has_human_readable_guardrails() -> None:
    html = leaderboard_app.TEMPLATES.get_template("analysis.html").render(
        data={"error": "Analysis data is not exported yet."}
    )
    assert "Analysis data is not exported yet." in html


def test_showcase_does_not_duplicate_the_full_scoreboard() -> None:
    metric = {
        "role": "batter", "stat": "woba", "n": 1,
        "mae": 0.02, "baseline_mae": 0.03, "coverage_80": 0.8,
    }
    payload = {
        "meta": {"backtest_target_season": 2025, "generated_at": "2026-09-21"},
        "leaderboard": {"data_mode": "real", "boards": {"batter": [], "pitcher": []}},
        "backtest": {
            "data_mode": "real",
            "validation": {"model_wins": 1, "metric_count": 1},
            "target_season_2025": {"metrics": [metric]},
            "rolling_2022_2025": {
                "target_seasons": [], "summary_by_season": [],
            },
        },
    }

    html = leaderboard_app.TEMPLATES.get_template("showcase.html").render(data=payload)

    assert "Top batters" not in html
    assert "Top pitchers" not in html
    assert "/analysis" in html


def _showcase_payload(posting_class: list[dict] | None = None, **backtest_extra) -> dict:
    metric = {
        "role": "batter", "stat": "woba", "n": 1,
        "mae": 0.02, "baseline_mae": 0.03, "coverage_80": 0.8,
    }
    leaderboard: dict = {"data_mode": "real", "boards": {"batter": [], "pitcher": []}}
    if posting_class is not None:
        leaderboard["posting_class"] = posting_class
    return {
        "meta": {
            "backtest_target_season": 2025,
            "generated_at": "2026-09-21T00:00:00+00:00",
            "rosetta_git_commit": "abc1234",
            "estimator": "ratio_of_means",
            "calibration": "none",
        },
        "leaderboard": leaderboard,
        "backtest": {
            "data_mode": "real",
            "validation": {"model_wins": 1, "metric_count": 1},
            "target_season_2025": {"metrics": [metric]},
            "rolling_2022_2025": {"target_seasons": [], "summary_by_season": []},
            **backtest_extra,
        },
    }


def _posting_row(**overrides) -> dict:
    row = {
        "player_name": "Teruaki Sato",
        "league": "NPB",
        "role": "batter",
        "note": "Posting class 2026-27; wRC+ grade pending validation",
        "woba": 0.414,
        "woba_low": 0.36,
        "woba_high": 0.47,
        "wrc_plus": 131.0,
        "wrc_plus_low": 110.0,
        "wrc_plus_high": 150.0,
        "fip_minus": None,
        "fip_minus_low": None,
        "fip_minus_high": None,
        "grade": 60.0,
        "grade_low": 50.0,
        "grade_high": 70.0,
        "war": 3.2,
        "war_low": 1.9,
        "war_high": 4.5,
        "dollars": 25_600_000.0,
        "dollars_low": 15_200_000.0,
        "dollars_high": 36_000_000.0,
    }
    row.update(overrides)
    return row


def _render_showcase(payload: dict) -> str:
    return leaderboard_app.TEMPLATES.get_template("showcase.html").render(data=payload)


def test_posting_class_panel_is_shown_with_rows() -> None:
    html = _render_showcase(_showcase_payload([_posting_row()]))

    assert "2026-27 posting class" in html
    assert "Teruaki Sato" in html
    assert "Posting class 2026-27; wRC+ grade pending validation" in html
    assert "110–150" in html
    assert "$25.6M" in html


def test_posting_class_panel_keeps_file_order() -> None:
    rows = [_posting_row(player_name="Second Player"), _posting_row(player_name="First Listed")]
    html = _render_showcase(_showcase_payload(rows))

    assert html.index("Second Player") < html.index("First Listed")


def test_posting_class_panel_is_hidden_when_empty() -> None:
    html_empty_list = _render_showcase(_showcase_payload([]))
    html_absent_key = _render_showcase(_showcase_payload(None))

    assert "2026-27 posting class" not in html_empty_list
    assert "2026-27 posting class" not in html_absent_key


def test_posting_class_panel_renders_n_a_for_null_fields() -> None:
    row = _posting_row(
        wrc_plus=None, wrc_plus_low=None, wrc_plus_high=None, grade=None, grade_low=None,
        grade_high=None, war=None, war_low=None, war_high=None, dollars=None,
        dollars_low=None, dollars_high=None, woba=None, woba_low=None, woba_high=None,
    )
    html = _render_showcase(_showcase_payload([row]))

    assert "2026-27 posting class" in html
    assert "Teruaki Sato" in html
    assert "n/a" in html
    assert "None" not in html


def test_what_it_cant_do_says_international_grades_pending_validation() -> None:
    html = _render_showcase(_showcase_payload())

    assert "NPB" in html and "KBO" in html
    assert "pending international validation" in html
    assert "fixed playing time" in html
    assert "fielding" in html


def test_what_it_cant_do_shows_international_validation_once_in_bundle() -> None:
    validation = [
        {
            "league": "NPB", "role": "batter", "stat": "woba", "n": 40,
            "mae": 0.021, "baseline_mae": 0.026, "coverage_80": 0.78,
        }
    ]
    html = _render_showcase(_showcase_payload(international_validation=validation))

    assert "pending international validation" not in html
    assert "NPB" in html
    assert "0.021" in html and "0.026" in html
