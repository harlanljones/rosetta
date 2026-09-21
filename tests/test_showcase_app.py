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
