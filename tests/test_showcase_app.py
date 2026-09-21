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
    response = leaderboard_app.showcase(request)
    assert response.template.name == "showcase.html"
