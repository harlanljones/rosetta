"""Small FastAPI + Jinja leaderboard for MLE translations."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "outputs" / "leaderboard.json"
SHOWCASE_BUNDLE = ROOT / "data" / "outputs" / "demo"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="Rosetta MLE Leaderboard", version="0.1.0")


def _load() -> dict:
    if not OUTPUT.exists():
        return {
            "season": None,
            "boards": {"batter": [], "pitcher": []},
            "error": "Run `make leaderboard` first",
        }
    return json.loads(OUTPUT.read_text())


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    data = _load()
    batters = data.get("boards", {}).get("batter", [])[:25]
    pitchers = data.get("boards", {}).get("pitcher", [])[:25]
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "season": data.get("season"),
            "data_mode": data.get("data_mode", "unknown"),
            "batters": batters,
            "pitchers": pitchers,
            "error": data.get("error"),
        },
    )


@app.get("/api/leaderboard")
def api_leaderboard() -> JSONResponse:
    return JSONResponse(_load())


def _load_showcase() -> dict:
    """Load the three-file static showcase bundle without silently using synthetic data."""
    payload: dict = {}
    for name in ("meta", "leaderboard", "backtest"):
        path = SHOWCASE_BUNDLE / f"{name}.json"
        if not path.exists():
            return {"error": "Showcase data is not exported yet. Run `python scripts/export_demo.py`."}
        payload[name] = json.loads(path.read_text())
    if payload["meta"].get("data_mode") != "real" or payload["leaderboard"].get("data_mode") != "real":
        return {"error": "Showcase data is not real-data-only; refusing to render numbers."}
    return payload


@app.get("/showcase", response_class=HTMLResponse)
def showcase(request: Request) -> HTMLResponse:
    """Render the portfolio proof page from meta, leaderboard, and backtest JSON."""
    data = _load_showcase()
    return TEMPLATES.TemplateResponse(request, "showcase.html", {"data": data})


def _load_analysis() -> dict:
    """Load the player-error report and keep synthetic bundles out of the UI."""
    path = SHOWCASE_BUNDLE / "analysis.json"
    if not path.exists():
        return {"error": "Analysis data is not exported yet. Run `python scripts/export_demo.py`."}
    payload = json.loads(path.read_text())
    if payload.get("data_mode") != "real":
        return {"error": "Analysis data is not real-data-only; refusing to render numbers."}
    return payload


@app.get("/analysis", response_class=HTMLResponse)
def analysis(request: Request) -> HTMLResponse:
    """Render the human-readable player-error and calibration report."""
    return TEMPLATES.TemplateResponse(request, "analysis.html", {"data": _load_analysis()})


@app.get("/health")
def health() -> dict:
    return {"ok": True, "leaderboard_exists": OUTPUT.exists()}
