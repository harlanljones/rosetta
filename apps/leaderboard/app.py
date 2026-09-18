"""Small FastAPI + Jinja leaderboard for MLE translations."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "outputs" / "leaderboard.json"
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


@app.get("/health")
def health() -> dict:
    return {"ok": True, "leaderboard_exists": OUTPUT.exists()}
