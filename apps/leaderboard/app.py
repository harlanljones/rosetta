"""Small FastAPI + Jinja leaderboard for MLE translations."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "outputs" / "leaderboard.json"
SHOWCASE_BUNDLE = ROOT / "data" / "outputs" / "demo"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

NA = "n/a"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def _fmt_num(value: object, digits: int) -> str:
    """Fixed-precision number, or "n/a" when the grade or WAR input is unavailable."""
    if _is_missing(value):
        return NA
    return f"{float(value):.{digits}f}"  # type: ignore[arg-type]


def _fmt_band(low: object, high: object, digits: int) -> str:
    if _is_missing(low) or _is_missing(high):
        return NA
    return f"{_fmt_num(low, digits)}–{_fmt_num(high, digits)}"


def _fmt_money(value: object) -> str:
    """Dollar value in millions; "n/a" until a $/WAR constant exists."""
    if _is_missing(value):
        return NA
    return f"${float(value) / 1_000_000:,.1f}M"  # type: ignore[arg-type]


def _fmt_money_band(low: object, high: object) -> str:
    if _is_missing(low) or _is_missing(high):
        return NA
    return f"{_fmt_money(low)}–{_fmt_money(high)}"


TEMPLATES.env.globals.update(
    fmt_num=_fmt_num,
    fmt_band=_fmt_band,
    fmt_money=_fmt_money,
    fmt_money_band=_fmt_money_band,
)

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


@app.get("/api/leaderboard.csv")
def api_leaderboard_csv() -> Response:
    """Full-board CSV export: every row from both boards, tagged with its role."""
    data = _load()
    buffer = io.StringIO()
    rows = data.get("boards", {})
    all_rows = [row for role in ("batter", "pitcher") for row in rows.get(role, [])]
    columns: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    fieldnames = ["role", *columns]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for role in ("batter", "pitcher"):
        for row in rows.get(role, []):
            writer.writerow({"role": role, **row})
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leaderboard.csv"},
    )


def _find_player(query: str) -> list[dict]:
    """Case-insensitive substring search over both boards; rows tagged with role."""
    needle = query.strip().lower()
    if not needle:
        return []
    rows = _load().get("boards", {})
    matches: list[dict] = []
    for role in ("batter", "pitcher"):
        for row in rows.get(role, []):
            if needle in str(row.get("player_name", "")).lower():
                matches.append({"role": role, **row})
    return matches


@app.get("/player", response_class=HTMLResponse)
def player_lookup(request: Request, q: str = "") -> HTMLResponse:
    """Per-player lookup view across both boards."""
    query = q.strip()
    matches = _find_player(query) if query else []
    return TEMPLATES.TemplateResponse(
        request,
        "player.html",
        {"query": query, "matches": matches, "searched": bool(query)},
    )


def _load_showcase() -> dict:
    """Load the three-file static showcase bundle without silently using synthetic data."""
    payload: dict = {}
    for name in ("meta", "leaderboard", "backtest"):
        path = SHOWCASE_BUNDLE / f"{name}.json"
        if not path.exists():
            return {
                "error": "Showcase data is not exported yet. Run `python scripts/export_demo.py`."
            }
        payload[name] = json.loads(path.read_text())
    if any(
        payload[name].get("data_mode") != "real" for name in ("meta", "leaderboard", "backtest")
    ):
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
