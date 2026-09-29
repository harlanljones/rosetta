"""CPBL ingest via cpbl.com.tw record tables (no auth, HTML).

Fetches season batting/pitching leader tables from ``/stats/recordall``.
Explicit fetch step only; never CI-live.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from rosetta.ingest.kbo import FIP_CONSTANT, WOBA_W
from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

BASE = "https://cpbl.com.tw"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}


def recordall_url(season: int, kind: str) -> str:
    """kind: 01 (batting leaderboard) or 02 (pitching leaderboard)."""
    query = urllib.parse.urlencode({"kind": kind, "year": season})
    return f"{BASE}/stats/recordall?{query}"


def _clean(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).strip()


def _player_id_from_cell(cell: str) -> tuple[str, str]:
    """Return (player_id, display_name) from the rank+team+name cell."""
    text = _clean(cell)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "", text).lower() or "unknown"
    return f"cpbl-name-{slug}", text


def parse_recordall_table(html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for block in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        if "<td" not in block:
            continue
        cells = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", block, re.S)]
        if len(cells) < 10:
            continue
        pid, name = _player_id_from_cell(cells[0])
        rows.append({"player_id": pid, "player_name": name, "cells": cells})
    return rows


class CPBLSession:
    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay

    def get(self, url: str) -> str:
        time.sleep(self.delay)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")


def normalize_batting(rows: list[dict[str, str]], season: int) -> pd.DataFrame:
    out: list[dict] = []
    for row in rows:
        cells = row["cells"]
        try:
            avg = float(cells[1])
            g = float(cells[2])
            pa = float(cells[3])
            ab = float(cells[4])
            bb = float(cells[6])
            so = float(cells[7])
            h = float(cells[8])
            doubles = float(cells[9])
            triples = float(cells[10])
            hr = float(cells[11])
            obp = float(cells[23])
            slg = float(cells[24])
        except (IndexError, ValueError):
            continue
        if pa <= 0:
            continue
        singles = h - doubles - triples - hr
        bb_pct = bb / pa
        k_pct = so / pa
        iso = slg - avg
        bip = ab - so - hr
        babip = ((h - hr) / bip) if bip > 0 else 0.0
        woba = (
            WOBA_W["ubb"] * bb
            + WOBA_W["hbp"] * 0
            + WOBA_W["1b"] * singles
            + WOBA_W["2b"] * doubles
            + WOBA_W["3b"] * triples
            + WOBA_W["hr"] * hr
        ) / pa
        out.append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "season": season,
                "league": "CPBL",
                "team": "",
                "role": "batter",
                "age": 27.0,
                "pa": pa,
                "g": g,
                "bb_pct": bb_pct,
                "k_pct": k_pct,
                "iso": iso,
                "babip": babip,
                "hr_pct": hr / pa,
                "avg": avg,
                "obp": obp,
                "slg": slg,
                "woba": woba,
                "home_pa": pa * 0.5,
                "road_pa": pa * 0.5,
                "home_woba": woba,
                "road_woba": woba,
                "park_id": "",
                "source": "cpbl-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(out, columns=HITTER_COLUMNS)


def normalize_pitching(rows: list[dict[str, str]], season: int) -> pd.DataFrame:
    out: list[dict] = []
    for row in rows:
        cells = row["cells"]
        try:
            era = float(cells[1])
            g = float(cells[2])
            ip = float(cells[6])
            so = float(cells[11])
            bb = float(cells[9])
            hr = float(cells[8])
            hits = float(cells[7])
        except (IndexError, ValueError):
            continue
        if ip <= 0:
            continue
        tbf = max(ip * 4.3, 1.0)
        k_pct = so / tbf
        bb_pct = bb / tbf
        hr_fb = (hr / hits) if hits > 0 else 0.12
        fip = (13 * hr + 3 * bb - 2 * so) / ip + FIP_CONSTANT
        out.append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "season": season,
                "league": "CPBL",
                "team": "",
                "role": "pitcher",
                "age": 27.0,
                "ip": ip,
                "g": g,
                "gs": 0.0,
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "hr_fb": hr_fb,
                "era": era,
                "fip": fip,
                "home_pa": tbf * 0.5,
                "road_pa": tbf * 0.5,
                "home_era": era,
                "road_era": era,
                "park_id": "",
                "source": "cpbl-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(out, columns=PITCHER_COLUMNS)


def fetch_cpbl_season(
    season: int, *, session: CPBLSession | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sess = session or CPBLSession()
    bat_html = sess.get(recordall_url(season, "01"))
    pit_html = sess.get(recordall_url(season, "02"))
    bat = normalize_batting(parse_recordall_table(bat_html), season)
    pit = normalize_pitching(parse_recordall_table(pit_html), season)
    if not pit.empty and pit["era"].max() < 1.0:
        # Site occasionally serves the batting table for both kinds; skip bad parses.
        pit = pd.DataFrame(columns=PITCHER_COLUMNS)
    return bat, pit


def write_cpbl_snapshots(
    seasons: list[int],
    *,
    out_dir: Path | None = None,
    append: bool = True,
    delay: float = 1.0,
) -> Path:
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    sess = CPBLSession(delay=delay)
    bats: list[pd.DataFrame] = []
    pits: list[pd.DataFrame] = []
    for season in seasons:
        b, p = fetch_cpbl_season(season, session=sess)
        bats.append(b)
        pits.append(p)
    bat_df = pd.concat(bats, ignore_index=True) if bats else pd.DataFrame(columns=HITTER_COLUMNS)
    pit_df = pd.concat(pits, ignore_index=True) if pits else pd.DataFrame(columns=PITCHER_COLUMNS)
    for fname, frame in (("cpbl_batters.csv", bat_df), ("cpbl_pitchers.csv", pit_df)):
        path = root / fname
        if append and path.exists() and not frame.empty:
            old = pd.read_csv(path)
            written_keys = {(row["league"], row["season"]) for row in frame.to_dict("records")}
            is_real = old["is_synthetic"].astype(str) != "True"
            rewritten = old.apply(lambda r, wk=written_keys: (r["league"], r["season"]) in wk, axis=1)
            old = old[~(is_real & rewritten)]
            frame = pd.concat([old, frame], ignore_index=True).drop_duplicates(
                ["player_id", "season", "league"], keep="last"
            )
        if not frame.empty:
            frame.to_csv(path, index=False)
    return root
