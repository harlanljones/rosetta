"""Statcast pitch-level ingest via Baseball Savant CSVs (no auth).

Two jobs in one client (verified live 2026-09-18):

1. **Minors validation + Phase-2 features (#4).** AAA pitch data since 2023
   aggregated to season rate rows (real K%/BB%/HR) plus pitch-mix features
   (usage, velo, spin by pitch type) for translation validation.
2. **Empirical park factors (#3).** Home/road wOBA per park from pitch-level
   events — no BR scraping needed.

Endpoint (reverse-engineered from the site JS bundle):
  GET /statcast_search/csv?all=true&type=details
      &game_date_gt=YYYY-MM-DD&game_date_lt=YYYY-MM-DD&minors=1
  MLB baseline: same URL with minors=0 (or omitted).

Volume is large (~7k pitches/AAA-day, ~1M/season): shard by day into
data/raw/savant/ (gitignored), snapshot only aggregates. Explicit fetch
steps; never CI-live.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rosetta.paths import DATA_DIR, SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

BASE = "https://baseballsavant.mlb.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}
WOBA_W = {"ubb": 0.69, "hbp": 0.72, "1b": 0.89, "2b": 1.27, "3b": 1.62, "hr": 2.10}
FIP_CONSTANT = 3.10

HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk"}
HBP_EVENTS = {"hit_by_pitch"}
SF_EVENTS = {"sac_fly"}
SH_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
SO_EVENTS = {"strikeout", "strikeout_double_play"}
NON_AB_EVENTS = BB_EVENTS | HBP_EVENTS | SF_EVENTS | SH_EVENTS | {
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "other_out",
}


def download_day(
    day: date | str, *, minors: bool = True, out_dir: Path | None = None, timeout: int = 120
) -> Path:
    """Download one day of pitch-level CSV (cached; returns path)."""
    day = str(day)
    root = Path(out_dir) if out_dir else DATA_DIR / "raw" / "savant"
    root.mkdir(parents=True, exist_ok=True)
    tag = "minors" if minors else "mlb"
    path = root / f"{tag}_{day}.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    params = {
        "all": "true",
        "type": "details",
        "game_date_gt": day,
        "game_date_lt": day,
        "minors": "1" if minors else "0",
    }
    url = f"{BASE}/statcast_search/csv?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        path.write_bytes(resp.read())
    return path


def download_range(
    start: date | str,
    end: date | str,
    *,
    minors: bool = True,
    out_dir: Path | None = None,
    delay: float = 2.0,
) -> list[Path]:
    """Download every day in [start, end]; polite delays between requests."""
    s = date.fromisoformat(str(start))
    e = date.fromisoformat(str(end))
    paths: list[Path] = []
    day = s
    while day <= e:
        paths.append(download_day(day, minors=minors, out_dir=out_dir))
        day += timedelta(days=1)
        if day <= e:
            time.sleep(delay)
    return paths


def load_pitches(paths: list[Path | str]) -> pd.DataFrame:
    """Read cached pitch CSVs into one frame (skips empty/missing files)."""
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pa_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed plate appearances (one row per PA)."""
    pa = df[df["events"].notna()].copy()
    return pa


def _is_home_game(pa: pd.DataFrame) -> pd.Series:
    """True where the batting team is the home team."""
    top = pa["inning_topbot"].astype(str).str.lower().str.startswith("top")
    return ~top.fillna(False)


def _woba_row(ev: str) -> tuple[float, float, float, float, float, float]:
    """(ubb, hbp, 1b, 2b, 3b, hr) indicators for one event."""
    return (
        1.0 if ev in BB_EVENTS else 0.0,
        1.0 if ev in HBP_EVENTS else 0.0,
        1.0 if ev == "single" else 0.0,
        1.0 if ev == "double" else 0.0,
        1.0 if ev == "triple" else 0.0,
        1.0 if ev == "home_run" else 0.0,
    )


def aggregate_batters(
    pitches: pd.DataFrame, season: int, league_label: str
) -> pd.DataFrame:
    """Pitch-level events → season hitter rows with REAL home/road splits."""
    pa = _pa_events(pitches)
    if pa.empty:
        return pd.DataFrame(columns=HITTER_COLUMNS)
    pa = pa.copy()
    pa["is_home"] = _is_home_game(pa)
    # Batter's own team: away_team on top, home_team on bottom.
    pa["bat_team"] = pa["away_team"].where(
        pa["inning_topbot"].astype(str).str.lower().str.startswith("top"),
        pa["home_team"],
    )
    pa["is_ab"] = ~pa["events"].isin(NON_AB_EVENTS)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS)
    pa["is_so"] = pa["events"].isin(SO_EVENTS)
    pa["is_bb"] = pa["events"].isin(BB_EVENTS)
    pa["is_hbp"] = pa["events"].isin(HBP_EVENTS)
    pa["is_hr"] = pa["events"] == "home_run"
    pa["is_2b"] = pa["events"] == "double"
    pa["is_3b"] = pa["events"] == "triple"
    pa["woba_num"] = (
        WOBA_W["ubb"] * (pa["is_bb"].astype(float))
        + WOBA_W["hbp"] * pa["is_hbp"].astype(float)
        + WOBA_W["1b"] * (pa["events"] == "single").astype(float)
        + WOBA_W["2b"] * pa["is_2b"].astype(float)
        + WOBA_W["3b"] * pa["is_3b"].astype(float)
        + WOBA_W["hr"] * pa["is_hr"].astype(float)
    )
    rows = []
    for (bid, name), g in pa.groupby(["batter", "player_name"], dropna=False):
        n = len(g)
        ab = g["is_ab"].sum()
        h = g["is_hit"].sum()
        so = g["is_so"].sum()
        bb = g["is_bb"].sum()
        hr = g["is_hr"].sum()
        home = g[g["is_home"]]
        road = g[~g["is_home"]]
        woba = g["woba_num"].sum() / n
        rows.append(
            {
                "player_id": f"mlbam-{int(bid) if pd.notna(bid) else ''}",
                "player_name": name if pd.notna(name) else "",
                "season": season,
                "league": league_label,
                "team": str(g["bat_team"].mode().iat[0]) if g["bat_team"].notna().any() else "",
                "role": "batter",
                "age": float(g["age_bat"].dropna().median()) if g["age_bat"].notna().any() else 27.0,
                "pa": float(n),
                "g": float(g["game_pk"].nunique()),
                "bb_pct": float(bb / n),
                "k_pct": float(so / n),
                "iso": float((g["is_2b"] + 2 * g["is_3b"] + 3 * g["is_hr"]).sum() / ab) if ab else 0.0,
                "babip": float((h - hr) / (ab - so - hr)) if (ab - so - hr) else 0.0,
                "hr_pct": float(hr / n),
                "avg": float(h / ab) if ab else 0.0,
                "obp": float((h + bb + g["is_hbp"].sum()) / n),
                "slg": float(
                    (h + g["is_2b"].sum() + 2 * g["is_3b"].sum() + 3 * hr) / ab
                ) if ab else 0.0,
                "woba": float(woba),
                "home_pa": float(len(home)),
                "road_pa": float(len(road)),
                "home_woba": float(home["woba_num"].sum() / len(home)) if len(home) else float(woba),
                "road_woba": float(road["woba_num"].sum() / len(road)) if len(road) else float(woba),
                "park_id": "",
                "source": "savant",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=HITTER_COLUMNS)


def aggregate_pitchers(
    pitches: pd.DataFrame, season: int, league_label: str
) -> pd.DataFrame:
    """Pitch-level events → season pitcher rows.

    K%/BB%/HR carry real signal. IP/ERA/FIP are NOT recoverable from pitch
    rows (no reliable runs/outs accounting) and stay null/zero — use the
    Stats API client for counting-stat pitchers.
    """
    pa = _pa_events(pitches)
    if pa.empty:
        return pd.DataFrame(columns=PITCHER_COLUMNS)
    pa = pa.copy()
    pa["is_so"] = pa["events"].isin(SO_EVENTS)
    pa["is_bb"] = pa["events"].isin(BB_EVENTS | HBP_EVENTS)
    pa["is_hr"] = pa["events"] == "home_run"
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS)
    rows = []
    for (pid, name), g in pa.groupby(["pitcher", "player_name"], dropna=False):
        bf = len(g)
        h = g["is_hit"].sum()
        rows.append(
            {
                "player_id": f"mlbam-{int(pid) if pd.notna(pid) else ''}",
                "player_name": name if pd.notna(name) else "",
                "season": season,
                "league": league_label,
                "team": "",
                "role": "pitcher",
                "age": float(g["age_pit"].dropna().median()) if g["age_pit"].notna().any() else 27.0,
                "ip": 0.0,
                "g": float(g["game_pk"].nunique()),
                "gs": 0.0,
                "k_pct": float(g["is_so"].sum() / bf),
                "bb_pct": float(g["is_bb"].sum() / bf),
                "hr_fb": float(g["is_hr"].sum() / h) if h else 0.12,
                "era": None,
                "fip": None,
                "home_pa": float(bf * 0.5),
                "road_pa": float(bf * 0.5),
                "home_era": None,
                "road_era": None,
                "park_id": "",
                "source": "savant",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=PITCHER_COLUMNS)


def pitch_mix_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher pitch-mix features (Phase-2 validation, not schema rows).

    Columns: pitcher id/name, n_pitches, arsenal (type:share), avg velo +
    spin per type, avg exit velo against, xwOBA-against.
    """
    if pitches.empty:
        return pd.DataFrame()
    rows = []
    for (pid, name), g in pitches.groupby(["pitcher", "player_name"], dropna=False):
        mix = g["pitch_type"].value_counts(normalize=True)
        by_type = g.groupby("pitch_type").agg(
            velo=("release_speed", "mean"), spin=("release_spin_rate", "mean"), n=("release_speed", "size")
        )
        rows.append(
            {
                "player_id": f"mlbam-{int(pid) if pd.notna(pid) else ''}",
                "player_name": name if pd.notna(name) else "",
                "n_pitches": int(len(g)),
                "arsenal": ";".join(f"{t}:{s:.2f}" for t, s in mix.items()),
                "avg_velo": float(g["release_speed"].dropna().mean()),
                "avg_spin": float(g["release_spin_rate"].dropna().mean()),
                "top_type_velo": float(by_type.loc[mix.index[0], "velo"]),
                "exit_velo_against": float(g["launch_speed"].dropna().mean()),
                "xwoba_against": float(g["estimated_woba_using_speedangle"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows)


def derive_park_factors(pitches: pd.DataFrame, *, min_pa: int = 200, shrink_k: float = 500.0) -> pd.DataFrame:
    """Regressed wOBA park factors from pitch-level home/road events.

    PF(park) = wOBA@park / league wOBA, shrunk toward 1.0 by PA volume.
    >1 = hitter-friendly. Feeds rosetta.models.park validation (#3).
    """
    pa = _pa_events(pitches)
    if pa.empty:
        return pd.DataFrame(columns=["park", "park_factor", "n_pa"])
    pa = pa.copy()
    pa["woba_num"] = (
        WOBA_W["ubb"] * pa["events"].isin(BB_EVENTS).astype(float)
        + WOBA_W["hbp"] * pa["events"].isin(HBP_EVENTS).astype(float)
        + WOBA_W["1b"] * (pa["events"] == "single").astype(float)
        + WOBA_W["2b"] * (pa["events"] == "double").astype(float)
        + WOBA_W["3b"] * (pa["events"] == "triple").astype(float)
        + WOBA_W["hr"] * (pa["events"] == "home_run").astype(float)
    )
    pa["park"] = pa["home_team"].fillna("").astype(str)
    league_woba = pa["woba_num"].sum() / len(pa)
    rows = []
    for park, g in pa.groupby("park"):
        if not park:
            continue
        n = len(g)
        if n < min_pa:
            continue
        raw = (g["woba_num"].sum() / n) / league_woba if league_woba else 1.0
        w = n / (n + shrink_k)
        rows.append({"park": park, "park_factor": float(1.0 + w * (raw - 1.0)), "n_pa": int(n)})
    return pd.DataFrame(rows, columns=["park", "park_factor", "n_pa"])


def write_savant_snapshots(
    start: date | str,
    end: date | str,
    season: int,
    league_label: str,
    *,
    minors: bool = True,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    delay: float = 2.0,
) -> Path:
    """Download a date window, aggregate, and merge savant rows into snapshots."""
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    paths = download_range(start, end, minors=minors, out_dir=raw_dir, delay=delay)
    pitches = load_pitches(paths)
    bat = aggregate_batters(pitches, season, league_label)
    pit = aggregate_pitchers(pitches, season, league_label)
    slug = league_label.lower()
    for df, name in ((bat, f"{slug}_batters.csv"), (pit, f"{slug}_pitchers.csv")):
        path = root / name
        if path.exists() and not df.empty:
            old = pd.read_csv(path)
            old = old[old["is_synthetic"].astype(str) == "True"]
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                ["player_id", "season", "league"], keep="last"
            )
        if not df.empty:
            df.to_csv(path, index=False)
    feats = pitch_mix_features(pitches)
    if not feats.empty:
        feats["season"] = season
        feats["league"] = league_label
        feats.to_csv(root / f"{slug}_savant_features.csv", index=False)
    return root
