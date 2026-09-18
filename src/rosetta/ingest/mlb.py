"""MLB seasonal ingest via pybaseball (optional dependency).

Writes normalized snapshot CSVs compatible with `rosetta.ingest.loaders`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS


def _require_pybaseball() -> Any:
    try:
        import pybaseball
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pybaseball is required for MLB ingest. Install with: pip install -e '.[scrape]'"
        ) from exc
    return pybaseball


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d = d.replace(0, pd.NA)
    return (n / d).astype(float)


def normalize_fangraphs_batting(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Map FanGraphs batting columns → Rosetta hitter schema."""
    col = {c.lower(): c for c in df.columns}

    def get(*names: str, default: float | str = 0.0) -> pd.Series:
        for n in names:
            if n.lower() in col:
                return df[col[n.lower()]]
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df))

    pa = pd.to_numeric(get("PA", "pa"), errors="coerce").fillna(0)
    bb = pd.to_numeric(get("BB%", "bb%"), errors="coerce")
    # FanGraphs BB%/K% are often 0-100; normalize if needed
    if bb.dropna().abs().median() > 1:
        bb = bb / 100.0
    k = pd.to_numeric(get("K%", "k%"), errors="coerce")
    if k.dropna().abs().median() > 1:
        k = k / 100.0
    iso = pd.to_numeric(get("ISO", "iso"), errors="coerce")
    babip = pd.to_numeric(get("BABIP", "babip"), errors="coerce")
    hr = pd.to_numeric(get("HR", "hr"), errors="coerce").fillna(0)
    woba = pd.to_numeric(get("wOBA", "woba"), errors="coerce")
    name = get("Name", "name", default="").astype(str)
    team = get("Team", "team", default="").astype(str)
    # IDs: prefer FanGraphs id if present
    pid = get("IDfg", "playerid", "key_fangraphs", default="")
    pid = pid.astype(str).where(pid.astype(str).str.len() > 0, name)

    out = pd.DataFrame(
        {
            "player_id": "mlb-" + pid.astype(str),
            "player_name": name,
            "season": season,
            "league": "MLB",
            "team": team,
            "role": "batter",
            "age": pd.to_numeric(get("Age", "age"), errors="coerce").fillna(27),
            "pa": pa,
            "g": pd.to_numeric(get("G", "g"), errors="coerce").fillna(0),
            "bb_pct": bb.fillna(0.08),
            "k_pct": k.fillna(0.22),
            "iso": iso.fillna(0.15),
            "babip": babip.fillna(0.30),
            "hr_pct": _safe_div(hr, pa).fillna(0.03),
            "avg": pd.to_numeric(get("AVG", "avg"), errors="coerce"),
            "obp": pd.to_numeric(get("OBP", "obp"), errors="coerce"),
            "slg": pd.to_numeric(get("SLG", "slg"), errors="coerce"),
            "woba": woba.fillna(0.32),
            "home_pa": pa * 0.5,
            "road_pa": pa * 0.5,
            "home_woba": woba.fillna(0.32),
            "road_woba": woba.fillna(0.32),
            "park_id": team,
            "source": "fangraphs",
            "is_synthetic": False,
        }
    )
    return out.reindex(columns=HITTER_COLUMNS)


def normalize_fangraphs_pitching(df: pd.DataFrame, season: int) -> pd.DataFrame:
    col = {c.lower(): c for c in df.columns}

    def get(*names: str, default: float | str = 0.0) -> pd.Series:
        for n in names:
            if n.lower() in col:
                return df[col[n.lower()]]
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df))

    ip = pd.to_numeric(get("IP", "ip"), errors="coerce").fillna(0)
    k = pd.to_numeric(get("K%", "k%"), errors="coerce")
    if k.dropna().abs().median() > 1:
        k = k / 100.0
    bb = pd.to_numeric(get("BB%", "bb%"), errors="coerce")
    if bb.dropna().abs().median() > 1:
        bb = bb / 100.0
    name = get("Name", "name", default="").astype(str)
    team = get("Team", "team", default="").astype(str)
    pid = get("IDfg", "playerid", "key_fangraphs", default="")
    pid = pid.astype(str).where(pid.astype(str).str.len() > 0, name)
    hr_fb = pd.to_numeric(get("HR/FB", "hr/fb"), errors="coerce")
    if hr_fb.dropna().abs().median() > 1:
        hr_fb = hr_fb / 100.0

    out = pd.DataFrame(
        {
            "player_id": "mlb-" + pid.astype(str),
            "player_name": name,
            "season": season,
            "league": "MLB",
            "team": team,
            "role": "pitcher",
            "age": pd.to_numeric(get("Age", "age"), errors="coerce").fillna(27),
            "ip": ip,
            "g": pd.to_numeric(get("G", "g"), errors="coerce").fillna(0),
            "gs": pd.to_numeric(get("GS", "gs"), errors="coerce").fillna(0),
            "k_pct": k.fillna(0.22),
            "bb_pct": bb.fillna(0.08),
            "hr_fb": hr_fb.fillna(0.12),
            "era": pd.to_numeric(get("ERA", "era"), errors="coerce").fillna(4.2),
            "fip": pd.to_numeric(get("FIP", "fip"), errors="coerce").fillna(4.2),
            "home_pa": ip * 2.2,
            "road_pa": ip * 2.1,
            "home_era": pd.to_numeric(get("ERA", "era"), errors="coerce").fillna(4.2),
            "road_era": pd.to_numeric(get("ERA", "era"), errors="coerce").fillna(4.2),
            "park_id": team,
            "source": "fangraphs",
            "is_synthetic": False,
        }
    )
    return out.reindex(columns=PITCHER_COLUMNS)


def fetch_mlb_season(season: int, *, qual_bat: int = 50, qual_pit: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    pb = _require_pybaseball()
    bat = pb.batting_stats(season, qual=qual_bat)
    pit = pb.pitching_stats(season, qual=qual_pit)
    return normalize_fangraphs_batting(bat, season), normalize_fangraphs_pitching(pit, season)


def write_mlb_snapshots(
    seasons: list[int],
    *,
    out_dir: Path | None = None,
    append: bool = True,
) -> Path:
    """Fetch one or more MLB seasons and write snapshot CSVs."""
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    bats: list[pd.DataFrame] = []
    pits: list[pd.DataFrame] = []
    for season in seasons:
        b, p = fetch_mlb_season(season)
        bats.append(b)
        pits.append(p)
    bat_df = pd.concat(bats, ignore_index=True)
    pit_df = pd.concat(pits, ignore_index=True)
    bat_path = root / "mlb_batters.csv"
    pit_path = root / "mlb_pitchers.csv"
    if append and bat_path.exists():
        old = pd.read_csv(bat_path)
        # Prefer real rows for overlapping player-seasons
        old = old[old["is_synthetic"] == True]  # noqa: E712
        bat_df = pd.concat([old, bat_df], ignore_index=True).drop_duplicates(
            ["player_id", "season", "league"], keep="last"
        )
    if append and pit_path.exists():
        old = pd.read_csv(pit_path)
        old = old[old["is_synthetic"] == True]  # noqa: E712
        pit_df = pd.concat([old, pit_df], ignore_index=True).drop_duplicates(
            ["player_id", "season", "league"], keep="last"
        )
    bat_df.to_csv(bat_path, index=False)
    pit_df.to_csv(pit_path, index=False)
    return root
