"""Chadwick Bureau register helpers for cross-league player IDs.

Downloads the public people.csv when network is available; otherwise load a
local copy from data/snapshots/chadwick_people.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs

CHADWICK_SHARDS = tuple(f"people-{c}.csv" for c in "0123456789abcdef")
CHADWICK_BASE_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"


def download_register(out_path: Path | None = None, *, timeout: int = 120) -> Path:
    """Fetch the Chadwick register (sharded people-*.csv) into the snapshot dir."""
    ensure_data_dirs()
    path = Path(out_path) if out_path else SNAPSHOT_DIR / "chadwick_people.csv"
    frames = []
    try:
        for shard in CHADWICK_SHARDS:
            frames.append(pd.read_csv(f"{CHADWICK_BASE_URL}/{shard}", low_memory=False))
        df = pd.concat(frames, ignore_index=True)
    except Exception as exc:  # pragma: no cover - network
        raise RuntimeError(
            f"Failed to download Chadwick register: {exc}. "
            "Place people.csv at data/snapshots/chadwick_people.csv manually."
        ) from exc
    # Keep a lean subset useful for linking
    keep = [
        c
        for c in [
            "key_uuid",
            "key_mlbam",
            "key_retro",
            "key_bbref",
            "key_bbref_minors",
            "key_fangraphs",
            "key_npb",
            "key_kbo",
            "name_last",
            "name_first",
            "name_given",
            "birth_year",
            "mlb_played_first",
            "mlb_played_last",
        ]
        if c in df.columns
    ]
    slim = df[keep].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(path, index=False)
    return path


def load_register(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else SNAPSHOT_DIR / "chadwick_people.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chadwick register not found at {path}. Run: rosetta fetch-chadwick"
        )
    return pd.read_csv(path, low_memory=False)


def match_name(register: pd.DataFrame, last: str, first: str | None = None) -> pd.DataFrame:
    """Simple name lookup against the register."""
    q = register["name_last"].fillna("").str.lower() == last.lower()
    if first:
        q &= register["name_first"].fillna("").str.lower().str.startswith(first.lower())
    return register.loc[q]


def build_id_crosswalk(register: pd.DataFrame) -> dict[str, str]:
    """Build {prefixed_league_id → key_uuid} crosswalk.

    Maps IDs like 'mlbam-660271', 'kbo-62404', 'npb-1305137' to their
    Chadwick person UUID, so cohort building can link the same person across
    leagues that use different ID namespaces.
    """
    cross: dict[str, str] = {}
    for _, row in register.iterrows():
        uid = str(row.get("key_uuid", ""))
        if not uid:
            continue
        for col, prefix in (("key_mlbam", "mlbam"), ("key_npb", "npb"), ("key_kbo", "kbo"),
                            ("key_bbref", "bbref"), ("key_fangraphs", "fg"),
                            ("key_retro", "retro")):
            val = row.get(col)
            if pd.isna(val):
                continue
            raw = str(val).strip()
            if not raw:
                continue
            # CSV can store integer IDs as float (e.g. 660271.0 -> 660271)
            try:
                raw = str(int(float(raw)))
            except (ValueError, TypeError):
                pass
            cross[f"{prefix}-{raw}"] = uid
    return cross


def cross_reference_seasons(
    df: pd.DataFrame,
    crosswalk: dict[str, str],
) -> pd.DataFrame:
    """Add a 'key_uuid' column to a seasons frame via Chadwick ID matching.

    After calling, df['key_uuid'] can be used to match a player across leagues
    where they have different local player IDs (e.g. kbo-62404 vs mlbam-660271).
    """
    out = df.copy()
    out["key_uuid"] = out["player_id"].map(crosswalk)
    return out
