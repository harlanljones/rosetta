"""Chadwick Bureau register helpers for cross-league player IDs.

Downloads the public people.csv when network is available; otherwise load a
local copy from data/snapshots/chadwick_people.csv.
"""

from __future__ import annotations

import re
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


_KBO_SURNAME_SWAPS: dict[str, tuple[str, ...]] = {
    "kim": ("gim",),
    "gim": ("kim",),
    "park": ("bak",),
    "bak": ("park",),
    "choi": ("choe",),
    "choe": ("choi",),
    "jung": ("jeong",),
    "jeong": ("jung",),
    "lee": ("i", "yi"),
    "i": ("lee",),
    "yi": ("lee",),
    "cho": ("jo",),
    "jo": ("cho",),
    "yoo": ("yu",),
    "yu": ("yoo",),
}


def _normalize_kbo_name_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def kbo_name_variants(last: str, given: str) -> set[str]:
    """Romanized keys that may appear on KBO leaderboards (RR vs MR spellings)."""
    last_n = _normalize_kbo_name_token(last)
    given_n = _normalize_kbo_name_token(given)
    if not last_n or not given_n:
        return set()
    last_forms = {last_n}
    for alt in _KBO_SURNAME_SWAPS.get(last_n, ()):
        last_forms.add(alt)
    variants: set[str] = set()
    for lf in last_forms:
        variants.add(f"{lf}{given_n}")
        variants.add(f"{given_n}{lf}")
    return variants


def _register_kbo_name_key(cross: dict[str, str], name_key: str, uid: str) -> None:
    existing = cross.get(name_key)
    if existing is None:
        cross[name_key] = uid
    elif existing != uid:
        cross[name_key] = ""


def load_kbo_id_map(path: Path | None = None) -> dict[str, str]:
    """Load optional ``kbo_id_map.csv`` rows ``kbo_id,key_uuid`` or ``kbo_id,key_mlbam``."""
    path = Path(path) if path else SNAPSHOT_DIR / "kbo_id_map.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "kbo_id" not in frame.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        raw_id = str(row["kbo_id"]).strip()
        if not raw_id:
            continue
        player_key = raw_id if raw_id.startswith("kbo-") else f"kbo-{raw_id}"
        if "key_uuid" in frame.columns and pd.notna(row.get("key_uuid")):
            out[player_key] = str(row["key_uuid"]).strip()
        elif "key_mlbam" in frame.columns and pd.notna(row.get("key_mlbam")):
            try:
                mlb = str(int(float(row["key_mlbam"])))
            except (ValueError, TypeError):
                mlb = str(row["key_mlbam"]).strip()
            out[player_key] = f"__mlbam__{mlb}"
    return out


def enrich_crosswalk_with_kbo(
    crosswalk: dict[str, str],
    register: pd.DataFrame,
    *,
    snapshot_dir: Path | None = None,
) -> dict[str, str]:
    """Add KBO player_id keys when Chadwick lacks ``key_kbo`` (not in public register)."""
    cross = dict(crosswalk)
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    id_map = load_kbo_id_map(root / "kbo_id_map.csv")
    mlbam_to_uuid = {
        f"mlbam-{str(int(float(row['key_mlbam'])))}": str(row["key_uuid"])
        for _, row in register.iterrows()
        if pd.notna(row.get("key_mlbam")) and pd.notna(row.get("key_uuid"))
    }
    for player_key, target in id_map.items():
        if target.startswith("__mlbam__"):
            uid = mlbam_to_uuid.get(f"mlbam-{target.removeprefix('__mlbam__')}")
            if uid:
                cross[player_key] = uid
        elif target:
            cross[player_key] = target

    for _, row in register.iterrows():
        uid = str(row.get("key_uuid", ""))
        if not uid or pd.isna(row.get("key_mlbam")):
            continue
        last = str(row.get("name_last", "") or "").strip()
        given = str(row.get("name_given", "") or "").strip()
        if not last or not given:
            continue
        for variant in kbo_name_variants(last, given):
            _register_kbo_name_key(cross, f"kbo-name-{variant}", uid)

    # Birth-year + inferred age from KBO snapshots when ages are known.
    try:
        from rosetta.ingest.loaders import load_league_seasons

        kbo = pd.concat(
            [
                load_league_seasons("KBO", role="batter", snapshot_dir=root),
                load_league_seasons("KBO", role="pitcher", snapshot_dir=root),
            ],
            ignore_index=True,
        )
    except (FileNotFoundError, ValueError):
        kbo = pd.DataFrame()
    if not kbo.empty and "birth_year" in register.columns:
        reg = register.dropna(subset=["birth_year"]).copy()
        reg["birth_year"] = pd.to_numeric(reg["birth_year"], errors="coerce")
        for _, line in kbo.drop_duplicates("player_id").iterrows():
            pid = str(line["player_id"])
            if cross.get(pid):
                continue
            age = float(line.get("age", 27) or 27)
            season = int(line["season"])
            if age == 27.0 and pid.startswith("kbo-name-"):  # default age placeholder
                continue
            by = int(season - age)
            cands = reg.loc[reg["birth_year"] == by]
            if cands.empty:
                continue
            name_norm = _normalize_kbo_name_token(str(line.get("player_name", "")))
            for _, cand in cands.iterrows():
                for variant in kbo_name_variants(str(cand["name_last"]), str(cand["name_given"])):
                    if variant == name_norm:
                        cross[pid] = str(cand["key_uuid"])
                        break
                if cross.get(pid):
                    break
    return cross


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
        for col, prefix in (
            ("key_mlbam", "mlbam"),
            ("key_npb", "npb"),
            ("key_kbo", "kbo"),
            ("key_bbref", "bbref"),
            ("key_fangraphs", "fg"),
            ("key_retro", "retro"),
        ):
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
        # Name-based fallback for the npb-name-* ids left by the fetcher when
        # a player is no longer on the BIS active index ("Last, First" format
        # on npb.jp matches name_last + name_given). Exact full-name match,
        # only for players that carry a key_npb, to keep collision risk low.
        last = str(row.get("name_last", "") or "").strip()
        given = str(row.get("name_given", "") or "").strip()
        if last and given and pd.notna(row.get("key_npb")):
            name_key = f"npb-name-{last}, {given}"
            existing = cross.get(name_key)
            if existing is None:
                cross[name_key] = uid
            elif existing != uid:
                # Ambiguous full name — drop it rather than mis-link.
                cross.pop(name_key, None)
                cross[name_key] = ""  # tombstone: never link ambiguous names
    return enrich_crosswalk_with_kbo(cross, register)


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
