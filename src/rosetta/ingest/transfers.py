"""Generate real transferred-player pairs from Chadwick register + season data.

Builds on the existing cross-league ID crosswalk — scans every player in the
season DataFrames for seasons across 2+ leagues, using the Chadwick key_uuid
to bridge different ID prefixes (mlbam-, npb-, bbref-).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.chadwick import build_id_crosswalk, load_register
from rosetta.ingest.loaders import load_all_seasons
from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs

LEAGUE_ORDER = ["CUBA", "CPBL", "KBO", "NPB", "AAA", "MLB"]


def generate_transfers(
    register_path: Path | None = None,
    snapshot_dir: Path | None = None,
    *,
    max_season_gap: int = 1,
) -> pd.DataFrame:
    """Build a real transfers.csv.

    Algorithm:
      1. Load all season lines and the Chadwick crosswalk.
      2. Group seasons by (key_uuid, role) — each group is one player.
      3. For players who appear in 2+ different leagues, emit a transfer
         row for each adjacent-season pair.
    """
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    register = load_register(register_path)
    crosswalk = build_id_crosswalk(register)

    batters = load_all_seasons(role="batter", snapshot_dir=root)
    pitchers = load_all_seasons(role="pitcher", snapshot_dir=root)
    batters["key_uuid"] = batters["player_id"].map(crosswalk)
    pitchers["key_uuid"] = pitchers["player_id"].map(crosswalk)

    all_seasons = pd.concat(
        [batters.assign(role="batter"), pitchers.assign(role="pitcher")],
        ignore_index=True,
    )

    rows: list[dict] = []
    seen: set[tuple] = set()

    for (uid, role), group in all_seasons.groupby(["key_uuid", "role"], sort=False):
        if pd.isna(uid) or not str(uid).strip():
            continue
        leagues = group[["league", "season", "player_id", "player_name", "age", "source"]].drop_duplicates(
            ["league", "season"]
        )
        unique_leagues = set(leagues["league"].unique())
        if len(unique_leagues) < 2:
            continue
        leagues = leagues.sort_values("season")
        player_id = str(leagues["player_id"].iloc[0])
        player_name = str(leagues["player_name"].dropna().iloc[0])

        lg_seasons: list[tuple[str, int, pd.Series]] = []
        for _, r in leagues.iterrows():
            lg_seasons.append((str(r["league"]), int(r["season"]), r))

        # For each adjacent-season cross-league pair
        for l_idx in range(len(lg_seasons)):
            lg_a, s_a, row_a = lg_seasons[l_idx]
            for l_idx2 in range(l_idx + 1, len(lg_seasons)):
                lg_b, s_b, row_b = lg_seasons[l_idx2]
                if lg_a == lg_b:
                    continue
                gap = abs(s_a - s_b)
                if gap > max_season_gap or gap == 0:
                    continue
                from_lg, to_lg = (lg_a, lg_b) if s_a <= s_b else (lg_b, lg_a)
                from_s, to_s = (s_a, s_b) if s_a <= s_b else (s_b, s_a)
                sk = (uid, role, from_lg, to_lg, from_s)
                if sk in seen:
                    continue
                seen.add(sk)
                age_from = float(row_a.get("age", 27)) if s_a <= s_b else float(row_b.get("age", 27))
                age_to = float(row_b.get("age", 27)) if s_a <= s_b else float(row_a.get("age", 27))
                rows.append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "role": role,
                        "from_league": from_lg,
                        "to_league": to_lg,
                        "from_season": from_s,
                        "to_season": to_s,
                        "age_from": age_from,
                        "age_to": age_to,
                        "holdout": to_s >= 2023,
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates(
            ["player_id", "role", "from_league", "to_league", "from_season"]
        ).sort_values(["from_league", "to_league", "from_season"]).reset_index(drop=True)
    return result


def write_transfer_snapshot(
    register_path: Path | None = None,
    snapshot_dir: Path | None = None,
    *,
    max_season_gap: int = 1,
    out_path: Path | None = None,
) -> Path:
    """Generate transfer pairs, append to existing transfers.csv, and write."""
    ensure_data_dirs()
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    new_pairs = generate_transfers(register_path, root, max_season_gap=max_season_gap)
    out = Path(out_path) if out_path else root / "transfers.csv"
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if not existing.empty and not new_pairs.empty:
        combined = pd.concat([existing, new_pairs], ignore_index=True).drop_duplicates(
            ["player_id", "role", "from_league", "to_league", "from_season"]
        ).reset_index(drop=True)
    elif not new_pairs.empty:
        combined = new_pairs
    else:
        combined = existing
    combined.to_csv(out, index=False)
    added = len(new_pairs) if not new_pairs.empty else 0
    total = len(combined)
    verb = "Appended" if added else "Wrote (no new)"
    print(f"  {verb} {added} new transfer pairs ({total} total) → {out}")
    return out