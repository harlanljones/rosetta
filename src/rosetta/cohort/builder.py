"""Build transferred-player pairs from season tables + transfer index.

Supports cross-league ID matching via Chadwick register: when a transfer row
and the season table use different ID prefixes (e.g. 'kbo-62404' in KBO vs
'mlbam-660271' in MLB), pass a Chadwick register to `build_cohort` so that
shared `key_uuid` values bridge the gap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.loaders import load_all_seasons, load_transfers
from rosetta.paths import SNAPSHOT_DIR

HITTER_STATS = ["bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba", "pa"]
PITCHER_STATS = ["k_pct", "bb_pct", "hr_fb", "era", "fip", "ip"]


def _with_uuid_index(
    df: pd.DataFrame, crosswalk: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Augment df with key_uuid; return (player_id_index, uuid_index)."""
    pid_idx = df.set_index(["player_id", "season", "league"], drop=False)
    df2 = df.copy()
    df2["key_uuid"] = df2["player_id"].map(crosswalk)
    df2 = df2.dropna(subset=["key_uuid"]).drop_duplicates(
        ["key_uuid", "season", "league", "player_id"]
    )
    uuid_idx = df2.set_index(["key_uuid", "season", "league"], drop=False)
    return pid_idx, uuid_idx


def pair_seasons(
    transfers: pd.DataFrame,
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    *,
    id_crosswalk: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Join pre/post season lines onto each transfer row.

    When `id_crosswalk` is provided and a direct (player_id, season, league)
    lookup fails, the function retries via Chadwick `key_uuid` mapping.
    """
    rows: list[dict] = []
    bat_idx, bat_uuid = _with_uuid_index(batters, id_crosswalk or {})
    pit_idx, pit_uuid = _with_uuid_index(pitchers, id_crosswalk or {})

    for rec in transfers.to_dict(orient="records"):
        role = rec["role"]
        pid_idx = bat_idx if role == "batter" else pit_idx
        uuid_idx = bat_uuid if role == "batter" else pit_uuid
        pk = (rec["player_id"], int(rec["from_season"]), rec["from_league"])
        pk2 = (rec["player_id"], int(rec["to_season"]), rec["to_league"])

        pre = pid_idx.loc[pk] if pk in pid_idx.index else None
        post = pid_idx.loc[pk2] if pk2 in pid_idx.index else None

        if (pre is None or post is None) and id_crosswalk:
            tuid = id_crosswalk.get(str(rec["player_id"]))
            if tuid:
                flg, tlg = rec["from_league"], rec["to_league"]
                fs, ts = int(rec["from_season"]), int(rec["to_season"])
                slots = [("pre", flg, fs), ("post", tlg, ts)]
                for slot_name, lg, s in slots:
                    if (slot_name == "pre" and pre is not None) or (
                        slot_name == "post" and post is not None
                    ):
                        continue
                    ukey = (tuid, s, lg)
                    if ukey in uuid_idx.index:
                        alt_row = uuid_idx.loc[ukey]
                        alt_pid = (
                            str(alt_row["player_id"])
                            if not isinstance(alt_row, pd.DataFrame)
                            else str(alt_row.iloc[0]["player_id"])
                        )
                        alt_key = (alt_pid, s, lg)
                        if alt_key in pid_idx.index:
                            v = pid_idx.loc[alt_key]
                            if isinstance(v, pd.DataFrame):
                                v = v.iloc[0]
                            if slot_name == "pre":
                                pre = v
                            else:
                                post = v

        if pre is None or post is None:
            continue

        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        row = {
            "player_id": rec["player_id"],
            "player_name": rec["player_name"],
            "role": role,
            "from_league": rec["from_league"],
            "to_league": rec["to_league"],
            "from_season": int(rec["from_season"]),
            "to_season": int(rec["to_season"]),
            "age_from": float(rec["age_from"]),
            "age_to": float(rec["age_to"]),
            "holdout": bool(rec.get("holdout", False)),
        }
        for s in stats:
            row[f"pre_{s}"] = float(pre[s])
            row[f"post_{s}"] = float(post[s])
        rows.append(row)
    return pd.DataFrame(rows)


def build_cohort(
    snapshot_dir: Path | None = None,
    *,
    min_pa: float = 80.0,
    min_ip: float = 30.0,
    chadwick_path: Path | None = None,
) -> pd.DataFrame:
    """Full transferred-player cohort with sample thresholds.

    When `chadwick_path` points to a Chadwick register CSV, the builder
    cross-references league-specific IDs (mlbam-, kbo-, npb- etc.) via
    the Chadwick key_uuid so that transfers across different ID namespaces
    are discovered.
    """
    transfers = load_transfers(snapshot_dir)
    batters = load_all_seasons(role="batter", snapshot_dir=snapshot_dir)
    pitchers = load_all_seasons(role="pitcher", snapshot_dir=snapshot_dir)

    crosswalk: dict[str, str] | None = None
    # Default to the snapshot register so cross-namespace links (npb-*,
    # kbo-* vs mlbam-*) are always joined — without it only leagues sharing
    # the mlbam- id space can pair.
    resolved_chadwick = chadwick_path or (SNAPSHOT_DIR / "chadwick_people.csv")
    if resolved_chadwick and Path(resolved_chadwick).exists():
        from rosetta.ingest.chadwick import build_id_crosswalk, load_register

        register = load_register(resolved_chadwick)
        crosswalk = build_id_crosswalk(register)

    paired = pair_seasons(transfers, batters, pitchers, id_crosswalk=crosswalk)
    if paired.empty:
        return paired

    batter_mask = (
        (paired["role"] == "batter") & (paired["pre_pa"] >= min_pa) & (paired["post_pa"] >= min_pa)
    )
    pitcher_mask = pd.Series(False, index=paired.index)
    if "pre_ip" in paired.columns and "post_ip" in paired.columns:
        pitcher_mask = (
            (paired["role"] == "pitcher")
            & (paired["pre_ip"] >= min_ip)
            & (paired["post_ip"] >= min_ip)
        )
    return paired.loc[batter_mask | pitcher_mask].reset_index(drop=True)
