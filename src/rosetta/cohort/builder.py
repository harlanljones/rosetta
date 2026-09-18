"""Build transferred-player pairs from season tables + transfer index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.loaders import load_all_seasons, load_transfers

HITTER_STATS = ["bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba", "pa"]
PITCHER_STATS = ["k_pct", "bb_pct", "hr_fb", "era", "fip", "ip"]


def pair_seasons(
    transfers: pd.DataFrame,
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> pd.DataFrame:
    """Join pre/post season lines onto each transfer row."""
    rows: list[dict] = []
    bat = batters.set_index(["player_id", "season", "league"], drop=False)
    pit = pitchers.set_index(["player_id", "season", "league"], drop=False)

    for rec in transfers.to_dict(orient="records"):
        role = rec["role"]
        table = bat if role == "batter" else pit
        pre_key = (rec["player_id"], int(rec["from_season"]), rec["from_league"])
        post_key = (rec["player_id"], int(rec["to_season"]), rec["to_league"])
        if pre_key not in table.index or post_key not in table.index:
            continue
        pre = table.loc[pre_key]
        post = table.loc[post_key]
        if isinstance(pre, pd.DataFrame):
            pre = pre.iloc[0]
        if isinstance(post, pd.DataFrame):
            post = post.iloc[0]

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
) -> pd.DataFrame:
    """Full transferred-player cohort with sample thresholds."""
    transfers = load_transfers(snapshot_dir)
    batters = load_all_seasons(role="batter", snapshot_dir=snapshot_dir)
    pitchers = load_all_seasons(role="pitcher", snapshot_dir=snapshot_dir)
    paired = pair_seasons(transfers, batters, pitchers)
    if paired.empty:
        return paired

    batter_mask = (paired["role"] == "batter") & (paired["pre_pa"] >= min_pa) & (paired["post_pa"] >= min_pa)
    pitcher_mask = (paired["role"] == "pitcher") & (paired["pre_ip"] >= min_ip) & (paired["post_ip"] >= min_ip)
    return paired.loc[batter_mask | pitcher_mask].reset_index(drop=True)
