"""Hold out recent transfers, predict MLB-equivalent rates, score vs actuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rosetta.cohort.builder import build_cohort
from rosetta.models.factors import HITTER_STATS, PITCHER_STATS, LeagueFactorModel, fit_factor_model
from rosetta.translate.api import translate_line


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_backtest(
    snapshot_dir: Path | str | None = None,
    *,
    factors_path: Path | str | None = None,
    holdout_season_from: int = 2023,
    n_boot: int = 300,
    out_path: Path | str | None = None,
) -> dict:
    """Fit on pre-holdout movers; score holdout pairs.

    Returns metrics dict and optionally writes JSON.
    """
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
    cohort = build_cohort(snapshot_dir)
    if cohort.empty:
        raise RuntimeError("Empty cohort — generate snapshots first")

    # Holdout: to_season >= threshold OR explicit flag
    if "holdout" in cohort.columns:
        holdout = cohort[cohort["holdout"].astype(bool)].copy()
        train = cohort[~cohort["holdout"].astype(bool)].copy()
    else:
        holdout = cohort[cohort["to_season"] >= holdout_season_from].copy()
        train = cohort[cohort["to_season"] < holdout_season_from].copy()

    if factors_path and Path(factors_path).exists():
        model = LeagueFactorModel.load(factors_path)
    else:
        model = fit_factor_model(train, n_boot=n_boot)

    rows = []
    for _, rec in holdout.iterrows():
        role = rec["role"]
        stats = HITTER_STATS if role == "batter" else PITCHER_STATS
        pre = {s: rec[f"pre_{s}"] for s in stats}
        pre.update(
            {
                "player_id": rec["player_id"],
                "player_name": rec["player_name"],
                "season": rec["from_season"],
                "league": rec["from_league"],
                "role": role,
                "age": rec["age_from"],
                "pa": rec.get("pre_pa", 0),
                "ip": rec.get("pre_ip", 0),
            }
        )
        tr = translate_line(
            pre,
            model=model,
            from_league=rec["from_league"],
            to_league=rec["to_league"],
        )
        for s in stats:
            if s not in tr.rates:
                continue
            y_true = float(rec[f"post_{s}"])
            y_pred = float(tr.rates[s])
            lo = float(tr.lower.get(s, y_pred))
            hi = float(tr.upper.get(s, y_pred))
            rows.append(
                {
                    "player_id": rec["player_id"],
                    "role": role,
                    "stat": s,
                    "from_league": rec["from_league"],
                    "to_league": rec["to_league"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "lo": lo,
                    "hi": hi,
                    "covered_80": lo <= y_true <= hi,
                    "ae": abs(y_true - y_pred),
                }
            )

    detail = pd.DataFrame(rows)
    n_pairs = int(
        holdout[["player_id", "from_league", "to_league"]].drop_duplicates().shape[0]
    )
    metrics: dict = {"n_holdout_pairs": n_pairs, "by_stat": {}}
    if detail.empty:
        metrics["warning"] = "No holdout rows scored"
    else:
        for stat, g in detail.groupby("stat"):
            yt = g["y_true"].to_numpy()
            yp = g["y_pred"].to_numpy()
            metrics["by_stat"][str(stat)] = {
                "n": int(len(g)),
                "rmse": _rmse(yt, yp),
                "mae": _mae(yt, yp),
                "coverage_80": float(g["covered_80"].mean()),
                "mean_ae": float(g["ae"].mean()),
            }
        # Headline metrics
        if "woba" in metrics["by_stat"]:
            metrics["headline_woba_rmse"] = metrics["by_stat"]["woba"]["rmse"]
            metrics["headline_woba_coverage80"] = metrics["by_stat"]["woba"]["coverage_80"]
        if "fip" in metrics["by_stat"]:
            metrics["headline_fip_rmse"] = metrics["by_stat"]["fip"]["rmse"]
            metrics["headline_fip_coverage80"] = metrics["by_stat"]["fip"]["coverage_80"]
        if "era" in metrics["by_stat"]:
            metrics["headline_era_rmse"] = metrics["by_stat"]["era"]["rmse"]

    metrics["model_n_links"] = {
        role: list(links.keys()) for role, links in model.links.items()
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": metrics, "detail_head": detail.head(50).to_dict(orient="records")}
        out.write_text(json.dumps(payload, indent=2))
        detail_path = out.with_suffix(".detail.csv")
        detail.to_csv(detail_path, index=False)
        metrics["detail_csv"] = str(detail_path)

    return metrics
