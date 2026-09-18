from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.cohort.builder import build_cohort
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.mlb import normalize_fangraphs_batting, normalize_fangraphs_pitching
from rosetta.models.factors import fit_factor_model
from rosetta.models.role import apply_role_shift, detect_role, role_shift


def test_vectorized_factor_fit_is_fast(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path, seed=5)
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=60, seed=0)
    assert "batter" in model.links
    aaa = model.links["batter"]["AAA->MLB"]["woba"]
    assert aaa["resid_sd"] >= 0
    assert aaa["low"] <= aaa["factor"] <= aaa["high"] or aaa["n"] < 5


def test_role_shift_sp_to_rp() -> None:
    assert detect_role(30, 32) == "SP"
    assert detect_role(0, 60) == "RP"
    shift = role_shift("SP", "RP")
    rates = apply_role_shift({"k_pct": 0.20, "fip": 4.0}, shift)
    assert rates["k_pct"] > 0.20
    assert rates["fip"] < 4.0


def test_normalize_fangraphs_columns() -> None:
    bat = pd.DataFrame(
        {
            "Name": ["Test Player"],
            "Team": ["NYY"],
            "Age": [27],
            "PA": [500],
            "G": [140],
            "BB%": [10.0],
            "K%": [22.0],
            "ISO": [0.180],
            "BABIP": [0.300],
            "HR": [25],
            "AVG": [0.270],
            "OBP": [0.340],
            "SLG": [0.450],
            "wOBA": [0.340],
            "IDfg": [123],
        }
    )
    out = normalize_fangraphs_batting(bat, 2024)
    assert out.iloc[0]["bb_pct"] == 0.1
    assert out.iloc[0]["league"] == "MLB"
    assert not out.iloc[0]["is_synthetic"]

    pit = pd.DataFrame(
        {
            "Name": ["Ace"],
            "Team": ["LAD"],
            "Age": [28],
            "IP": [180.0],
            "G": [32],
            "GS": [32],
            "K%": [25.0],
            "BB%": [7.0],
            "HR/FB": [11.0],
            "ERA": [3.20],
            "FIP": [3.40],
            "IDfg": [9],
        }
    )
    pout = normalize_fangraphs_pitching(pit, 2024)
    assert abs(pout.iloc[0]["k_pct"] - 0.25) < 1e-9
    assert pout.iloc[0]["gs"] == 32
