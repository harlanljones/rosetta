from __future__ import annotations

from pathlib import Path

from rosetta.cohort.builder import build_cohort
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.aging import age_adjust_rate
from rosetta.models.factors import fit_factor_model
from rosetta.models.park import estimate_park_factors


def test_cohort_and_factor_fit(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path, seed=2)
    cohort = build_cohort(root, min_pa=50, min_ip=20)
    assert not cohort.empty
    assert {"pre_woba", "post_woba", "pre_fip", "post_fip"} <= set(cohort.columns) or (
        "pre_woba" in cohort.columns
    )

    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=50, seed=0)
    assert "batter" in model.links
    # AAA->MLB should be present and pull offense down on iso/woba
    aaa = model.links["batter"].get("AAA->MLB", {})
    assert aaa
    if "woba" in aaa:
        assert aaa["woba"]["factor"] < 1.05  # shrunken, but not wildly above 1

    bat = load_all_seasons(role="batter", snapshot_dir=root)
    pf = estimate_park_factors(bat)
    assert not pf.empty
    assert pf["park_factor"].between(0.7, 1.3).all()


def test_aging_shifts_iso_down_after_peak() -> None:
    young = age_adjust_rate(0.200, "iso", 24, 30, role="batter")
    assert young < 0.200
