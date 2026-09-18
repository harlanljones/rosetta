from __future__ import annotations

from pathlib import Path

from rosetta.backtest.evaluate import run_backtest
from rosetta.cohort.builder import build_cohort
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.factors import fit_factor_model
from rosetta.translate.api import translate, translate_frame


def test_translate_and_backtest(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path / "snap", seed=3)
    out = tmp_path / "out"
    out.mkdir()
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=40, seed=1)
    factors = out / "factors.json"
    model.save(factors)

    bat = load_all_seasons(role="batter", snapshot_dir=root)
    npb = bat[(bat["league"] == "NPB") & (bat["pa"] >= 100)].iloc[0]
    tr = translate(
        npb["player_id"],
        "NPB",
        season=int(npb["season"]),
        role="batter",
        factors_path=factors,
        snapshot_dir=root,
    )
    assert tr.woba is not None
    assert tr.path[0] == "NPB"
    assert tr.path[-1] == "MLB"
    assert "woba" in tr.rates

    frame = translate_frame(bat[bat["league"] == "KBO"].head(5), model=model)
    assert "mle_woba" in frame.columns
    assert len(frame) == 5

    metrics = run_backtest(root, factors_path=factors, n_boot=40, out_path=out / "backtest.json")
    assert metrics["n_holdout_pairs"] >= 1
    assert "by_stat" in metrics
    # Synthetic data should produce finite errors
    if "woba" in metrics["by_stat"]:
        assert metrics["by_stat"]["woba"]["rmse"] < 0.5
