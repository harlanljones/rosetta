from __future__ import annotations

from pathlib import Path

from rosetta.backtest.evaluate import run_backtest
from rosetta.cohort.builder import build_cohort
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.factors import fit_factor_model
from rosetta.schema import LEAGUE_CHAIN
from rosetta.translate.api import translate_line


def test_cuba_in_chain_with_movers(tmp_path: Path) -> None:
    assert "CUBA" in LEAGUE_CHAIN
    root = build_snapshots(out_dir=tmp_path / "snap", seed=11)
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    cuba_links = cohort[cohort["from_league"] == "CUBA"]
    assert not cuba_links.empty, "synthetic fixtures must include CUBA movers"
    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=40, seed=2)
    assert model.path("CUBA", "MLB")[0] == "CUBA"
    assert model.path("CUBA", "MLB")[-1] == "MLB"
    # CUBA link must be estimated, not silent identity
    assert model.get_factor("batter", "CUBA", "CPBL", "woba") != 1.0


def test_missing_link_still_reports_width() -> None:
    from rosetta.models.factors import LeagueFactorModel

    model = LeagueFactorModel(links={})
    point, lo, hi, resid = model.get_interval("batter", "CUBA", "MLB", "woba")
    assert point == 1.0  # identity fallback
    assert resid > 0  # but never zero-width: missing links widen the band


def test_pitcher_role_shift_applies(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path / "snap", seed=12)
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=40, seed=3)
    pit = load_all_seasons(role="pitcher", snapshot_dir=root)
    starters = pit[(pit["g"] > 5) & (pit["gs"] / pit["g"] >= 0.6)]
    assert not starters.empty
    row = starters.iloc[0]
    base = translate_line(row, model=model, from_league=row["league"], to_league="MLB")
    shifted = translate_line(
        row, model=model, from_league=row["league"], to_league="MLB", target_pitch_role="RP"
    )
    assert any("Role adjustment" in n for n in shifted.notes)
    assert shifted.rates["k_pct"] > base.rates["k_pct"]


def test_backtest_reports_mlb_slice(tmp_path: Path) -> None:
    root = build_snapshots(out_dir=tmp_path / "snap", seed=13)
    out = tmp_path / "out"
    out.mkdir()
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    train = cohort[~cohort["holdout"].astype(bool)]
    model = fit_factor_model(train, n_boot=40, seed=4)
    factors = out / "factors.json"
    model.save(factors)
    metrics = run_backtest(root, factors_path=factors, n_boot=40, out_path=out / "backtest.json")
    assert "by_to_league" in metrics
    assert "by_stat_mlb_target" in metrics
    assert metrics["n_holdout_pairs_mlb_target"] >= 1
