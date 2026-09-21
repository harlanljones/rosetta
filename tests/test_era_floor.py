"""Offline tests for the AAA ball-standardization era floor.

AAA switched to the MLB (livelier) ball starting in the 2019 season, so
pre-2019 AAA rates are a different run environment from 2019+ AAA and from
MLB in any era. `LEAGUE_ERA_FLOOR` / `apply_era_floor` exclude cohort rows
with a pre-2019 AAA endpoint from factor fitting; these tests pin that
behavior down at the helper, estimator, and historical-backtest layers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rosetta.models.factors import (
    LEAGUE_ERA_FLOOR,
    apply_era_floor,
    estimate_link_factor,
    fit_factor_model,
)


def _cohort_row(
    *,
    from_league: str,
    to_league: str,
    from_season: int,
    to_season: int,
    role: str = "batter",
    pre_woba: float = 0.30,
    post_woba: float = 0.31,
) -> dict:
    return {
        "from_league": from_league,
        "to_league": to_league,
        "from_season": from_season,
        "to_season": to_season,
        "role": role,
        "pre_woba": pre_woba,
        "post_woba": post_woba,
        "age_from": 27.0,
        "age_to": 27.0,
        "pre_pa": 100.0,
        "post_pa": 100.0,
    }


def test_apply_era_floor_drops_aaa_pre_2019_source_row() -> None:
    cohort = pd.DataFrame(
        [_cohort_row(from_league="AAA", to_league="MLB", from_season=2018, to_season=2019)]
    )
    out = apply_era_floor(cohort)
    assert out.empty


def test_apply_era_floor_drops_aaa_pre_2019_destination_row() -> None:
    # An AAA destination season before the floor (e.g. a demotion pair used
    # by some other cohort construction) must also be dropped.
    cohort = pd.DataFrame(
        [_cohort_row(from_league="MLB", to_league="AAA", from_season=2019, to_season=2018)]
    )
    out = apply_era_floor(cohort)
    assert out.empty


def test_apply_era_floor_keeps_2019_and_later_aaa_rows() -> None:
    cohort = pd.DataFrame(
        [
            _cohort_row(from_league="AAA", to_league="MLB", from_season=2019, to_season=2020),
            _cohort_row(from_league="AAA", to_league="MLB", from_season=2023, to_season=2024),
        ]
    )
    out = apply_era_floor(cohort)
    assert len(out) == 2


def test_apply_era_floor_keeps_non_aaa_rows_regardless_of_season() -> None:
    cohort = pd.DataFrame(
        [
            _cohort_row(from_league="KBO", to_league="NPB", from_season=2010, to_season=2011),
            _cohort_row(from_league="CUBA", to_league="CPBL", from_season=2005, to_season=2006),
        ]
    )
    out = apply_era_floor(cohort)
    assert len(out) == 2


def test_apply_era_floor_none_disables_filtering() -> None:
    cohort = pd.DataFrame(
        [_cohort_row(from_league="AAA", to_league="MLB", from_season=2018, to_season=2019)]
    )
    out = apply_era_floor(cohort, era_floor=None)
    assert len(out) == 1


def test_apply_era_floor_safe_without_season_columns() -> None:
    cohort = pd.DataFrame([{"from_league": "AAA", "to_league": "MLB", "role": "batter"}])
    out = apply_era_floor(cohort)
    assert len(out) == 1


def test_apply_era_floor_safe_on_empty_cohort() -> None:
    cohort = pd.DataFrame(columns=["from_league", "to_league", "from_season", "to_season"])
    out = apply_era_floor(cohort)
    assert out.empty


def test_default_era_floor_constant_is_aaa_2019() -> None:
    assert LEAGUE_ERA_FLOOR == {"AAA": 2019}


def test_fit_factor_model_default_floor_ignores_pathological_aaa_2018_pair() -> None:
    """A wildly different pre-2019 AAA pair must not move the fitted factor."""
    clean_rows = [
        _cohort_row(
            from_league="AAA", to_league="MLB", from_season=2022, to_season=2023,
            pre_woba=0.30 + i * 0.002, post_woba=0.31 + i * 0.002,
        )
        for i in range(8)
    ]
    pathological_row = _cohort_row(
        from_league="AAA", to_league="MLB", from_season=2018, to_season=2019,
        pre_woba=0.10, post_woba=0.45,  # would drag the raw factor sharply if pooled
    )
    cohort_without_pathological = pd.DataFrame(clean_rows)
    cohort_with_pathological = pd.DataFrame([*clean_rows, pathological_row])

    model_without = fit_factor_model(cohort_without_pathological, n_boot=5, seed=1)
    model_with_default_floor = fit_factor_model(cohort_with_pathological, n_boot=5, seed=1)

    factor_without = model_without.links["batter"]["AAA->MLB"]["woba"]["factor"]
    factor_with_floor = model_with_default_floor.links["batter"]["AAA->MLB"]["woba"]["factor"]
    assert factor_with_floor == pytest.approx(factor_without)

    # Disabling the floor lets the pathological pair through and changes the factor.
    model_no_floor = fit_factor_model(cohort_with_pathological, n_boot=5, seed=1, era_floor=None)
    factor_no_floor = model_no_floor.links["batter"]["AAA->MLB"]["woba"]["factor"]
    assert factor_no_floor != pytest.approx(factor_without)


def test_estimate_link_factor_is_unaffected_by_fit_factor_model_floor() -> None:
    """estimate_link_factor itself does not filter; only fit_factor_model applies the floor."""
    cohort = pd.DataFrame(
        [_cohort_row(from_league="AAA", to_league="MLB", from_season=2018, to_season=2019)]
    )
    est = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=0.0,
    )
    assert est["n"] == 1.0


# --- Historical-backtest metadata plumbing --------------------------------

from pathlib import Path  # noqa: E402

import rosetta.backtest.historical as historical  # noqa: E402
from rosetta.backtest.historical import run_historical_backtest  # noqa: E402
from rosetta.schema import HITTER_COLUMNS  # noqa: E402


def _hist_line(player_id: str, season: int, league: str, *, woba: float = 0.32) -> dict:
    row = {column: 0.0 for column in HITTER_COLUMNS}
    row.update(
        {
            "player_id": player_id,
            "player_name": player_id,
            "season": season,
            "league": league,
            "team": "T",
            "role": "batter",
            "age": 25.0,
            "pa": 100.0,
            "g": 30.0,
            "bb_pct": 0.08,
            "k_pct": 0.20,
            "iso": 0.16,
            "babip": 0.30,
            "hr_pct": 0.03,
            "avg": 0.27,
            "obp": 0.35,
            "slg": 0.43,
            "woba": woba,
            "source": "fixture",
            "is_synthetic": False,
        }
    )
    return row


def _era_floor_fixture(root: Path) -> Path:
    """One pre-2019 AAA training pair (p0), one clean training pair (p1), one eval pair (p1)."""
    aaa = [
        _hist_line("p0", 2018, "AAA", woba=0.05),  # pathological pre-floor pair
        _hist_line("p1", 2023, "AAA", woba=0.30),
        _hist_line("p1", 2024, "AAA", woba=0.301),
    ]
    mlb = [
        _hist_line("p0", 2019, "MLB", woba=0.55),  # pathological pre-floor pair
        _hist_line("p1", 2024, "MLB", woba=0.31),
        _hist_line("p1", 2025, "MLB", woba=0.311),
    ]
    pd.DataFrame(aaa, columns=HITTER_COLUMNS).to_csv(root / "aaa_batters.csv", index=False)
    pd.DataFrame(mlb, columns=HITTER_COLUMNS).to_csv(root / "mlb_batters.csv", index=False)
    return root


def test_historical_backtest_records_era_floor_and_excluded_count(tmp_path: Path, monkeypatch) -> None:
    root = _era_floor_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_historical_backtest(n_boot=1, min_pa=80, target_season=2025)

    assert payload["metadata"]["parameters"]["era_floor"] == {"AAA": 2019}
    assert payload["metadata"]["counts"]["training_pairs_excluded_era_floor"] == 1
    # The pre-2019 pair (p0) must not appear in the training_pairs artifact.
    assert all(pair["player_id"] != "p0" for pair in payload["training_pairs"])
    assert any(pair["player_id"] == "p1" for pair in payload["training_pairs"])


def test_historical_backtest_no_era_floor_includes_pre_2019_pair(tmp_path: Path, monkeypatch) -> None:
    root = _era_floor_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_historical_backtest(n_boot=1, min_pa=80, target_season=2025, era_floor=None)

    assert payload["metadata"]["parameters"]["era_floor"] is None
    assert payload["metadata"]["counts"]["training_pairs_excluded_era_floor"] == 0
    assert any(pair["player_id"] == "p0" for pair in payload["training_pairs"])


def test_rolling_backtest_records_era_floor_metadata(tmp_path: Path, monkeypatch) -> None:
    from rosetta.backtest.historical import run_rolling_backtest

    root = _era_floor_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_rolling_backtest(target_seasons=[2025], n_boot=1, min_pa=80)

    assert payload["metadata"]["parameters"]["era_floor"] == {"AAA": 2019}
    assert payload["metadata"]["counts"]["training_pairs_excluded_era_floor"] == 1
