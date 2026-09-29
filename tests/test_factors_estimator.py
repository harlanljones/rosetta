from __future__ import annotations

import pandas as pd
import pytest

from rosetta.models.factors import (
    _to_odds,
    bootstrap_link_factor,
    estimate_link_factor,
    fit_factor_model,
)


def _cohort_two_pairs(pre_a: float, post_a: float, pre_b: float, post_b: float) -> pd.DataFrame:
    """Two batter woba pairs, equal weight, ages equal (identity age adjustment)."""
    rows = [
        {
            "from_league": "AAA",
            "to_league": "MLB",
            "role": "batter",
            "pre_woba": pre_a,
            "post_woba": post_a,
            "age_from": 27.0,
            "age_to": 27.0,
            "pre_pa": 100.0,
            "post_pa": 100.0,
        },
        {
            "from_league": "AAA",
            "to_league": "MLB",
            "role": "batter",
            "pre_woba": pre_b,
            "post_woba": post_b,
            "age_from": 27.0,
            "age_to": 27.0,
            "pre_pa": 100.0,
            "post_pa": 100.0,
        },
    ]
    return pd.DataFrame(rows)


def _bigger_cohort() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "batter",
                "pre_woba": 0.30 + i * 0.003,
                "post_woba": 0.31 + i * 0.004,
                "age_from": 25.0 + i * 0.1,
                "age_to": 26.0 + i * 0.1,
                "pre_pa": 100.0 + i,
                "post_pa": 100.0 + i,
            }
        )
        rows.append(
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "pitcher",
                "pre_fip": 4.0 - i * 0.02,
                "post_fip": 3.9 - i * 0.01,
                "pre_k_pct": 0.2,
                "post_k_pct": 0.21,
                "pre_bb_pct": 0.08,
                "post_bb_pct": 0.07,
                "pre_hr_fb": 0.10,
                "post_hr_fb": 0.11,
                "pre_era": 3.8,
                "post_era": 3.7,
                "age_from": 26.0 + i * 0.1,
                "age_to": 27.0 + i * 0.1,
                "pre_ip": 40.0 + i,
                "post_ip": 40.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_ratio_of_means_default_matches_explicit_and_omitted_kwarg() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    omitted = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=20.0
    )
    explicit = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="woba",
        role="batter",
        prior_n=20.0,
        estimator="ratio_of_means",
    )
    assert omitted == explicit
    # Hand-computed ratio_of_means raw: (0.05 + 0.10) / (0.01 + 0.10) = 0.15/0.11
    assert explicit["raw_factor"] == pytest.approx(0.15 / 0.11)


def test_mean_ratio_explicit_still_available() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    explicit = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="woba",
        role="batter",
        prior_n=20.0,
        estimator="mean_ratio",
    )
    # Hand-computed mean_ratio raw: mean(5.0, 1.0) = 3.0
    assert explicit["raw_factor"] == pytest.approx(3.0)


def test_ratio_of_means_hand_computed() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    result = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="woba",
        role="batter",
        prior_n=0.0,
        estimator="ratio_of_means",
    )
    # raw = sum(w*post) / sum(w*pre_adj) = (0.05 + 0.10) / (0.01 + 0.10) = 0.15/0.11
    assert result["raw_factor"] == pytest.approx(0.15 / 0.11)
    # prior_n=0 -> shrink=1.0 -> factor == raw_factor
    assert result["shrink"] == pytest.approx(1.0)
    assert result["factor"] == pytest.approx(0.15 / 0.11)

    mean_ratio_result = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="woba",
        role="batter",
        prior_n=0.0,
        estimator="mean_ratio",
    )
    assert mean_ratio_result["raw_factor"] == pytest.approx(3.0)
    assert result["raw_factor"] != pytest.approx(mean_ratio_result["raw_factor"])


@pytest.mark.parametrize(
    "fn_name", ["estimate_link_factor", "bootstrap_link_factor", "fit_factor_model"]
)
def test_unknown_estimator_raises(fn_name: str) -> None:
    cohort = _bigger_cohort()
    with pytest.raises(ValueError, match="unknown estimator"):
        if fn_name == "estimate_link_factor":
            estimate_link_factor(
                cohort,
                from_league="AAA",
                to_league="MLB",
                stat="woba",
                role="batter",
                prior_n=20.0,
                estimator="bogus",
            )
        elif fn_name == "bootstrap_link_factor":
            bootstrap_link_factor(
                cohort,
                from_league="AAA",
                to_league="MLB",
                stat="woba",
                role="batter",
                prior_n=20.0,
                n_boot=5,
                seed=1,
                estimator="bogus",
            )
        else:
            fit_factor_model(cohort, n_boot=5, seed=1, estimator="bogus")


def test_fit_factor_model_ratio_of_means_reduces_small_pre_blowup() -> None:
    """A near-zero pre-move rate should not explode the factor under ratio_of_means."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "pitcher",
                "pre_hr_fb": 0.11 + i * 0.001,
                "post_hr_fb": 0.11 + i * 0.001,
                "pre_k_pct": 0.2,
                "post_k_pct": 0.2,
                "pre_bb_pct": 0.08,
                "post_bb_pct": 0.08,
                "pre_era": 3.8,
                "post_era": 3.8,
                "pre_fip": 3.9,
                "post_fip": 3.9,
                "age_from": 27.0,
                "age_to": 27.0,
                "pre_ip": 40.0,
                "post_ip": 40.0,
            }
        )
    # One pathological pair with a tiny pre_adj rate that would dominate a
    # per-pair-ratio mean but is properly weighted down in ratio-of-means.
    rows.append(
        {
            "from_league": "AAA",
            "to_league": "MLB",
            "role": "pitcher",
            "pre_hr_fb": 0.001,
            "post_hr_fb": 0.11,
            "pre_k_pct": 0.2,
            "post_k_pct": 0.2,
            "pre_bb_pct": 0.08,
            "post_bb_pct": 0.08,
            "pre_era": 3.8,
            "post_era": 3.8,
            "pre_fip": 3.9,
            "post_fip": 3.9,
            "age_from": 27.0,
            "age_to": 27.0,
            "pre_ip": 40.0,
            "post_ip": 40.0,
        }
    )
    cohort = pd.DataFrame(rows)
    mean_ratio = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="hr_fb",
        role="pitcher",
        prior_n=0.0,
        estimator="mean_ratio",
    )
    ratio_of_means = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="hr_fb",
        role="pitcher",
        prior_n=0.0,
        estimator="ratio_of_means",
    )
    assert mean_ratio["raw_factor"] > ratio_of_means["raw_factor"]
    assert ratio_of_means["raw_factor"] == pytest.approx(1.0, abs=0.15)
    assert mean_ratio["raw_factor"] > 5.0


def test_odds_ratio_hand_computed() -> None:
    """Odds-ratio estimator: weighted mean of per-pair odds(post)/odds(pre_adj)."""
    cohort = pd.DataFrame(
        [
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "batter",
                "pre_k_pct": 0.10,
                "post_k_pct": 0.05,
                "age_from": 27.0,
                "age_to": 27.0,
                "pre_pa": 100.0,
                "post_pa": 100.0,
            },
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "batter",
                "pre_k_pct": 0.20,
                "post_k_pct": 0.10,
                "age_from": 27.0,
                "age_to": 27.0,
                "pre_pa": 100.0,
                "post_pa": 100.0,
            },
        ]
    )
    result = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="k_pct",
        role="batter",
        prior_n=0.0,
        estimator="odds_ratio",
    )
    odds = lambda p: p / (1 - p)  # noqa: E731
    expected = (odds(0.05) / odds(0.10) + odds(0.10) / odds(0.20)) / 2
    assert result["raw_factor"] == pytest.approx(expected)
    assert result["factor"] == pytest.approx(expected)


def test_odds_ratio_rejects_non_proportion_stats() -> None:
    cohort = _bigger_cohort()
    for stat in ("era", "fip"):
        with pytest.raises(ValueError, match="proportion"):
            estimate_link_factor(
                cohort,
                from_league="AAA",
                to_league="MLB",
                stat=stat,
                role="pitcher",
                prior_n=20.0,
                estimator="odds_ratio",
            )


def test_odds_ratio_clips_extreme_rates() -> None:
    """A 1.0 post rate must not produce an infinite odds ratio."""
    cohort = pd.DataFrame(
        [
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "batter",
                "pre_k_pct": 0.20,
                "post_k_pct": 1.0,
                "age_from": 27.0,
                "age_to": 27.0,
                "pre_pa": 100.0,
                "post_pa": 100.0,
            },
            {
                "from_league": "AAA",
                "to_league": "MLB",
                "role": "batter",
                "pre_k_pct": 0.20,
                "post_k_pct": 0.25,
                "age_from": 27.0,
                "age_to": 27.0,
                "pre_pa": 100.0,
                "post_pa": 100.0,
            },
        ]
    )
    result = estimate_link_factor(
        cohort,
        from_league="AAA",
        to_league="MLB",
        stat="k_pct",
        role="batter",
        prior_n=0.0,
        estimator="odds_ratio",
    )
    assert result["raw_factor"] == pytest.approx(
        (_to_odds(0.99) / _to_odds(0.20) + _to_odds(0.25) / _to_odds(0.20)) / 2
    )
    assert result["raw_factor"] < 10_000.0


def test_odds_ratio_model_falls_back_for_era_fip() -> None:
    """Full-cohort odds_ratio fit: proportion stats use odds, era/fip use ROM."""
    cohort = _bigger_cohort()
    model = fit_factor_model(cohort, n_boot=5, seed=1, estimator="odds_ratio")
    rom = fit_factor_model(cohort, n_boot=5, seed=1, estimator="ratio_of_means")
    # era/fip fell back to ratio_of_means → identical to the ROM fit
    for stat in ("era", "fip"):
        assert model.links["pitcher"]["AAA->MLB"][stat]["factor"] == pytest.approx(
            rom.links["pitcher"]["AAA->MLB"][stat]["factor"]
        )
    # k_pct actually used the odds estimator → differs from ROM
    assert model.links["pitcher"]["AAA->MLB"]["k_pct"]["factor"] != pytest.approx(
        rom.links["pitcher"]["AAA->MLB"]["k_pct"]["factor"]
    )


def test_odds_ratio_flows_through_fit_factor_model() -> None:
    rows = [
        {
            "from_league": "AAA",
            "to_league": "MLB",
            "role": "batter",
            "pre_woba": 0.30 + i * 0.003,
            "post_woba": 0.31 + i * 0.004,
            "age_from": 25.0 + i * 0.1,
            "age_to": 26.0 + i * 0.1,
            "pre_pa": 100.0 + i,
            "post_pa": 100.0 + i,
        }
        for i in range(6)
    ]
    model = fit_factor_model(pd.DataFrame(rows), n_boot=5, seed=1, estimator="odds_ratio")
    assert model.links["batter"]["AAA->MLB"]["woba"]["n"] == 6
