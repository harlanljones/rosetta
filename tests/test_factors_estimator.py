from __future__ import annotations

import pandas as pd
import pytest

from rosetta.models.factors import (
    bootstrap_link_factor,
    estimate_link_factor,
    fit_factor_model,
)


def _cohort_two_pairs(pre_a: float, post_a: float, pre_b: float, post_b: float) -> pd.DataFrame:
    """Two batter woba pairs, equal weight, ages equal (identity age adjustment)."""
    rows = [
        {
            "from_league": "AAA", "to_league": "MLB", "role": "batter",
            "pre_woba": pre_a, "post_woba": post_a,
            "age_from": 27.0, "age_to": 27.0, "pre_pa": 100.0, "post_pa": 100.0,
        },
        {
            "from_league": "AAA", "to_league": "MLB", "role": "batter",
            "pre_woba": pre_b, "post_woba": post_b,
            "age_from": 27.0, "age_to": 27.0, "pre_pa": 100.0, "post_pa": 100.0,
        },
    ]
    return pd.DataFrame(rows)


def _bigger_cohort() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "from_league": "AAA", "to_league": "MLB", "role": "batter",
                "pre_woba": 0.30 + i * 0.003, "post_woba": 0.31 + i * 0.004,
                "age_from": 25.0 + i * 0.1, "age_to": 26.0 + i * 0.1,
                "pre_pa": 100.0 + i, "post_pa": 100.0 + i,
            }
        )
        rows.append(
            {
                "from_league": "AAA", "to_league": "MLB", "role": "pitcher",
                "pre_fip": 4.0 - i * 0.02, "post_fip": 3.9 - i * 0.01,
                "pre_k_pct": 0.2, "post_k_pct": 0.21,
                "pre_bb_pct": 0.08, "post_bb_pct": 0.07,
                "pre_hr_fb": 0.10, "post_hr_fb": 0.11,
                "pre_era": 3.8, "post_era": 3.7,
                "age_from": 26.0 + i * 0.1, "age_to": 27.0 + i * 0.1,
                "pre_ip": 40.0 + i, "post_ip": 40.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_ratio_of_means_default_matches_explicit_and_omitted_kwarg() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    omitted = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=20.0
    )
    explicit = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=20.0,
        estimator="ratio_of_means",
    )
    assert omitted == explicit
    # Hand-computed ratio_of_means raw: (0.05 + 0.10) / (0.01 + 0.10) = 0.15/0.11
    assert explicit["raw_factor"] == pytest.approx(0.15 / 0.11)


def test_mean_ratio_explicit_still_available() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    explicit = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=20.0,
        estimator="mean_ratio",
    )
    # Hand-computed mean_ratio raw: mean(5.0, 1.0) = 3.0
    assert explicit["raw_factor"] == pytest.approx(3.0)


def test_ratio_of_means_hand_computed() -> None:
    cohort = _cohort_two_pairs(0.01, 0.05, 0.10, 0.10)
    result = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=0.0,
        estimator="ratio_of_means",
    )
    # raw = sum(w*post) / sum(w*pre_adj) = (0.05 + 0.10) / (0.01 + 0.10) = 0.15/0.11
    assert result["raw_factor"] == pytest.approx(0.15 / 0.11)
    # prior_n=0 -> shrink=1.0 -> factor == raw_factor
    assert result["shrink"] == pytest.approx(1.0)
    assert result["factor"] == pytest.approx(0.15 / 0.11)

    mean_ratio_result = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter", prior_n=0.0,
        estimator="mean_ratio",
    )
    assert mean_ratio_result["raw_factor"] == pytest.approx(3.0)
    assert result["raw_factor"] != pytest.approx(mean_ratio_result["raw_factor"])


@pytest.mark.parametrize("fn_name", ["estimate_link_factor", "bootstrap_link_factor", "fit_factor_model"])
def test_unknown_estimator_raises(fn_name: str) -> None:
    cohort = _bigger_cohort()
    with pytest.raises(ValueError, match="unknown estimator"):
        if fn_name == "estimate_link_factor":
            estimate_link_factor(
                cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter",
                prior_n=20.0, estimator="bogus",
            )
        elif fn_name == "bootstrap_link_factor":
            bootstrap_link_factor(
                cohort, from_league="AAA", to_league="MLB", stat="woba", role="batter",
                prior_n=20.0, n_boot=5, seed=1, estimator="bogus",
            )
        else:
            fit_factor_model(cohort, n_boot=5, seed=1, estimator="bogus")


def test_fit_factor_model_ratio_of_means_reduces_small_pre_blowup() -> None:
    """A near-zero pre-move rate should not explode the factor under ratio_of_means."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "from_league": "AAA", "to_league": "MLB", "role": "pitcher",
                "pre_hr_fb": 0.11 + i * 0.001, "post_hr_fb": 0.11 + i * 0.001,
                "pre_k_pct": 0.2, "post_k_pct": 0.2,
                "pre_bb_pct": 0.08, "post_bb_pct": 0.08,
                "pre_era": 3.8, "post_era": 3.8,
                "pre_fip": 3.9, "post_fip": 3.9,
                "age_from": 27.0, "age_to": 27.0,
                "pre_ip": 40.0, "post_ip": 40.0,
            }
        )
    # One pathological pair with a tiny pre_adj rate that would dominate a
    # per-pair-ratio mean but is properly weighted down in ratio-of-means.
    rows.append(
        {
            "from_league": "AAA", "to_league": "MLB", "role": "pitcher",
            "pre_hr_fb": 0.001, "post_hr_fb": 0.11,
            "pre_k_pct": 0.2, "post_k_pct": 0.2,
            "pre_bb_pct": 0.08, "post_bb_pct": 0.08,
            "pre_era": 3.8, "post_era": 3.8,
            "pre_fip": 3.9, "post_fip": 3.9,
            "age_from": 27.0, "age_to": 27.0,
            "pre_ip": 40.0, "post_ip": 40.0,
        }
    )
    cohort = pd.DataFrame(rows)
    mean_ratio = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="hr_fb", role="pitcher",
        prior_n=0.0, estimator="mean_ratio",
    )
    ratio_of_means = estimate_link_factor(
        cohort, from_league="AAA", to_league="MLB", stat="hr_fb", role="pitcher",
        prior_n=0.0, estimator="ratio_of_means",
    )
    assert mean_ratio["raw_factor"] > ratio_of_means["raw_factor"]
    assert ratio_of_means["raw_factor"] == pytest.approx(1.0, abs=0.15)
    assert mean_ratio["raw_factor"] > 5.0
