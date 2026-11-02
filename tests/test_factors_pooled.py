"""Pooled international link factors: pooling, direction term, chain shrink target, thin links."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rosetta.models import factors as F
from rosetta.models.factors import (
    LeagueFactorModel,
    chain_target,
    fit_factor_model,
    fit_pooled_link,
    invert_transfer_rows,
    pool_pair_sample,
)

FIXTURE = Path(__file__).parent / "fixtures" / "factors_aaa_mlb_frozen.json"


def _approx_tree(actual, expected, path=""):
    """Recursive comparison: floats approx (rel 1e-9), everything else exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for k in expected:
            _approx_tree(actual[k], expected[k], f"{path}/{k}")
    elif isinstance(expected, list):
        assert list(actual) == expected, path
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), path
    else:
        assert actual == expected, path


def test_aaa_mlb_only_cohort_matches_frozen_pre_change_output() -> None:
    """D1 regression: expected values were captured from the pre-pooling code."""
    data = json.loads(FIXTURE.read_text())
    cohort = pd.DataFrame(data["cohort"])
    model = fit_factor_model(cohort, n_boot=data["n_boot"], seed=data["seed"])
    _approx_tree(json.loads(json.dumps(model.to_dict())), data["expected"])


# ---------------------------------------------------------------------------
# synthetic international cohorts with known export / import factors
# ---------------------------------------------------------------------------

BAT_STATS = ("bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba")


def _bat_row(
    pid: str,
    frm: str,
    to: str,
    pre_woba: float,
    post_woba: float,
    *,
    pa: float = 400.0,
    direction: str | None = None,
    age_from: float = 27.0,
    age_to: float = 27.0,
    holdout: bool = False,
    **extra,
) -> dict:
    row = {
        "player_id": pid,
        "role": "batter",
        "from_league": frm,
        "to_league": to,
        "from_season": 2020,
        "to_season": 2021,
        "age_from": age_from,
        "age_to": age_to,
        "holdout": holdout,
        "pre_pa": pa,
        "post_pa": pa,
    }
    if direction:
        row["direction"] = direction
    for s in BAT_STATS:
        row[f"pre_{s}"] = pre_woba
        row[f"post_{s}"] = post_woba
    row.update(extra)
    return row


def _wobas(n: int, level: float) -> list[float]:
    return [level + 0.01 * ((i % 5) - 2) for i in range(n)]


def _npb_mlb_cohort(n_exp: int = 8, n_imp: int = 12, f_exp: float = 0.90, f_imp: float = 1.20):
    """NPB->MLB exports with ratio f_exp; MLB->NPB imports whose NPB/MLB ratio is f_imp.

    Import rows are recorded MLB->NPB, so post/pre = 1/f_imp.
    """
    rows = []
    for i, w in enumerate(_wobas(n_exp, 0.330)):
        rows.append(_bat_row(f"e{i}", "NPB", "MLB", w, w * f_exp, direction="export"))
    for i, w in enumerate(_wobas(n_imp, 0.300)):  # MLB-side pre rate
        rows.append(_bat_row(f"i{i}", "MLB", "NPB", w, w / f_imp, direction="import"))
    return pd.DataFrame(rows)


def test_invert_swaps_pre_post_ages_leagues_and_keeps_weight() -> None:
    df = pd.DataFrame(
        [
            _bat_row(
                "x",
                "MLB",
                "NPB",
                0.31,
                0.35,
                pa=300.0,
                direction="import",
                age_from=29.0,
                age_to=30.0,
                pre_age_source="birth_year",
                post_age_source="imputed",
            )
        ]
    )
    df["post_pa"] = 500.0
    inv = invert_transfer_rows(df).iloc[0]
    assert (inv.from_league, inv.to_league) == ("NPB", "MLB")
    assert (inv.pre_woba, inv.post_woba) == (0.35, 0.31)
    assert (inv.pre_pa, inv.post_pa) == (500.0, 300.0)
    assert (inv.age_from, inv.age_to) == (30.0, 29.0)
    assert (inv.pre_age_source, inv.post_age_source) == ("imputed", "birth_year")
    assert inv.direction == "export"
    assert min(inv.pre_pa, inv.post_pa) == 300.0  # weight unchanged


def test_pooled_sample_orients_everything_a_to_b() -> None:
    cohort = _npb_mlb_cohort(3, 2)
    sample = pool_pair_sample(cohort, "NPB", "MLB", "batter")
    assert len(sample) == 5
    assert set(sample["from_league"]) == {"NPB"} and set(sample["to_league"]) == {"MLB"}
    assert int(sample["_forward"].sum()) == 3


def test_pooled_estimate_import_export_and_direction_term() -> None:
    cohort = _npb_mlb_cohort(n_exp=8, n_imp=12, f_exp=0.90, f_imp=1.20)
    model = fit_factor_model(cohort, n_boot=30, seed=1)
    node = model.links["batter"]["NPB->MLB"]["woba"]
    assert node["n_export"] == 8 and node["n_import"] == 12 and node["n"] == 20
    assert node["factor_export"] == pytest.approx(0.90, rel=1e-9)
    assert node["factor_import"] == pytest.approx(1.20, rel=1e-9)
    assert node["direction_gap"] == pytest.approx(math.log(0.90 / 1.20), rel=1e-9)
    # ratio of means on the pooled sample (equal weights, inverted import rows):
    # sum(post)/sum(pre) with pre/post taken in NPB->MLB orientation.
    exp_w, imp_w = _wobas(8, 0.330), _wobas(12, 0.300)
    post = [w * 0.90 for w in exp_w] + [w for w in imp_w]
    pre = list(exp_w) + [w / 1.20 for w in imp_w]
    raw = sum(post) / sum(pre)
    assert node["raw_factor"] == pytest.approx(raw, rel=1e-9)
    # shrunk toward 1.0 (no AAA chain here) in log space
    s = 20 / (20 + F.DEFAULT_PRIOR_N["woba"])
    assert node["shrink"] == pytest.approx(s)
    assert node["shrink_target"] == 1.0
    assert node["factor"] == pytest.approx(math.exp(s * math.log(raw)), rel=1e-9)
    assert node["low"] < node["factor"] < node["high"]
    assert node["thin"] is False


def test_reverse_node_is_reciprocal_with_mirrored_direction_fields() -> None:
    cohort = _npb_mlb_cohort()
    model = fit_factor_model(cohort, n_boot=30, seed=1)
    fwd = model.links["batter"]["NPB->MLB"]["woba"]
    rev = model.links["batter"]["MLB->NPB"]["woba"]
    assert rev["factor"] == pytest.approx(1 / fwd["factor"])
    assert rev["raw_factor"] == pytest.approx(1 / fwd["raw_factor"])
    assert rev["low"] == pytest.approx(1 / fwd["high"])
    assert rev["high"] == pytest.approx(1 / fwd["low"])
    assert rev["n"] == fwd["n"]
    assert rev["n_export"] == fwd["n_export"] and rev["n_import"] == fwd["n_import"]
    assert rev["factor_export"] == pytest.approx(1 / fwd["factor_export"])
    assert rev["factor_import"] == pytest.approx(1 / fwd["factor_import"])
    assert rev["direction_gap"] == pytest.approx(-fwd["direction_gap"])
    # get_factor / get_interval agree with reciprocity in either direction
    assert model.get_factor("batter", "MLB", "NPB", "woba") == pytest.approx(
        1 / model.get_factor("batter", "NPB", "MLB", "woba")
    )
    p, lo, hi, _ = model.get_interval("batter", "MLB", "NPB", "woba")
    p2, lo2, hi2, _ = model.get_interval("batter", "NPB", "MLB", "woba")
    assert (p, lo, hi) == pytest.approx((1 / p2, 1 / hi2, 1 / lo2))


def test_pooling_is_symmetric_in_input_orientation() -> None:
    """Only-reverse rows still produce the A->B orientation, inverted."""
    cohort = _npb_mlb_cohort(n_exp=0, n_imp=10, f_imp=1.25)
    model = fit_factor_model(cohort, n_boot=20, seed=1)
    node = model.links["batter"]["NPB->MLB"]["woba"]
    assert node["n_export"] == 0 and node["n_import"] == 10
    assert node["factor_export"] is None and node["direction_gap"] is None
    assert node["factor_import"] == pytest.approx(1.25, rel=1e-9)
    assert node["raw_factor"] == pytest.approx(1.25, rel=1e-9)


def test_holdout_rows_are_excluded_before_pooling() -> None:
    cohort = _npb_mlb_cohort(n_exp=6, n_imp=6)
    extra = pd.DataFrame(
        [_bat_row("h", "NPB", "MLB", 0.33, 0.10, holdout=True, direction="export")]
    )
    both = pd.concat([cohort, extra], ignore_index=True)
    a = fit_factor_model(cohort, n_boot=10, seed=1).links["batter"]["NPB->MLB"]["woba"]
    b = fit_factor_model(both, n_boot=10, seed=1).links["batter"]["NPB->MLB"]["woba"]
    assert b["n"] == a["n"] and b["raw_factor"] == pytest.approx(a["raw_factor"])


def test_shrink_toward_chain_target_through_aaa() -> None:
    rows = []
    # NPB->AAA at 0.95 (batters, 12 movers), AAA->MLB at 0.90 (12 movers)
    for i, w in enumerate(_wobas(12, 0.330)):
        rows.append(_bat_row(f"a{i}", "NPB", "AAA", w, w * 0.95))
    for i, w in enumerate(_wobas(12, 0.320)):
        rows.append(_bat_row(f"b{i}", "AAA", "MLB", w, w * 0.90))
    # thin NPB->MLB link (3 exports) with raw ratio 0.60: strongly lower than chain
    for i, w in enumerate(_wobas(3, 0.330)):
        rows.append(_bat_row(f"c{i}", "NPB", "MLB", w, w * 0.60, direction="export"))
    cohort = pd.DataFrame(rows)
    model = fit_factor_model(cohort, n_boot=20, seed=3, era_floor=None)
    links = model.links["batter"]
    npb_aaa = links["NPB->AAA"]["woba"]["factor"]
    aaa_mlb = links["AAA->MLB"]["woba"]["factor"]
    node = links["NPB->MLB"]["woba"]
    assert node["shrink_target"] == pytest.approx(npb_aaa * aaa_mlb)
    assert node["shrink_target"] < 1.0
    s = 3 / (3 + F.DEFAULT_PRIOR_N["woba"])
    expected = math.exp(
        math.log(node["shrink_target"]) + s * (math.log(0.60) - math.log(node["shrink_target"]))
    )
    assert node["factor"] == pytest.approx(expected, rel=1e-9)
    # adjacent links shrink toward 1.0 and carry target 1.0
    assert links["NPB->AAA"]["woba"]["shrink_target"] == 1.0
    # reverse side stored too
    assert links["MLB->NPB"]["woba"]["shrink_target"] == pytest.approx(1 / node["shrink_target"])


def test_shrink_target_is_one_when_chain_incomplete() -> None:
    rows = [_bat_row(f"a{i}", "NPB", "AAA", w, w * 0.95) for i, w in enumerate(_wobas(12, 0.33))]
    rows += [
        _bat_row(f"c{i}", "NPB", "MLB", w, w * 0.8, direction="export")
        for i, w in enumerate(_wobas(6, 0.33))
    ]
    model = fit_factor_model(pd.DataFrame(rows), n_boot=10, seed=3, era_floor=None)
    assert model.links["batter"]["NPB->MLB"]["woba"]["shrink_target"] == 1.0


def test_chain_target_helper_uses_reverse_node_and_requires_n() -> None:
    links = {
        "batter": {
            "AAA->MLB": {"woba": {"factor": 0.9, "n": 5.0}},
            "AAA->NPB": {"woba": {"factor": 1.0 / 0.95, "n": 4.0}},
        }
    }
    chain = F.LEAGUE_CHAIN
    assert chain_target(links, chain, "batter", "woba", "NPB", "MLB") == pytest.approx(0.95 * 0.9)
    links["batter"]["AAA->MLB"]["woba"]["n"] = 0.0
    assert chain_target(links, chain, "batter", "woba", "NPB", "MLB") == 1.0
    assert chain_target(links, chain, "batter", "woba", "NPB", "AAA") == 1.0  # adjacent


def test_thin_link_gets_residual_band_and_flag() -> None:
    cohort = _npb_mlb_cohort(n_exp=2, n_imp=1)
    # add noise so resid_sd > 0
    cohort.loc[0, "post_woba"] *= 1.1
    model = fit_factor_model(cohort, n_boot=20, seed=1)
    node = model.links["batter"]["NPB->MLB"]["woba"]
    assert node["thin"] is True and node["n"] == 3
    assert node["resid_sd"] > 0
    assert node["low"] < node["factor"] < node["high"]
    # documented transform: factor * exp(-/+ z80 * (resid_sd / mean post) / sqrt(n))
    post = np.array(
        [cohort.loc[0, "post_woba"], cohort.loc[1, "post_woba"], cohort.loc[2, "pre_woba"]]
    )
    half = F.Z80 * node["resid_sd"] / float(np.mean(post)) / math.sqrt(3)
    assert node["high"] == pytest.approx(node["factor"] * math.exp(half), rel=1e-6)
    assert node["low"] == pytest.approx(node["factor"] * math.exp(-half), rel=1e-6)


def test_bootstrap_resamples_pooled_rows() -> None:
    """Pooled n>=5 -> bootstrap band; same-seed fit is deterministic and wider than point."""
    cohort = _npb_mlb_cohort(n_exp=3, n_imp=3)  # 3 + 3 = 6 pooled rows, neither side alone has 5
    cohort.loc[0, "post_woba"] *= 1.2
    a = fit_factor_model(cohort, n_boot=40, seed=9).links["batter"]["NPB->MLB"]["woba"]
    b = fit_factor_model(cohort, n_boot=40, seed=9).links["batter"]["NPB->MLB"]["woba"]
    assert a["thin"] is False and a == b
    assert a["low"] < a["high"]


def test_aaa_mlb_link_unaffected_by_international_rows_in_cohort() -> None:
    data = json.loads(FIXTURE.read_text())
    base = pd.DataFrame(data["cohort"])
    mixed = pd.concat([base, _npb_mlb_cohort()], ignore_index=True)
    m0 = fit_factor_model(base, n_boot=20, seed=7)
    m1 = fit_factor_model(mixed, n_boot=20, seed=7)
    for role in ("batter", "pitcher"):
        assert m1.links[role]["AAA->MLB"] == m0.links[role]["AAA->MLB"]


def test_imputed_age_uses_skip_when_available_and_falls_back_otherwise(monkeypatch) -> None:
    calls = []

    def with_skip(rate, stat, age_from, age_to, role="batter", skip=False):
        calls.append(skip)
        return rate if skip else rate + 0.5

    monkeypatch.setattr(F, "age_adjust_rate", with_skip)
    assert F._age_adjust(0.3, "woba", 25.0, 30.0, "batter", imputed=True) == 0.3
    assert F._age_adjust(0.3, "woba", 25.0, 30.0, "batter", imputed=False) == 0.8
    assert calls == [True, False]

    def without_skip(rate, stat, age_from, age_to, role="batter"):
        return rate + 0.5

    monkeypatch.setattr(F, "age_adjust_rate", without_skip)
    assert F._age_adjust(0.3, "woba", 25.0, 30.0, "batter", imputed=True) == 0.8


def test_imputed_rows_reach_age_adjust_with_skip(monkeypatch) -> None:
    seen = []

    def spy(rate, stat, age_from, age_to, role="batter", skip=False):
        seen.append(skip)
        return rate

    monkeypatch.setattr(F, "age_adjust_rate", spy)
    cohort = _npb_mlb_cohort(n_exp=2, n_imp=2)
    cohort["pre_age_source"] = "imputed"
    cohort["post_age_source"] = "birth_year"
    fit_pooled_link(cohort, a="NPB", b="MLB", stat="woba", role="batter", prior_n=35.0, n_boot=5)
    assert seen and all(seen)


def test_fit_pooled_link_returns_none_without_rows() -> None:
    cohort = _npb_mlb_cohort(2, 2)
    assert (
        fit_pooled_link(cohort, a="KBO", b="MLB", stat="woba", role="batter", prior_n=35.0) is None
    )


def test_pitcher_pooling_and_weights_use_ip() -> None:
    rows = []
    for i in range(6):
        for frm, to, mult in (("NPB", "MLB", 1.25), ("MLB", "NPB", 1 / 1.25)):
            rows.append(
                {
                    "player_id": f"{frm}{i}",
                    "role": "pitcher",
                    "from_league": frm,
                    "to_league": to,
                    "from_season": 2020,
                    "to_season": 2021,
                    "age_from": 27.0,
                    "age_to": 27.0,
                    "holdout": False,
                    "pre_ip": 100.0,
                    "post_ip": 80.0,
                    **{f"pre_{s}": 3.0 for s in F.PITCHER_STATS},
                    **{f"post_{s}": 3.0 * mult for s in F.PITCHER_STATS},
                }
            )
    model = fit_factor_model(pd.DataFrame(rows), n_boot=10, seed=1)
    node = model.links["pitcher"]["NPB->MLB"]["fip"]
    assert node["raw_factor"] == pytest.approx(1.25)
    assert node["factor_export"] == pytest.approx(1.25) and node["factor_import"] == pytest.approx(
        1.25
    )
    assert node["direction_gap"] == pytest.approx(0.0, abs=1e-12)
    assert node["weight"] == pytest.approx(12 * 80.0)


def test_roundtrip_json_serializable() -> None:
    model = fit_factor_model(_npb_mlb_cohort(2, 1), n_boot=5, seed=1)
    again = LeagueFactorModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.links == model.links
