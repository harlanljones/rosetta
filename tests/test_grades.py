"""Grades, WAR and dollars: formulas, bands, None-propagation (offline, fixture constants)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rosetta.models.factors import LeagueFactorModel
from rosetta.models.grades import (
    MlbConstants,
    dollars,
    fip_minus,
    grade_20_80,
    load_constants,
    war_batter,
    war_pitcher,
    wrc_plus,
)
from rosetta.paths import DATA_DIR
from rosetta.translate.api import translate_line

HEADER = (
    "season,woba_scale,lg_woba,lg_r_pa,runs_per_win,lg_fip,fip_constant,"
    "dollars_per_war,pitcher_repl_r9,source"
)


def full_constants() -> MlbConstants:
    """Arbitrary round fixture numbers; not published constants."""
    return MlbConstants(
        season=2024,
        woba_scale=1.25,
        lg_woba=0.320,
        lg_r_pa=0.11,
        runs_per_win=10.0,
        lg_fip=4.00,
        fip_constant=3.0,
        dollars_per_war=8.0,
        pitcher_repl_r9=0.5,
    )


def test_shipped_constants_table_is_header_only() -> None:
    lines = (DATA_DIR / "reference" / "mlb_constants.csv").read_text().strip().splitlines()
    assert lines == [HEADER]


def test_wrc_plus_formula_and_none() -> None:
    c = full_constants()
    assert wrc_plus(0.320, c) == pytest.approx(100.0)
    expected = 100 * (((0.370 - 0.320) / 1.25) + 0.11) / 0.11
    assert wrc_plus(0.370, c) == pytest.approx(expected)
    c.woba_scale = None
    assert wrc_plus(0.370, c) is None
    c = full_constants()
    c.lg_r_pa = None
    assert wrc_plus(0.370, c) is None


def test_fip_minus_formula_and_none() -> None:
    c = full_constants()
    assert fip_minus(3.0, c) == pytest.approx(75.0)
    c.lg_fip = None
    assert fip_minus(3.0, c) is None


def test_grade_20_80_median_sd_clip_round() -> None:
    ref = list(np.linspace(0, 100, 101))  # median 50, sd ~29.3
    sd = float(np.std(ref, ddof=1))
    assert grade_20_80(50.0, ref) == 50.0
    assert grade_20_80(50.0 + sd, ref) == 60.0
    assert grade_20_80(50.0 + 0.26 * sd, ref) == 55.0  # 52.6 -> 55
    assert grade_20_80(50.0 + 0.24 * sd, ref) == 50.0  # 52.4 -> 50
    assert grade_20_80(1e6, ref) == 80.0
    assert grade_20_80(-1e6, ref) == 20.0
    # inverted for lower-is-better
    assert grade_20_80(50.0 - sd, ref, higher_is_better=False) == 60.0
    assert grade_20_80(50.0 + sd, ref, higher_is_better=False) == 40.0


def test_grade_20_80_degenerate_reference() -> None:
    assert grade_20_80(1.0, []) is None
    assert grade_20_80(1.0, [2.0]) is None
    assert grade_20_80(1.0, [2.0, 2.0, 2.0]) is None
    assert grade_20_80(None, [1.0, 2.0]) is None


def test_war_batter_formula() -> None:
    c = full_constants()
    notes: list[str] = []
    war = war_batter(0.360, 600, "SS", 5.0, c, notes)
    batting = (0.360 - 0.320) / 1.25 * 600
    pos = 7.5 * 600 / 700
    repl = 20.0
    assert war == pytest.approx((batting + pos + 5.0 + repl) / 10.0)
    assert notes == []


def test_war_batter_position_table() -> None:
    c = full_constants()
    base = war_batter(0.320, 700, "DH", 0.0, c)
    assert base == pytest.approx((-17.5 + 20.0 * 700 / 600) / 10.0)
    for pos, adj in {"C": 12.5, "2B": 2.5, "3B": 2.5, "CF": 2.5, "LF": -7.5, "RF": -7.5, "1B": -12.5}.items():
        assert war_batter(0.320, 700, pos, 0.0, c) == pytest.approx((adj + 20.0 * 700 / 600) / 10.0)


def test_war_batter_notes_for_unknown_position_and_missing_def() -> None:
    c = full_constants()
    notes: list[str] = []
    war = war_batter(0.320, 600, "XX", None, c, notes)
    assert war == pytest.approx(20.0 / 10.0)
    assert any("position" in n.lower() for n in notes)
    assert any("def_runs" in n for n in notes)
    notes = []
    war_batter(0.320, 600, "SS", float("nan"), c, notes)
    assert any("def_runs" in n for n in notes)


def test_war_batter_none_when_constants_missing() -> None:
    for attr in ("woba_scale", "lg_woba", "runs_per_win"):
        c = full_constants()
        setattr(c, attr, None)
        assert war_batter(0.340, 600, "SS", 0.0, c) is None


def test_war_pitcher_formula_and_none() -> None:
    c = full_constants()
    war = war_pitcher(3.50, 180, c)
    assert war == pytest.approx(((4.0 - 3.5) / 9 * 180 + 180 / 9 * 0.5) / 10.0)
    assert war_pitcher(3.50, 180, c, repl_runs_per_9=1.0) == pytest.approx(
        ((4.0 - 3.5) / 9 * 180 + 180 / 9 * 1.0) / 10.0
    )
    c.pitcher_repl_r9 = None
    assert war_pitcher(3.50, 180, c) is None  # no baked default
    assert war_pitcher(3.50, 180, c, repl_runs_per_9=0.5) is not None
    c = full_constants()
    c.runs_per_win = None
    assert war_pitcher(3.50, 180, c) is None


def test_dollars() -> None:
    c = full_constants()
    assert dollars(2.5, c) == pytest.approx(20.0)
    assert dollars(None, c) is None
    c.dollars_per_war = None
    assert dollars(2.5, c) is None


# ---- load_constants -----------------------------------------------------


def _write_snapshots(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    bat = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2023],
            "woba": [0.300, 0.400, 0.999, 0.350, 0.900],
            "pa": [100.0, 300.0, 5000.0, 400.0, 600.0],
            "is_synthetic": [False, False, True, False, False],
        }
    )
    bat.to_csv(root / "mlb_batters.csv", index=False)
    pit = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "fip": [3.0, 5.0, 0.1, 4.0],
            "ip": [100.0, 50.0, 900.0, 150.0],
            "is_synthetic": [False, False, True, False],
        }
    )
    pit.to_csv(root / "mlb_pitchers.csv", index=False)


def test_load_constants_computes_lg_woba_and_lg_fip(tmp_path: Path) -> None:
    _write_snapshots(tmp_path / "snap")
    ref = tmp_path / "ref.csv"
    ref.write_text(HEADER + "\n")
    c = load_constants(2024, tmp_path / "snap", ref)
    assert c.lg_woba == pytest.approx((0.3 * 100 + 0.4 * 300 + 0.35 * 400) / 800)
    assert c.lg_fip == pytest.approx((3 * 100 + 5 * 50 + 4 * 150) / 300)
    assert c.woba_scale is None and c.lg_r_pa is None and c.runs_per_win is None
    assert c.dollars_per_war is None and c.pitcher_repl_r9 is None
    assert c.ref_woba == [0.4, 0.35]  # pa >= 300, real only
    assert c.ref_fip == [3.0, 4.0]  # ip >= 100, real only


def test_load_constants_table_values_win(tmp_path: Path) -> None:
    _write_snapshots(tmp_path / "snap")
    ref = tmp_path / "ref.csv"
    ref.write_text(HEADER + "\n2024,1.2,0.31,0.12,9.5,4.1,3.1,,0.4,test fixture\n")
    c = load_constants(2024, tmp_path / "snap", ref)
    assert c.woba_scale == 1.2 and c.lg_woba == 0.31 and c.lg_fip == 4.1
    assert c.runs_per_win == 9.5 and c.pitcher_repl_r9 == 0.4
    assert c.dollars_per_war is None  # blank cell stays None
    assert c.source == "test fixture"


def test_load_constants_missing_files_all_none(tmp_path: Path) -> None:
    c = load_constants(2024, tmp_path / "nosnap", tmp_path / "noref.csv")
    assert c.lg_woba is None and c.lg_fip is None and c.ref_woba == []


# ---- translate_line integration -----------------------------------------


def _model() -> LeagueFactorModel:
    return LeagueFactorModel(links={})  # identity links, nonzero residual bands


BATTER = {
    "player_id": "b1",
    "player_name": "Bat Ter",
    "season": 2024,
    "league": "NPB",
    "role": "batter",
    "age": 27,
    "pa": 500,
    "woba": 0.360,
    "bb_pct": 0.09,
    "k_pct": 0.18,
    "iso": 0.18,
    "babip": 0.31,
    "hr_pct": 0.04,
    "position": "SS",
}
PITCHER = {
    "player_id": "p1",
    "player_name": "Pit Cher",
    "season": 2024,
    "league": "NPB",
    "role": "pitcher",
    "age": 27,
    "ip": 150,
    "g": 25,
    "gs": 25,
    "k_pct": 0.25,
    "bb_pct": 0.07,
    "hr_fb": 0.10,
    "era": 3.2,
    "fip": 3.4,
}


def _fixture_constants() -> MlbConstants:
    c = full_constants()
    c.ref_woba = [0.290, 0.310, 0.320, 0.330, 0.350]
    c.ref_fip = [3.2, 3.8, 4.0, 4.2, 4.8]
    return c


def test_translate_line_without_constants_leaves_new_fields_none() -> None:
    res = translate_line(BATTER, model=_model())
    for name in (
        "position", "wrc_plus", "wrc_plus_low", "wrc_plus_high", "grade", "grade_low",
        "grade_high", "war", "war_low", "war_high", "dollars", "dollars_low", "dollars_high",
        "fip_minus", "fip_minus_low", "fip_minus_high",
    ):  # fmt: skip
        assert getattr(res, name) is None


def test_translate_line_existing_fields_unchanged_with_constants() -> None:
    m = _model()
    for line in (BATTER, PITCHER):
        plain = translate_line(line, model=m)
        graded = translate_line(line, model=m, constants=_fixture_constants())
        a = plain.model_dump()
        b = graded.model_dump()
        grade_fields = {
            f"{n}{s}"
            for n in ("wrc_plus", "fip_minus", "grade", "war", "dollars")
            for s in ("", "_low", "_high")
        } | {"position"}
        for key in a:
            if key in grade_fields:
                continue
            if key == "notes":
                assert b["notes"][: len(a["notes"])] == a["notes"]
                continue
            assert a[key] == b[key], key


def test_translate_line_batter_grades_and_band_order() -> None:
    c = _fixture_constants()
    res = translate_line(BATTER, model=_model(), constants=c)
    assert res.position == "SS"
    assert res.woba_low < res.woba < res.woba_high
    assert res.wrc_plus == pytest.approx(wrc_plus(res.woba, c))
    assert res.wrc_plus_low < res.wrc_plus < res.wrc_plus_high
    assert res.grade_low <= res.grade <= res.grade_high
    assert res.war_low < res.war < res.war_high
    assert res.war == pytest.approx(war_batter(res.woba, 600, "SS", 0.0, c))
    assert res.dollars == pytest.approx(res.war * 8.0)
    assert res.dollars_low < res.dollars_high
    assert any("def_runs" in n for n in res.notes)


def test_translate_line_pitcher_grades_and_inverted_bands() -> None:
    c = _fixture_constants()
    res = translate_line(PITCHER, model=_model(), constants=c)
    assert res.fip_low < res.fip < res.fip_high
    assert res.fip_minus == pytest.approx(100 * res.fip / 4.0)
    assert res.fip_minus_low < res.fip_minus < res.fip_minus_high
    # grade / WAR / dollars improve as FIP falls: low comes from fip_high
    assert res.grade_low <= res.grade <= res.grade_high
    assert res.war_low < res.war < res.war_high
    assert res.war_high == pytest.approx(war_pitcher(res.fip_low, 170, c))
    assert res.war_low == pytest.approx(war_pitcher(res.fip_high, 170, c))
    assert res.dollars_low < res.dollars < res.dollars_high
    assert res.position is None


def test_pitcher_default_innings_by_role() -> None:
    c = _fixture_constants()
    reliever = {**PITCHER, "g": 60, "gs": 2}
    rp = translate_line(reliever, model=_model(), constants=c)
    assert rp.war == pytest.approx(war_pitcher(rp.fip, 65, c))
    sp = translate_line(PITCHER, model=_model(), constants=c)
    assert sp.war == pytest.approx(war_pitcher(sp.fip, 170, c))
    forced = translate_line(PITCHER, model=_model(), constants=c, target_pitch_role="RP")
    assert forced.war is not None
    no_split = {k: v for k, v in PITCHER.items() if k not in {"g", "gs"}}
    unk = translate_line(no_split, model=_model(), constants=c)
    assert unk.war == pytest.approx(war_pitcher(unk.fip, 170, c))


def test_none_propagation_through_translate_line() -> None:
    c = _fixture_constants()
    c.woba_scale = None
    c.dollars_per_war = None
    c.pitcher_repl_r9 = None
    bat = translate_line(BATTER, model=_model(), constants=c)
    assert bat.wrc_plus is None and bat.wrc_plus_low is None and bat.wrc_plus_high is None
    assert bat.war is None and bat.dollars is None and bat.dollars_low is None
    assert bat.grade is not None  # falls back to wOBA reference
    pit = translate_line(PITCHER, model=_model(), constants=c)
    assert pit.fip_minus is not None
    assert pit.war is None and pit.dollars is None
    assert not math.isnan(pit.fip_minus)


def test_batter_grade_uses_woba_reference_when_wrc_plus_not_computable() -> None:
    c = _fixture_constants()
    c.lg_r_pa = None
    res = translate_line(BATTER, model=_model(), constants=c)
    assert res.wrc_plus is None
    assert res.grade == grade_20_80(res.woba, c.ref_woba)
