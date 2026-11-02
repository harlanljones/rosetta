"""International (NPB/KBO -> MLB) rolling validation: leakage, scoring, gate, report, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import rosetta.backtest.historical as historical
from rosetta import cli
from rosetta.backtest.historical import (
    international_ship_gate,
    render_international_markdown,
    run_international_rolling,
    write_international_report,
)
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

EXPORT_SEASONS = (2019, 2020, 2021, 2022, 2023)  # MLB season of each export cohort
N_EXPORT = 12
N_IMPORT = 4
BAT_FACTOR = 0.85
PIT_FACTOR = 1.15


def _bat(pid: str, season: int, league: str, woba: float, *, team: str = "T1") -> dict:
    row = dict.fromkeys(HITTER_COLUMNS, 0.0)
    row.update(
        player_id=pid, player_name=pid, season=season, league=league, team=team, role="batter",
        age=26.0, pa=400.0, g=100.0, bb_pct=woba / 4, k_pct=0.2, iso=woba / 2, babip=woba,
        hr_pct=woba / 10, avg=0.27, obp=0.34, slg=0.43, woba=woba, source="fixture",
        is_synthetic=False,
    )
    return row


def _pit(pid: str, season: int, league: str, fip: float, *, team: str = "T1") -> dict:
    row = dict.fromkeys(PITCHER_COLUMNS, 0.0)
    row.update(
        player_id=pid, player_name=pid, season=season, league=league, team=team, role="pitcher",
        age=26.0, ip=120.0, g=25.0, gs=25.0, k_pct=0.2, bb_pct=0.08, hr_fb=fip / 40, era=fip,
        fip=fip, source="fixture", is_synthetic=False,
    )
    return row


def _transfer(pid: str, role: str, frm: str, to: str, from_season: int) -> dict:
    return {
        "player_id": pid, "player_name": pid, "role": role, "from_league": frm, "to_league": to,
        "from_season": from_season, "to_season": from_season + 1, "age_from": 26.0, "age_to": 27.0,
        "holdout": False,
    }


def _build_snapshots(root: Path, *, mlb_shift: dict[int, float] | None = None) -> Path:
    """Exports NPB(s-1)->MLB(s) with MLB = 0.85 x NPB woba / 1.15 x NPB fip, plus a few imports."""
    mlb_shift = mlb_shift or {}
    npb_b, mlb_b, npb_p, mlb_p, transfers = [], [], [], [], []
    for s in EXPORT_SEASONS:
        shift = mlb_shift.get(s, 0.0)
        for j in range(N_EXPORT):
            woba = 0.330 + 0.004 * j
            fip = 3.0 + 0.1 * j
            pid = f"e-{s}-{j}"
            npb_b.append(_bat(pid, s - 1, "NPB", woba))
            mlb_b.append(_bat(pid, s, "MLB", woba * BAT_FACTOR + shift))
            npb_p.append(_pit(f"p{pid}", s - 1, "NPB", fip))
            mlb_p.append(_pit(f"p{pid}", s, "MLB", fip * PIT_FACTOR + shift * 10))
            transfers.append(_transfer(pid, "batter", "NPB", "MLB", s - 1))
            transfers.append(_transfer(f"p{pid}", "pitcher", "NPB", "MLB", s - 1))
        for j in range(N_IMPORT):
            woba = 0.310 + 0.004 * j
            fip = 4.0 + 0.1 * j
            pid = f"i-{s}-{j}"
            mlb_b.append(_bat(pid, s - 1, "MLB", woba))
            npb_b.append(_bat(pid, s, "NPB", woba / BAT_FACTOR))
            mlb_p.append(_pit(f"p{pid}", s - 1, "MLB", fip))
            npb_p.append(_pit(f"p{pid}", s, "NPB", fip / PIT_FACTOR))
            transfers.append(_transfer(pid, "batter", "MLB", "NPB", s - 1))
            transfers.append(_transfer(f"p{pid}", "pitcher", "MLB", "NPB", s - 1))
    pd.DataFrame(npb_b, columns=HITTER_COLUMNS).to_csv(root / "npb_batters.csv", index=False)
    pd.DataFrame(mlb_b, columns=HITTER_COLUMNS).to_csv(root / "mlb_batters.csv", index=False)
    pd.DataFrame(npb_p, columns=PITCHER_COLUMNS).to_csv(root / "npb_pitchers.csv", index=False)
    pd.DataFrame(mlb_p, columns=PITCHER_COLUMNS).to_csv(root / "mlb_pitchers.csv", index=False)
    pd.DataFrame(transfers).to_csv(root / "transfers.csv", index=False)
    return root


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> Path:
    return _build_snapshots(tmp_path_factory.mktemp("intl"))


@pytest.fixture(scope="module")
def result(snapshots: Path) -> dict:
    return run_international_rolling(
        snapshots, league="NPB", target_seasons=range(2019, 2024), n_boot=5, seed=1
    )


def _row(rows: list[dict], role: str, stat: str) -> dict:
    return next(r for r in rows if r["role"] == role and r["stat"] == stat)


def test_target_without_training_data_is_skipped_not_an_error(result: dict) -> None:
    assert "2019" not in result["by_target"]
    skipped = {item["target_season"]: item["reason"] for item in result["metadata"]["skipped"]}
    assert 2019 in skipped and "to_season < 2019" in skipped[2019]
    assert set(result["by_target"]) == {"2020", "2021", "2022", "2023"}


def test_target_with_no_exports_is_skipped(tmp_path: Path) -> None:
    root = _build_snapshots(tmp_path)
    payload = run_international_rolling(
        root, league="NPB", target_seasons=[2020, 2030], n_boot=3, seed=1
    )
    reasons = {i["target_season"]: i["reason"] for i in payload["metadata"]["skipped"]}
    assert "exports" in reasons[2030]
    assert list(payload["by_target"]) == ["2020"]


def test_all_targets_skipped_returns_insufficient_gate(tmp_path: Path) -> None:
    root = _build_snapshots(tmp_path)
    payload = run_international_rolling(root, league="NPB", target_seasons=[2030], n_boot=3)
    assert payload["by_target"] == {} and payload["pooled"] == []
    assert payload["gate"]["verdict"] == "INSUFFICIENT"


def test_unknown_league_rejected(snapshots: Path) -> None:
    with pytest.raises(ValueError, match="league"):
        run_international_rolling(snapshots, league="AAA", target_seasons=[2020])


def test_training_uses_only_earlier_destination_seasons(result: dict) -> None:
    for season, target in result["by_target"].items():
        assert target["latest_training_to_season"] < int(season)
        assert {d["to_season"] for d in target["detail"]} == {int(season)}
        assert {d["from_season"] for d in target["detail"]} == {int(season) - 1}


def test_future_outcomes_do_not_change_earlier_targets(tmp_path: Path, snapshots: Path) -> None:
    base = run_international_rolling(snapshots, league="NPB", target_seasons=[2021], n_boot=5, seed=1)
    shifted_root = _build_snapshots(tmp_path, mlb_shift={2022: 0.05, 2023: -0.05})
    other = run_international_rolling(
        shifted_root, league="NPB", target_seasons=[2021], n_boot=5, seed=1
    )
    assert base["by_target"]["2021"]["metrics"] == other["by_target"]["2021"]["metrics"]
    assert base["by_target"]["2021"]["links"] == other["by_target"]["2021"]["links"]


def test_model_beats_carry_forward_baseline_when_factor_is_real(result: dict) -> None:
    woba = _row(result["pooled"], "batter", "woba")
    fip = _row(result["pooled"], "pitcher", "fip")
    # Pooled n is the sum over scored targets (4 targets x 12 exports).
    assert woba["n"] == fip["n"] == 4 * N_EXPORT
    assert woba["mae"] < woba["baseline_mae"]
    assert fip["mae"] < fip["baseline_mae"]
    assert 0.0 <= woba["coverage_80"] <= 1.0
    assert woba["league"] == "NPB"
    per_target = [r for r in result["summary"] if r["stat"] == "woba" and r["role"] == "batter"]
    assert sorted(r["target_season"] for r in per_target) == [2020, 2021, 2022, 2023]
    assert all(r["n"] == N_EXPORT for r in per_target)


def test_baseline_is_the_untouched_foreign_rate(result: dict) -> None:
    for target in result["by_target"].values():
        for d in target["detail"]:
            if d["role"] == "batter" and d["stat"] == "woba":
                j = int(d["player_id"].rsplit("-", 1)[1])
                assert d["prior"] == pytest.approx(0.330 + 0.004 * j)


def test_link_report_has_pooled_factor_counts_and_direction_gap(result: dict) -> None:
    links = [r for r in result["by_target"]["2022"]["links"]]
    assert {(r["link"], r["role"], r["stat"]) for r in links} == {
        ("NPB->MLB", "batter", "woba"), ("NPB->MLB", "pitcher", "fip"),
        ("NPB->AAA", "batter", "woba"), ("NPB->AAA", "pitcher", "fip"),
    }
    bat = next(r for r in links if r["link"] == "NPB->MLB" and r["role"] == "batter")
    # 2019..2021 destination seasons: 3 x 12 exports and 3 x 4 imports.
    assert bat["fitted"] and bat["n_export"] == 36 and bat["n_import"] == 12
    assert bat["n"] == 48
    assert bat["direction_gap"] == pytest.approx(0.0, abs=0.05)
    assert 0.8 < bat["factor"] < 1.0
    # No AAA data in the fixture: reported as unfitted with zero counts.
    aaa = next(r for r in links if r["link"] == "NPB->AAA")
    assert not aaa["fitted"] and aaa["n"] == 0 and aaa["factor"] is None
    assert len(result["links"]) == 4 * 4


def test_park_factors_are_passed_to_translate_line(snapshots: Path, monkeypatch) -> None:
    sentinel = pd.DataFrame(
        {"league": ["NPB"], "team": ["T1"], "season": [2020], "park_factor": [1.1], "games": [100]}
    )
    seen: list[object] = []
    real = historical.translate_line

    def spy(line, **kwargs):
        seen.append(kwargs.get("park_factors"))
        return real(line, **kwargs)

    monkeypatch.setattr(historical, "load_runs_park_factors", lambda root: sentinel)
    monkeypatch.setattr(historical, "translate_line", spy)
    payload = run_international_rolling(snapshots, league="NPB", target_seasons=[2021], n_boot=3)
    assert seen and all(p is sentinel for p in seen)
    # 2021 exports come from NPB 2020, which the sentinel covers for team T1.
    assert payload["by_target"]["2021"]["counts"]["park_adjusted_pairs"] == 2 * N_EXPORT


def test_kbo_without_data_is_all_skipped(snapshots: Path, tmp_path: Path) -> None:
    root = _build_snapshots(tmp_path)
    for role, cols in (("batters", HITTER_COLUMNS), ("pitchers", PITCHER_COLUMNS)):
        pd.DataFrame(columns=cols).to_csv(root / f"kbo_{role}.csv", index=False)
    payload = run_international_rolling(root, league="KBO", target_seasons=[2021], n_boot=3)
    assert payload["by_target"] == {}
    assert payload["gate"]["verdict"] == "INSUFFICIENT"


# --------------------------------------------------------------------------
# ship gate
# --------------------------------------------------------------------------


def _pooled(woba=(40, 0.020, 0.026, 0.80), fip=(40, 0.50, 0.60, 0.80)) -> list[dict]:
    rows = []
    for role, stat, (n, mae, base, cov) in (("batter", "woba", woba), ("pitcher", "fip", fip)):
        rows.append(
            {"role": role, "stat": stat, "n": n, "mae": mae, "baseline_mae": base, "coverage_80": cov}
        )
    return rows


def test_gate_passes_when_model_beats_baseline_and_coverage_in_range() -> None:
    gate = international_ship_gate(_pooled())
    assert gate["verdict"] == "PASS" and gate["failing"] == []


def test_gate_coverage_bounds_are_inclusive() -> None:
    assert international_ship_gate(_pooled(woba=(40, 0.02, 0.026, 0.70)))["verdict"] == "PASS"
    assert international_ship_gate(_pooled(fip=(40, 0.5, 0.6, 0.90)))["verdict"] == "PASS"


def test_gate_fails_on_mae_and_lists_the_failing_row() -> None:
    gate = international_ship_gate(_pooled(woba=(40, 0.030, 0.026, 0.80)))
    assert gate["verdict"] == "FAIL"
    assert [(r["role"], r["stat"]) for r in gate["failing"]] == [("batter", "woba")]
    assert "does not beat baseline" in gate["failing"][0]["reasons"][0]


@pytest.mark.parametrize("cov", [0.69, 0.91])
def test_gate_fails_on_coverage_out_of_range(cov: float) -> None:
    gate = international_ship_gate(_pooled(fip=(40, 0.5, 0.6, cov)))
    assert gate["verdict"] == "FAIL"
    assert [(r["role"], r["stat"]) for r in gate["failing"]] == [("pitcher", "fip")]


def test_gate_is_insufficient_below_ten_rows_or_when_missing() -> None:
    assert international_ship_gate(_pooled(woba=(9, 0.02, 0.026, 0.8)))["verdict"] == "INSUFFICIENT"
    assert international_ship_gate(_pooled(woba=(10, 0.02, 0.026, 0.8)))["verdict"] == "PASS"
    assert international_ship_gate(_pooled()[:1])["verdict"] == "INSUFFICIENT"
    assert international_ship_gate([])["verdict"] == "INSUFFICIENT"


def test_gate_insufficient_takes_precedence_over_fail() -> None:
    gate = international_ship_gate(
        _pooled(woba=(5, 0.02, 0.026, 0.8), fip=(40, 0.7, 0.6, 0.8))
    )
    assert gate["verdict"] == "INSUFFICIENT"
    assert {r["status"] for r in gate["rows"]} == {"INSUFFICIENT", "FAIL"}


def test_result_carries_gate_matching_pooled_rows(result: dict) -> None:
    assert result["gate"] == international_ship_gate(result["pooled"])


# --------------------------------------------------------------------------
# report + CLI
# --------------------------------------------------------------------------


def test_report_writes_json_and_markdown(result: dict, tmp_path: Path) -> None:
    paths = write_international_report(result, tmp_path / "out")
    assert paths["json"].name == "international-rolling-npb.json"
    loaded = json.loads(paths["json"].read_text())
    assert loaded["gate"]["verdict"] == result["gate"]["verdict"]
    md = paths["markdown"].read_text()
    assert md == render_international_markdown(result)
    assert f"Ship gate: {result['gate']['verdict']}" in md
    assert "NPB->MLB" in md and "direction_gap" in md and "2019" in md


def test_cli_international_rolling_prints_gate_and_writes_files(
    snapshots: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ir"
    res = CliRunner().invoke(
        cli.app,
        [
            "international-rolling", "--league", "NPB", "--targets", "2020-2022",
            "--boot", "3", "--snapshots", str(snapshots), "--out-dir", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Ship gate" in res.output and ("PASS" in res.output or "FAIL" in res.output)
    payload = json.loads((out / "international-rolling-npb.json").read_text())
    assert payload["metadata"]["target_seasons"] == [2020, 2021, 2022]
    assert (out / "international-rolling-npb.md").exists()


def test_cli_rejects_unknown_league(snapshots: Path, tmp_path: Path) -> None:
    res = CliRunner().invoke(
        cli.app,
        ["international-rolling", "--league", "CPBL", "--snapshots", str(snapshots),
         "--out-dir", str(tmp_path)],
    )
    assert res.exit_code != 0


def test_transfer_cohort_matches_build_cohort(snapshots: Path) -> None:
    from rosetta.cohort.builder import build_cohort

    expected = build_cohort(snapshots, chadwick_path=snapshots / "absent.csv")
    got = historical._transfer_cohort(
        snapshots, None, pd.DataFrame(), min_pa=80.0, min_ip=30.0
    )
    pd.testing.assert_frame_equal(got, expected)
    assert len(got) > 0


def test_results_doc_is_generated_from_payload_numbers(result: dict) -> None:
    doc = historical.render_international_doc([result])
    woba = _row(result["pooled"], "batter", "woba")
    assert doc.startswith("# International backtest results")
    assert f"| NPB | {result['gate']['verdict']} |" in doc
    assert f"| batter | woba | {woba['n']} | {woba['mae']:.4f} | {woba['baseline_mae']:.4f} |" in doc
    assert "Limitations" in doc and "2018" in doc and "imputed" in doc and "game logs" in doc
    assert "NPB->MLB" in doc
    with pytest.raises(ValueError):
        historical.render_international_doc([])
