from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from apps.leaderboard import app as leaderboard_app

_SPEC = spec_from_file_location("export_demo", Path(__file__).parents[1] / "scripts" / "export_demo.py")
assert _SPEC and _SPEC.loader
_EXPORT_DEMO = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXPORT_DEMO)
build_analysis = _EXPORT_DEMO.build_analysis
build_backtest = _EXPORT_DEMO.build_backtest
build_leaderboard = _EXPORT_DEMO.build_leaderboard


def _detail(
    player_id: str,
    *,
    season: int,
    role: str,
    stat: str,
    predicted: float,
    actual: float,
    prior: float,
    lower: float,
    upper: float,
    covered: bool,
) -> dict:
    absolute_error = abs(predicted - actual)
    return {
        "player_id": player_id,
        "player_name": player_id.title(),
        "role": role,
        "stat": stat,
        "from_league": "AAA",
        "prediction": predicted,
        "predicted": predicted,
        "actual": actual,
        "prior": prior,
        "lower": lower,
        "upper": upper,
        "covered_80": covered,
        "absolute_error": absolute_error,
        "error": predicted - actual,
        "from_season": season - 1,
        "to_season": season,
        "pre_sample": 100,
        "post_sample": 100,
        "pre_source": "fixture-real",
        "post_source": "fixture-real",
    }


def _rolling(*details: dict) -> dict:
    by_target: dict[str, dict] = {}
    for row in details:
        target = str(row["to_season"])
        payload = by_target.setdefault(target, {"detail": [], "metrics": []})
        payload["detail"].append(row)
    return {
        "metadata": {"data_mode": "real", "target_seasons": sorted(map(int, by_target))},
        "by_target": by_target,
        "summary": [],
        "pooled": [],
    }


def test_analysis_emits_stat_specific_calibration_and_error_buckets() -> None:
    woba = _detail(
        "batter-1", season=2024, role="batter", stat="woba", predicted=0.37,
        actual=0.34, prior=0.30, lower=0.31, upper=0.39, covered=True,
    )
    fip = _detail(
        "pitcher-1", season=2024, role="pitcher", stat="fip", predicted=4.4,
        actual=3.8, prior=4.0, lower=3.4, upper=4.9, covered=True,
    )
    result = build_analysis(_rolling(woba, fip))

    assert result["schema_version"] == "analysis.v2"
    summary = {(row["role"], row["stat"]): row for row in result["current_stat_summary"]}
    assert summary["batter", "woba"]["baseline_improvement"] == pytest.approx(0.01)
    assert summary["batter", "woba"]["coverage_delta"] == pytest.approx(0.2)
    assert summary["batter", "woba"]["error_buckets"][1]["bucket"] == "moderate"
    assert summary["pitcher", "fip"]["error_buckets"][1]["bucket"] == "moderate"
    assert result["definitions"]["coverage_delta"]


def test_analysis_handles_a_target_with_only_one_role() -> None:
    row = _detail(
        "batter-1", season=2025, role="batter", stat="woba", predicted=0.35,
        actual=0.34, prior=0.30, lower=0.30, upper=0.40, covered=True,
    )
    result = build_analysis(_rolling(row))

    assert result["current_season"] == 2025
    assert [item["role"] for item in result["current_role_summary"]] == ["batter", "pitcher"]
    missing = result["current_role_summary"][1]
    assert missing["role"] == "pitcher"
    assert missing["n"] == 0
    assert missing["mae"] is None
    assert missing["coverage_80"] is None


def test_analysis_template_renders_when_a_stat_role_is_missing() -> None:
    row = _detail(
        "batter-1", season=2025, role="batter", stat="woba", predicted=0.35,
        actual=0.34, prior=0.30, lower=0.30, upper=0.40, covered=True,
    )
    data = build_analysis(_rolling(row))

    html = leaderboard_app.TEMPLATES.get_template("analysis.html").render(data=data)

    assert "Batter WOBA" in html
    assert "2025" in html


def test_build_backtest_allows_partial_metric_set_and_reports_score() -> None:
    backtest = {
        "metadata": {"data_mode": "real"},
        "metrics": [
            {
                "role": "batter", "stat": "woba", "n": 2,
                "mae": 0.02, "rmse": 0.03, "baseline_mae": 0.04,
                "baseline_rmse": 0.05, "coverage_80": 0.5,
            }
        ],
        "detail": [],
    }
    rolling = {"metadata": {"target_seasons": []}, "summary": [], "pooled": []}

    result = build_backtest(backtest, rolling, target_season=2025)

    assert result["data_mode"] == "real"
    assert result["validation"] == {
        "metric_count": 1,
        "model_wins": 1,
        "baseline_wins": 0,
        "model_win_rate": 1.0,
    }


def test_build_backtest_uses_rolling_target_for_calibrated_public_card() -> None:
    raw_metric = {
        "role": "batter", "stat": "woba", "n": 1,
        "mae": 0.04, "rmse": 0.04, "baseline_mae": 0.05,
        "baseline_rmse": 0.05, "coverage_80": 0.8,
    }
    calibrated_metric = {**raw_metric, "mae": 0.02, "rmse": 0.02}
    raw_detail = _detail(
        "raw-player", season=2025, role="batter", stat="woba", predicted=0.30,
        actual=0.34, prior=0.29, lower=0.25, upper=0.35, covered=True,
    )
    calibrated_detail = _detail(
        "calibrated-player", season=2025, role="batter", stat="woba", predicted=0.33,
        actual=0.34, prior=0.29, lower=0.28, upper=0.38, covered=True,
    )
    raw = {"metadata": {"data_mode": "real"}, "metrics": [raw_metric], "detail": [raw_detail]}
    rolling = {
        "metadata": {"data_mode": "real", "target_seasons": [2025]},
        "by_target": {
            "2025": {
                "metadata": {"calibration": {"method": "prior_affine"}},
                "metrics": [calibrated_metric],
                "detail": [calibrated_detail],
            }
        },
        "summary": [],
        "pooled": [],
    }

    result = build_backtest(raw, rolling, target_season=2025)

    assert result["evaluation"] == {
        "source": "historical-rolling target fold",
        "calibration": "prior_affine",
    }
    assert result["target_season_2025"]["metrics"][0]["mae"] == pytest.approx(0.02)
    assert result["player_predictions"]["batter_woba"][0]["player_id"] == "calibrated-player"


@pytest.mark.parametrize(
    "builder,payload",
    [
        (build_leaderboard, {"data_mode": "synthetic", "boards": {"batter": []}}),
        (build_backtest, ({"metadata": {"data_mode": "synthetic"}}, {"summary": [], "pooled": []})),
        (build_analysis, {"metadata": {"data_mode": "synthetic"}}),
    ],
)
def test_exporters_refuse_non_real_data(builder, payload) -> None:
    with pytest.raises(SystemExit, match="real"):
        if builder is build_backtest:
            builder(*payload, target_season=2025)
        else:
            builder(payload)
