from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rosetta.backtest.historical import _score_metrics, run_historical_backtest
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS


def _hitter_row(
    player_id: str,
    season: int,
    *,
    synthetic: bool = False,
    source: str = "fixture-real",
    woba: float | None = None,
) -> dict:
    i = sum(ord(char) for char in player_id) % 17
    bb = 0.07 + i * 0.002 + (season - 2021) * 0.001
    k = 0.18 + i * 0.003
    iso = 0.14 + i * 0.004
    babip = 0.29 + i * 0.002
    hr = 0.03 + i * 0.001
    return {
        "player_id": player_id,
        "player_name": player_id,
        "season": season,
        "league": "AAA",
        "team": "Fixture club",
        "role": "batter",
        "age": 25.0 + (season - 2021) * 0.2,
        "pa": 120.0,
        "g": 40.0,
        "bb_pct": bb,
        "k_pct": k,
        "iso": iso,
        "babip": babip,
        "hr_pct": hr,
        "avg": 0.27,
        "obp": 0.35,
        "slg": 0.45,
        "woba": woba if woba is not None else 0.32 + i * 0.004,
        "home_pa": 60.0,
        "road_pa": 60.0,
        "home_woba": 0.32,
        "road_woba": 0.32,
        "park_id": "fixture",
        "source": source,
        "is_synthetic": synthetic,
    }


def _pitcher_row(
    player_id: str,
    season: int,
    *,
    synthetic: bool = False,
    source: str = "fixture-real",
    fip: float | None = None,
) -> dict:
    i = sum(ord(char) for char in player_id) % 17
    k = 0.19 + i * 0.003
    bb = 0.07 + i * 0.001
    hr_fb = 0.10 + i * 0.002
    era = 3.7 + i * 0.04
    return {
        "player_id": player_id,
        "player_name": player_id,
        "season": season,
        "league": "AAA",
        "team": "Fixture club",
        "role": "pitcher",
        "age": 25.0 + (season - 2021) * 0.2,
        "ip": 45.0,
        "g": 20.0,
        "gs": 10.0,
        "k_pct": k,
        "bb_pct": bb,
        "hr_fb": hr_fb,
        "era": era,
        "fip": fip if fip is not None else era - 0.2,
        "home_pa": 100.0,
        "road_pa": 100.0,
        "home_era": era,
        "road_era": era,
        "park_id": "fixture",
        "source": source,
        "is_synthetic": synthetic,
    }


def _write_fixture(root: Path, *, with_training: bool = True) -> None:
    aaa_hitters: list[dict] = []
    mlb_hitters: list[dict] = []
    aaa_pitchers: list[dict] = []
    mlb_pitchers: list[dict] = []
    if with_training:
        # Six train pairs (two in each completed destination season) ensure
        # bootstrap samples are active for each role/stat.
        for n, (from_season, to_season) in enumerate(
            ((2021, 2022), (2022, 2023), (2023, 2024))
        ):
            for j in range(2):
                player_id = f"train-{n}-{j}"
                aaa_hitters.append(_hitter_row(player_id, from_season))
                mlb_hitters.append(
                    _hitter_row(player_id, to_season, woba=0.34 + n * 0.01 + j * 0.002)
                )
                aaa_pitchers.append(_pitcher_row(player_id, from_season))
                mlb_pitchers.append(
                    _pitcher_row(player_id, to_season, fip=3.4 + n * 0.1 + j * 0.03)
                )

    for j in range(2):
        player_id = f"eval-{j}"
        aaa_hitters.append(_hitter_row(player_id, 2024))
        mlb_hitters.append(_hitter_row(player_id, 2025, woba=0.35 + j * 0.01))
        aaa_pitchers.append(_pitcher_row(player_id, 2024))
        mlb_pitchers.append(_pitcher_row(player_id, 2025, fip=3.6 + j * 0.1))

    # These rows must never enter a real-data result, even though they look
    # like valid target pairs.
    aaa_hitters.append(
        _hitter_row("synthetic-eval", 2024, synthetic=True, source="fixture-synthetic")
    )
    mlb_hitters.append(
        _hitter_row("synthetic-eval", 2025, synthetic=True, source="fixture-synthetic")
    )
    aaa_pitchers.append(
        _pitcher_row("synthetic-eval", 2024, synthetic=True, source="fixture-synthetic")
    )
    mlb_pitchers.append(
        _pitcher_row("synthetic-eval", 2025, synthetic=True, source="fixture-synthetic")
    )

    # Future malformed lines are filtered before value validation and cannot
    # leak into a target-2025 result.
    future = _hitter_row("future-only", 2026)
    future["woba"] = np.nan
    mlb_hitters.append(future)

    pd.DataFrame(aaa_hitters, columns=HITTER_COLUMNS).to_csv(root / "aaa_batters.csv", index=False)
    pd.DataFrame(mlb_hitters, columns=HITTER_COLUMNS).assign(league="MLB").to_csv(
        root / "mlb_batters.csv", index=False
    )
    pd.DataFrame(aaa_pitchers, columns=PITCHER_COLUMNS).to_csv(root / "aaa_pitchers.csv", index=False)
    pd.DataFrame(mlb_pitchers, columns=PITCHER_COLUMNS).assign(league="MLB").to_csv(
        root / "mlb_pitchers.csv", index=False
    )


def _append_pair(
    root: Path,
    role: str,
    player_id: str,
    *,
    from_season: int = 2024,
    to_season: int = 2025,
    pre_synthetic: bool,
    post_synthetic: bool,
) -> None:
    if role == "batter":
        columns, builder, name = HITTER_COLUMNS, _hitter_row, "batters"
    else:
        columns, builder, name = PITCHER_COLUMNS, _pitcher_row, "pitchers"
    pre = builder(player_id, from_season, synthetic=pre_synthetic, source="fixture-synthetic" if pre_synthetic else "fixture-real")
    post = builder(player_id, to_season, synthetic=post_synthetic, source="fixture-synthetic" if post_synthetic else "fixture-real")
    aaa_path = root / f"aaa_{name}.csv"
    mlb_path = root / f"mlb_{name}.csv"
    pd.concat([pd.read_csv(aaa_path), pd.DataFrame([pre], columns=columns)], ignore_index=True).to_csv(aaa_path, index=False)
    pd.concat([pd.read_csv(mlb_path), pd.DataFrame([post], columns=columns).assign(league="MLB")], ignore_index=True).to_csv(mlb_path, index=False)


def _run(root: Path) -> dict:
    return run_historical_backtest(root, target_season=2025, n_boot=5, seed=17)


def test_historical_excludes_synthetic_lines_and_future_rows(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    payload = _run(tmp_path)

    assert payload["metadata"]["data_mode"] == "real"
    assert payload["metadata"]["counts"]["evaluation_pairs"] == 4
    assert payload["metadata"]["counts"]["training_pairs"] == 12
    assert all(row["player_id"] != "synthetic-eval" for row in payload["detail"])
    assert all(row["post_source"] == "fixture-real" for row in payload["detail"])
    assert all(row["to_season"] <= 2025 for row in payload["detail"])
    assert all(pair["to_season"] < 2025 for pair in payload["training_pairs"])
    assert payload["metadata"]["latest_training_to_season"] == 2024


@pytest.mark.parametrize("role", ["batter", "pitcher"])
@pytest.mark.parametrize("synthetic_side", ["pre", "post"])
def test_synthetic_source_or_actual_is_excluded_even_with_real_opposite(
    tmp_path: Path, role: str, synthetic_side: str
) -> None:
    _write_fixture(tmp_path)
    _append_pair(
        tmp_path,
        role,
        f"synthetic-{role}-{synthetic_side}",
        pre_synthetic=synthetic_side == "pre",
        post_synthetic=synthetic_side == "post",
    )
    payload = _run(tmp_path)
    assert payload["metadata"]["counts"]["evaluation_pairs"] == 4
    assert all("synthetic-" not in row["player_id"] for row in payload["detail"])
    assert all("synthetic-" not in row["player_id"] for row in payload["training_pairs"])


@pytest.mark.parametrize("role", ["batter", "pitcher"])
@pytest.mark.parametrize("synthetic_side", ["pre", "post"])
def test_synthetic_training_endpoint_cannot_change_factors(
    tmp_path: Path, role: str, synthetic_side: str
) -> None:
    _write_fixture(tmp_path)
    before = _run(tmp_path)
    _append_pair(
        tmp_path,
        role,
        f"synthetic-train-{role}-{synthetic_side}",
        from_season=2023,
        to_season=2024,
        pre_synthetic=synthetic_side == "pre",
        post_synthetic=synthetic_side == "post",
    )
    after = _run(tmp_path)
    assert before["model"] == after["model"]
    for key in ("candidate_adjacent_pairs", "training_pairs", "evaluation_pairs", "detail_rows"):
        assert before["metadata"]["counts"][key] == after["metadata"]["counts"][key]

    control_root = tmp_path / "real-control"
    control_root.mkdir()
    _write_fixture(control_root)
    _append_pair(
        control_root,
        role,
        f"real-train-{role}",
        from_season=2023,
        to_season=2024,
        pre_synthetic=False,
        post_synthetic=False,
    )
    assert before["model"] != _run(control_root)["model"]


def test_temporal_cutoff_ignores_misleading_transfer_flags(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    # The historical evaluator must not consult this stale legacy index.
    pd.DataFrame(
        [
            {
                "player_id": "future-only",
                "player_name": "future-only",
                "role": "batter",
                "from_league": "AAA",
                "to_league": "MLB",
                "from_season": 2025,
                "to_season": 2026,
                "age_from": 30,
                "age_to": 31,
                "holdout": False,
            }
        ]
    ).to_csv(tmp_path / "transfers.csv", index=False)
    payload = _run(tmp_path)
    assert payload["metadata"]["latest_training_to_season"] < 2025
    assert {pair["to_season"] for pair in payload["training_pairs"]} == {2022, 2023, 2024}


def test_target_actual_perturbation_does_not_change_fit_or_predictions(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    before = _run(tmp_path)
    path = tmp_path / "mlb_batters.csv"
    frame = pd.read_csv(path)
    mask = (frame["player_id"] == "eval-0") & (frame["season"] == 2025)
    frame.loc[mask, "woba"] = 0.49
    frame.to_csv(path, index=False)
    after = _run(tmp_path)

    assert before["model"] == after["model"]
    before_predictions = [(r["player_id"], r["stat"], r["prediction"]) for r in before["detail"]]
    after_predictions = [(r["player_id"], r["stat"], r["prediction"]) for r in after["detail"]]
    assert before_predictions == after_predictions
    assert before["metadata"]["counts"] == after["metadata"]["counts"]
    assert before["detail"] != after["detail"]


def test_metrics_and_baseline_reproduce_from_machine_readable_detail(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    payload = _run(tmp_path)
    detail = pd.DataFrame(payload["detail"])
    for metric in payload["metrics"]:
        group = detail[(detail.role == metric["role"]) & (detail.stat == metric["stat"])]
        error = group["prediction"].to_numpy() - group["actual"].to_numpy()
        baseline = group["prior"].to_numpy() - group["actual"].to_numpy()
        assert metric["n"] == len(group)
        assert metric["mae"] == pytest.approx(np.abs(error).mean())
        assert metric["rmse"] == pytest.approx(np.sqrt(np.square(error).mean()))
        assert metric["baseline_mae"] == pytest.approx(np.abs(baseline).mean())
        assert metric["baseline_rmse"] == pytest.approx(np.sqrt(np.square(baseline).mean()))
        assert metric["coverage_80"] == pytest.approx(group["covered_80"].mean())

    required = {
        "player_id", "player_name", "role", "from_season", "to_season", "from_league",
        "to_league", "pre_sample", "post_sample", "pre_source", "post_source", "stat",
        "prior", "prediction", "actual", "error", "absolute_error", "lower", "upper",
        "covered_80",
    }
    assert required <= set(detail.columns)


def test_hand_calculated_metric_control_case() -> None:
    metrics = _score_metrics(
        [
            {"role": "batter", "stat": "woba", "prediction": 0.4, "actual": 0.3,
             "prior": 0.2, "error": 0.1, "covered_80": True},
            {"role": "batter", "stat": "woba", "prediction": 0.4, "actual": 0.5,
             "prior": 0.2, "error": -0.1, "covered_80": False},
        ]
    )
    assert metrics == [
        {
            "role": "batter", "stat": "woba", "n": 2, "mae": pytest.approx(0.1),
            "rmse": pytest.approx(0.1), "baseline_mae": pytest.approx(0.2),
            "baseline_rmse": pytest.approx(np.sqrt(0.05)), "coverage_80": pytest.approx(0.5),
        }
    ]


def test_historical_output_reproducible_across_python_hash_seeds(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    code = (
        "import json; from rosetta.backtest.historical import run_historical_backtest; "
        "print(json.dumps(run_historical_backtest(r'"
        + str(tmp_path)
        + "', target_season=2025, n_boot=5, seed=17), sort_keys=True, separators=(',', ':')))"
    )
    repo = Path(__file__).parents[1]
    outputs = []
    for hash_seed in ("1", "991"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(repo / "src")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    json.loads(outputs[0])


def test_insufficient_data_and_ambiguous_duplicate_are_actionable(tmp_path: Path) -> None:
    _write_fixture(tmp_path, with_training=False)
    with pytest.raises(ValueError, match="insufficient real AAA-to-MLB training pairs"):
        _run(tmp_path)

    _write_fixture(tmp_path)
    path = tmp_path / "aaa_batters.csv"
    frame = pd.read_csv(path)
    frame = pd.concat([frame, frame.loc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ambiguous duplicate season keys"):
        _run(tmp_path)
