"""Focused offline checks for historical report artifacts and CLI wiring."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from rosetta.backtest.report import _DETAIL_COLUMNS, write_historical_report
from rosetta.cli import app


def _payload() -> dict:
    return {
        "metadata": {
            "data_mode": "real",
            "target_season": 2025,
            "source_season": 2024,
            "training_cutoff": 2024,
            "source_counts": {"AAA": 2, "MLB": 2},
            "input_sha256": {"aaa.csv": "abc"},
            "limitations": ["Normalized <wOBA> is approximate."],
        },
        "metrics": [{
            "role": "batter", "stat": "woba", "n": 1, "mae": 0.01,
            "rmse": 0.02, "baseline_mae": 0.03, "baseline_rmse": 0.04,
            "coverage_80": 0.5,
        }],
        "detail": [{
            "player_id": "p1", "player_name": "A <script>alert(1)</script>",
            "role": "batter", "from_season": 2024, "to_season": 2025,
            "from_league": "AAA", "to_league": "MLB", "pre_sample": 100,
            "post_sample": 110, "pre_source": "aaa.csv", "post_source": "mlb.csv",
            "stat": "woba", "prior": 0.30, "prediction": 0.29, "actual": 0.31,
            "error": -0.02, "absolute_error": 0.02, "lower": 0.25,
            "upper": 0.34, "covered_80": True,
        }],
        "training_pairs": [{"player_id": "train-1", "to_season": 2024}],
        "model": {"links": {"batter": {"AAA->MLB": {"woba": {"n": 1}}}}},
    }


def test_report_artifacts_round_trip_and_escape(tmp_path: Path) -> None:
    payload = _payload()
    paths = write_historical_report(payload, tmp_path)

    assert set(paths) == {"json", "csv", "markdown", "html", "training_csv"}
    assert json.loads(paths["json"].read_text()) == payload
    assert "A &lt;script&gt;alert(1)&lt;/script&gt;" in paths["html"].read_text()
    assert "<script>alert(1)</script>" not in paths["html"].read_text()
    assert "50.0%" in paths["html"].read_text()
    assert "training_cutoff" in paths["html"].read_text()
    assert "player_id,player_name" in paths["csv"].read_text()
    assert "train-1" in paths["training_csv"].read_text()
    assert "A <script>alert(1)</script>" in paths["markdown"].read_text()


def test_named_examples_are_capped_per_role(tmp_path: Path) -> None:
    payload = _payload()
    payload["metadata"]["counts"] = {"training_pairs": 4, "evaluation_pairs": 9}
    payload["detail"] = [
        {**payload["detail"][0], "player_id": f"p{i}", "player_name": f"Player {i}"}
        for i in range(7)
    ]
    html = write_historical_report(payload, tmp_path)["html"].read_text()
    examples = html.split("<h2>Metrics</h2>", 1)[0]
    assert "Player 4" in examples
    assert "Player 5" not in examples
    assert "training pairs</b><br>4" in html


def test_historical_cli_passes_flags_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_run(snapshots, **kwargs):
        observed["snapshots"] = snapshots
        observed.update(kwargs)
        return _payload()

    monkeypatch.setattr("rosetta.backtest.historical.run_historical_backtest", fake_run)
    result = CliRunner().invoke(app, [
        "historical-backtest", "--snapshots", str(tmp_path / "snapshots"),
        "--target-season", "2026", "--boot", "7", "--seed", "9",
        "--min-pa", "12", "--min-ip", "13", "--out-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert observed == {
        "snapshots": Path(tmp_path / "snapshots"), "target_season": 2026,
        "n_boot": 7, "seed": 9, "min_pa": 12.0, "min_ip": 13.0, "estimator": "ratio_of_means",
        "era_floor": {"AAA": 2019},
    }
    assert "Historical backtest" in result.output
    assert (tmp_path / "out" / "historical-backtest.html").exists()


def test_historical_cli_no_era_floor_flag_disables_floor(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_run(snapshots, **kwargs):
        observed["snapshots"] = snapshots
        observed.update(kwargs)
        return _payload()

    monkeypatch.setattr("rosetta.backtest.historical.run_historical_backtest", fake_run)
    result = CliRunner().invoke(app, [
        "historical-backtest", "--snapshots", str(tmp_path / "snapshots"),
        "--no-era-floor", "--out-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert observed["era_floor"] is None


def test_detail_csv_matches_payload_detail(tmp_path: Path) -> None:
    payload = _payload()
    paths = write_historical_report(payload, tmp_path)

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        csv_rows = list(reader)

    detail = payload["detail"]
    # The CSV must lead with the canonical column order, even if extra keys
    # were appended for rows with additional fields.
    assert fieldnames[: len(_DETAIL_COLUMNS)] == list(_DETAIL_COLUMNS)
    assert set(fieldnames) == set(_DETAIL_COLUMNS)
    assert len(csv_rows) == len(detail)

    # The JSON artifact is written with sort_keys=True, so its column order
    # differs from the CSV's; compare column sets rather than order there.
    json_detail = json.loads(paths["json"].read_text())["detail"]
    assert set(json_detail[0].keys()) == set(fieldnames)

    for expected, actual in zip(detail, csv_rows, strict=True):
        for key in _DETAIL_COLUMNS:
            expected_value = expected[key]
            actual_value = actual[key]
            if key == "covered_80":
                assert (actual_value.strip().lower() == "true") == bool(expected_value)
            elif isinstance(expected_value, bool):
                assert (actual_value.strip().lower() == "true") == expected_value
            elif isinstance(expected_value, (int, float)):
                assert np.allclose(float(actual_value), float(expected_value))
            else:
                assert actual_value == str(expected_value)


def test_named_examples_case_insensitive_ordering(tmp_path: Path) -> None:
    payload = _payload()
    payload["detail"] = [
        {**payload["detail"][0], "player_id": "p-aj", "player_name": "AJ Zed"},
        {**payload["detail"][0], "player_id": "p-aaron", "player_name": "Aaron Able"},
    ]
    markdown = write_historical_report(payload, tmp_path)["markdown"].read_text()
    examples = markdown.split("## Named examples", 1)[1].split("## Selection and cutoff", 1)[0]

    assert "Aaron Able" in examples
    assert "AJ Zed" in examples
    # Case-sensitive sorting would put "AJ Zed" first (uppercase 'J' sorts
    # before lowercase 'a'); casefold sorting must put "Aaron Able" first.
    assert examples.index("Aaron Able") < examples.index("AJ Zed")


def test_historical_cli_reports_unusable_data(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise ValueError("no real AAA-to-MLB pairs found")

    monkeypatch.setattr("rosetta.backtest.historical.run_historical_backtest", fail)
    result = CliRunner().invoke(app, ["historical-backtest", "--snapshots", str(tmp_path)])
    assert result.exit_code == 1
    assert "no real AAA-to-MLB pairs found" in result.output
