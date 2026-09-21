from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import rosetta.backtest.historical as historical
from rosetta.backtest.historical import run_rolling_backtest
from rosetta.backtest.report import write_rolling_report
from rosetta.schema import HITTER_COLUMNS


def _line(player_id: str, season: int, league: str, *, woba: float) -> dict:
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


def _rolling_fixture(root: Path) -> Path:
    """Adjacent AAA->MLB pairs at (2021,22),(2022,23),(2023,24),(2024,25).

    Target 2022 has no to_season<2022 training data and must be skipped;
    2023, 2024, 2025 each have at least one prior-season training pair.
    """
    aaa_rows = []
    mlb_rows = []
    for n, (from_season, to_season) in enumerate(((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))):
        for j in range(2):
            player_id = f"p-{n}-{j}"
            aaa_rows.append(_line(player_id, from_season, "AAA", woba=0.30 + n * 0.005 + j * 0.001))
            mlb_rows.append(_line(player_id, to_season, "MLB", woba=0.31 + n * 0.006 + j * 0.001))
    pd.DataFrame(aaa_rows, columns=HITTER_COLUMNS).to_csv(root / "aaa_batters.csv", index=False)
    pd.DataFrame(mlb_rows, columns=HITTER_COLUMNS).to_csv(root / "mlb_batters.csv", index=False)
    return root


def test_rolling_backtest_skips_target_with_no_training_and_pools_correctly(
    tmp_path: Path, monkeypatch
) -> None:
    root = _rolling_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_rolling_backtest(target_seasons=[2022, 2023, 2024, 2025], n_boot=1, min_pa=80)

    metadata = payload["metadata"]
    assert metadata["data_mode"] == "real"
    assert metadata["target_seasons"] == [2022, 2023, 2024, 2025]
    assert metadata["parameters"]["estimator"] == "ratio_of_means"

    skipped_seasons = {item["target_season"] for item in metadata["skipped"]}
    assert skipped_seasons == {2022}
    assert "2022" not in payload["by_target"]
    assert {2023, 2024, 2025} == {int(k) for k in payload["by_target"]}

    # Every target's training pairs must all have to_season < that target.
    for season_key, target_payload in payload["by_target"].items():
        season = int(season_key)
        for pair in target_payload["training_pairs"]:
            assert pair["to_season"] < season

    # Pooled n equals sum of per-target n for each (role, stat).
    per_target_n: dict[tuple[str, str], int] = {}
    for row in payload["summary"]:
        key = (row["role"], row["stat"])
        per_target_n[key] = per_target_n.get(key, 0) + row["n"]
    pooled_n = {(row["role"], row["stat"]): row["n"] for row in payload["pooled"]}
    assert pooled_n == per_target_n

    # Pooled MAE recomputes from the concatenated detail rows.
    for row in payload["pooled"]:
        details = [
            d
            for target_payload in payload["by_target"].values()
            for d in target_payload["detail"]
            if d["role"] == row["role"] and d["stat"] == row["stat"]
        ]
        errors = [abs(d["error"]) for d in details]
        assert row["mae"] == pytest.approx(sum(errors) / len(errors))
        assert row["n"] == len(details)


def test_rolling_backtest_estimator_recorded_and_propagated(tmp_path: Path, monkeypatch) -> None:
    root = _rolling_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_rolling_backtest(
        target_seasons=[2023, 2024, 2025], n_boot=1, min_pa=80, estimator="ratio_of_means"
    )
    assert payload["metadata"]["parameters"]["estimator"] == "ratio_of_means"
    for target_payload in payload["by_target"].values():
        assert target_payload["metadata"]["parameters"]["estimator"] == "ratio_of_means"


def test_rolling_backtest_all_targets_skipped_raises(tmp_path: Path, monkeypatch) -> None:
    root = _rolling_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    # Only 2021 has been established; with target 2021 there is no
    # to_season < 2021 training data available at all, so it must be skipped
    # and (being the only target requested) the whole run must raise.
    with pytest.raises(ValueError, match="skipped"):
        run_rolling_backtest(target_seasons=[2021], n_boot=1, min_pa=80)


def test_write_rolling_report_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    root = _rolling_fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_rolling_backtest(target_seasons=[2023, 2024, 2025], n_boot=1, min_pa=80)
    out_dir = tmp_path / "out"
    paths = write_rolling_report(payload, out_dir)

    assert paths["json"].exists()
    assert paths["summary_csv"].exists()
    assert paths["detail_csv"].exists()
    assert paths["markdown"].exists()

    detail_frame = pd.read_csv(paths["detail_csv"])
    assert list(detail_frame.columns)[0] == "target_season"
    assert "player_id" in detail_frame.columns

    summary_frame = pd.read_csv(paths["summary_csv"])
    assert "target_season" in summary_frame.columns
    assert "mae" in summary_frame.columns

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Historical rolling backtest" in markdown
