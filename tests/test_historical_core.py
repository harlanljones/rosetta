from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import rosetta.backtest.historical as historical
from rosetta.backtest.historical import run_historical_backtest
from rosetta.schema import HITTER_COLUMNS


def _line(player_id: str, season: int, league: str, *, source: str = "fixture", woba: float = 0.32) -> dict:
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
            "source": source,
            "is_synthetic": False,
        }
    )
    return row


def _fixture(root: Path) -> Path:
    aaa = [_line("p1", season, "AAA", woba=0.30 + season / 10000) for season in (2023, 2024)]
    mlb = [_line("p1", season, "MLB", woba=0.31 + season / 10000) for season in (2024, 2025)]
    pd.DataFrame(aaa, columns=HITTER_COLUMNS).to_csv(root / "aaa_batters.csv", index=False)
    pd.DataFrame(mlb, columns=HITTER_COLUMNS).to_csv(root / "mlb_batters.csv", index=False)
    return root


def test_historical_fixture_is_real_and_uses_default_snapshot_constant(tmp_path: Path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(historical, "SNAPSHOT_DIR", root)
    payload = run_historical_backtest(n_boot=1, min_pa=80)
    assert payload["metadata"]["data_mode"] == "real"
    assert payload["metadata"]["input_sha256"]["aaa_batters.csv"]
    assert payload["metadata"]["counts"]["training_pairs"] == 1
    assert payload["metadata"]["counts"]["evaluation_pairs"] == 1
    assert payload["detail"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_season": True},
        {"target_season": 2025.5},
        {"n_boot": False},
        {"n_boot": 0.5},
        {"seed": -1},
    ],
)
def test_historical_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="integer|at least|non-negative"):
        run_historical_backtest(**kwargs)


def test_historical_rejects_wrong_league(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "aaa_batters.csv")
    frame.loc[0, "league"] = "MLB"
    frame.to_csv(root / "aaa_batters.csv", index=False)
    with pytest.raises(ValueError, match="wrong league"):
        run_historical_backtest(root, n_boot=1)


@pytest.mark.parametrize("field", ["player_id", "source"])
def test_historical_rejects_blank_identity_or_provenance(tmp_path: Path, field: str) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "aaa_batters.csv")
    frame.loc[0, field] = ""
    frame.to_csv(root / "aaa_batters.csv", index=False)
    with pytest.raises(ValueError, match=f"blank {field}"):
        run_historical_backtest(root, n_boot=1)


def test_historical_rejects_nonfinite_pair_rate(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "mlb_batters.csv")
    frame.loc[0, "woba"] = float("nan")
    frame.to_csv(root / "mlb_batters.csv", index=False)
    with pytest.raises(ValueError, match="non-finite required values"):
        run_historical_backtest(root, n_boot=1)

