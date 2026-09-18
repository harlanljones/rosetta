from pathlib import Path

import pandas as pd

from rosetta.ingest.savant import (
    aggregate_batters,
    derive_park_factors,
    pitch_mix_features,
)

FIX = Path(__file__).parent / "fixtures" / "savant_minors_sample.csv"


def _pitches() -> pd.DataFrame:
    return pd.read_csv(FIX, low_memory=False)


def test_aggregate_batters_rates() -> None:
    df = aggregate_batters(_pitches(), 2024, "AAA")
    assert len(df) > 10
    top = df.sort_values("pa", ascending=False).head(3)
    for _, row in top.iterrows():
        assert 0 <= row["bb_pct"] < 0.5
        assert 0 <= row["k_pct"] < 0.6
    # woba can be extreme with tiny single-day samples (< 10 PA); just check range
    assert all(0.1 < top["woba"]) and all(top["woba"] < 1.2)
    row = df.iloc[0]
    assert row["league"] == "AAA"
    assert row["player_id"].startswith("mlbam-")
    assert row["source"] == "savant"
    assert row["home_pa"] + row["road_pa"] > 0


def test_aggregate_pitchers_rates() -> None:
    from rosetta.ingest.savant import aggregate_pitchers

    df = aggregate_pitchers(_pitches(), 2024, "AAA")
    assert len(df) > 5
    # With single-day fixture, each pitcher has few BF. Just check values are in bounds.
    for _, row in df.iterrows():
        assert 0 <= row["k_pct"] <= 1
        assert 0 <= row["bb_pct"] <= 1
    row = df.iloc[0]
    assert row["source"] == "savant"
    assert row["ip"] == 0.0
    try:
        is_missing = pd.isna(row["era"])
    except Exception:
        is_missing = row["era"] is None
    assert is_missing


def test_pitch_mix_features() -> None:
    df = pitch_mix_features(_pitches())
    assert len(df) > 5
    assert "arsenal" in df.columns
    assert "avg_velo" in df.columns
    assert "xwoba_against" in df.columns


def test_derive_park_factors() -> None:
    df = derive_park_factors(_pitches(), min_pa=10)
    assert len(df) > 0
    row = df.iloc[0]
    assert row["park_factor"] > 0.5
    assert row["park_factor"] < 2.0
    assert row["n_pa"] >= 10