from __future__ import annotations

import pandas as pd

from rosetta.ingest.chadwick import build_id_crosswalk, cross_reference_seasons


def test_build_id_crosswalk() -> None:
    register = pd.DataFrame(
        {
            "key_uuid": ["u1", "u2"],
            "key_mlbam": [660271.0, None],
            "key_kbo": [None, 62404.0],
            "key_npb": [1305137.0, None],
            "key_bbref": ["ohtansh01", None],
        }
    )
    cw = build_id_crosswalk(register)
    assert cw["mlbam-660271"] == "u1"
    assert cw["npb-1305137"] == "u1"
    assert cw["bbref-ohtansh01"] == "u1"
    assert cw["kbo-62404"] == "u2"
    # No prefix for non-matching keys
    assert "mlbam-NaN" not in cw


def test_cross_reference_seasons() -> None:
    register = pd.DataFrame(
        {
            "key_uuid": ["u1"],
            "key_mlbam": [660271.0],
            "key_npb": [1305137.0],
            "key_kbo": [None],
            "key_bbref": [None],
        }
    )
    cw = build_id_crosswalk(register)
    df = pd.DataFrame({"player_id": ["mlbam-660271", "mlbam-999999", "npb-1305137", "unknown"]})
    out = cross_reference_seasons(df, cw)
    assert out.loc[0, "key_uuid"] == "u1"
    assert out.loc[2, "key_uuid"] == "u1"
    assert pd.isna(out.loc[1, "key_uuid"])
    assert pd.isna(out.loc[3, "key_uuid"])