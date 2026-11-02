"""WP3: transfer direction, KBO foreign-player crosswalk, cohort passthrough."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rosetta.cohort.builder import build_cohort
from rosetta.ingest.chadwick import (
    apply_kbo_foreign_map,
    build_id_crosswalk,
    load_kbo_foreign_map,
)
from rosetta.ingest.loaders import classify_direction, load_transfers
from rosetta.ingest.transfers import generate_transfers
from rosetta.paths import SNAPSHOT_DIR
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

REGISTER_COLS = ["key_uuid", "key_mlbam", "key_npb", "key_kbo", "name_last", "name_given"]


def _line(pid: str, name: str, season: int, league: str, **extra) -> dict:
    row = {c: "" for c in HITTER_COLUMNS}
    row.update(
        player_id=pid, player_name=name, season=season, league=league, role="batter",
        team="T", age=27.0, pa=300.0, bb_pct=0.08, k_pct=0.2, iso=0.15, babip=0.3,
        hr_pct=0.03, avg=0.27, obp=0.34, slg=0.45, woba=0.32, home_pa=0.0, road_pa=0.0,
        home_woba=0.32, road_woba=0.32, source="test", is_synthetic=False,
    )
    row.update(extra)
    return row


def _write_snapshots(root: Path, lines: list[dict]) -> None:
    frame = pd.DataFrame(lines)
    for league in ("MLB", "AAA", "NPB", "KBO"):
        sub = frame[frame["league"] == league]
        cols = [c for c in sub.columns if c in HITTER_COLUMNS or c == "age_source"]
        sub[cols].to_csv(root / f"{league.lower()}_batters.csv", index=False)
        pd.DataFrame(columns=PITCHER_COLUMNS).to_csv(root / f"{league.lower()}_pitchers.csv", index=False)


def _write_register(root: Path, rows: list[dict]) -> Path:
    path = root / "chadwick_people.csv"
    pd.DataFrame(rows, columns=REGISTER_COLS).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("frm", "to", "expected"),
    [
        ("NPB", "MLB", "export"),
        ("KBO", "AAA", "export"),
        ("CPBL", "MLB", "export"),
        ("CUBA", "AAA", "export"),
        ("MLB", "NPB", "import"),
        ("AAA", "KBO", "import"),
        ("AAA", "CPBL", "import"),
        ("AAA", "MLB", "lateral"),
        ("MLB", "AAA", "lateral"),
        ("NPB", "KBO", "lateral"),
        ("KBO", "NPB", "lateral"),
    ],
)
def test_classify_direction(frm: str, to: str, expected: str) -> None:
    assert classify_direction(frm, to) == expected


def test_load_transfers_derives_direction_for_old_files(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            ["a", "A", "batter", "NPB", "MLB", 2022, 2023, 27.0, 28.0, False],
            ["b", "B", "batter", "AAA", "KBO", 2022, 2023, 27.0, 28.0, False],
            ["c", "C", "batter", "AAA", "MLB", 2022, 2023, 27.0, 28.0, False],
        ],
        columns=[
            "player_id", "player_name", "role", "from_league", "to_league",
            "from_season", "to_season", "age_from", "age_to", "holdout",
        ],
    ).to_csv(tmp_path / "transfers.csv", index=False)
    t = load_transfers(tmp_path)
    assert list(t["direction"]) == ["export", "import", "lateral"]


def test_load_transfers_keeps_existing_direction_and_fills_blanks(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "player_id": ["a", "b"], "player_name": ["A", "B"], "role": ["batter"] * 2,
            "from_league": ["NPB", "AAA"], "to_league": ["MLB", "KBO"],
            "from_season": [2022, 2022], "to_season": [2023, 2023],
            "age_from": [27.0, 27.0], "age_to": [28.0, 28.0],
            "direction": ["export", None],
        }
    ).to_csv(tmp_path / "transfers.csv", index=False)
    assert list(load_transfers(tmp_path)["direction"]) == ["export", "import"]


def test_committed_snapshots_still_load() -> None:
    t = load_transfers(SNAPSHOT_DIR)
    assert not t.empty
    assert set(t["direction"]) <= {"export", "import", "lateral"}
    assert (SNAPSHOT_DIR / "kbo_foreign_map.csv").read_text().strip() == (
        "kbo_player_id,mlbam_id,name_en,source"
    )
    assert load_kbo_foreign_map(SNAPSHOT_DIR / "kbo_foreign_map.csv") == {}


def test_generate_transfers_adds_direction(tmp_path: Path) -> None:
    reg = _write_register(
        tmp_path,
        [
            {"key_uuid": "u1", "key_mlbam": 1, "key_npb": 11, "name_last": "A", "name_given": "A"},
            {"key_uuid": "u2", "key_mlbam": 2, "key_npb": 22, "name_last": "B", "name_given": "B"},
            {"key_uuid": "u3", "key_mlbam": 3, "name_last": "C", "name_given": "C"},
        ],
    )
    _write_snapshots(
        tmp_path,
        [
            _line("npb-11", "A", 2022, "NPB"), _line("mlbam-1", "A", 2023, "MLB"),
            _line("mlbam-2", "B", 2022, "MLB"), _line("npb-22", "B", 2023, "NPB"),
            _line("mlbam-3", "C", 2022, "AAA"), _line("mlbam-3", "C", 2023, "MLB"),
        ],
    )
    t = generate_transfers(register_path=reg, snapshot_dir=tmp_path)
    by_name = dict(zip(t["player_name"], t["direction"], strict=True))
    assert by_name == {"A": "export", "B": "import", "C": "lateral"}


def test_kbo_foreign_map_loader_formats(tmp_path: Path) -> None:
    path = tmp_path / "kbo_foreign_map.csv"
    path.write_text(
        "kbo_player_id,mlbam_id,name_en,source\n"
        "55123,600001.0,Foreign One,kbo-en\n"
        "kbo-55124,600002,Foreign Two,kbo-en\n"
        "55125,,No Mlbam,kbo-en\n"
    )
    assert load_kbo_foreign_map(path) == {"kbo-55123": "600001", "kbo-55124": "600002"}
    assert load_kbo_foreign_map(tmp_path / "missing.csv") == {}


def test_foreign_map_takes_precedence_over_chadwick() -> None:
    cross = {"mlbam-9": "uuid-9", "kbo-5": "wrong-uuid"}
    out = apply_kbo_foreign_map(cross, {"kbo-5": "9", "kbo-6": "77"})
    assert out["kbo-5"] == "uuid-9"
    assert out["kbo-6"] == out["mlbam-77"]  # not in Chadwick: shared synthetic key


def test_kbo_import_pair_via_foreign_map(tmp_path: Path) -> None:
    """AAA -> KBO import: only the foreign map can link the romanized KBO row."""
    reg = _write_register(
        tmp_path,
        [{"key_uuid": "u1", "key_mlbam": 600001, "name_last": "Smith", "name_given": "John"}],
    )
    _write_snapshots(
        tmp_path,
        [
            _line("mlbam-600001", "John Smith", 2023, "AAA"),
            _line("kbo-55123", "Jonseu Seumiseu", 2024, "KBO", age_source="birth_year"),
        ],
    )
    # Without the map: no pair.
    assert generate_transfers(register_path=reg, snapshot_dir=tmp_path).empty

    (tmp_path / "kbo_foreign_map.csv").write_text(
        "kbo_player_id,mlbam_id,name_en,source\n55123,600001,John Smith,test\n"
    )
    t = generate_transfers(register_path=reg, snapshot_dir=tmp_path)
    assert len(t) == 1
    row = t.iloc[0]
    assert (row["from_league"], row["to_league"], row["direction"]) == ("AAA", "KBO", "import")
    assert row["player_id"] == "mlbam-600001"

    # Mapped id not in Chadwick at all still pairs.
    (tmp_path / "kbo_foreign_map.csv").write_text(
        "kbo_player_id,mlbam_id,name_en,source\n55123,600001,John Smith,test\n"
    )
    # And the crosswalk bridges both ids for cohort building.
    cw = build_id_crosswalk(
        pd.read_csv(reg), kbo_foreign_map_path=tmp_path / "kbo_foreign_map.csv"
    )
    assert cw["kbo-55123"] == cw["mlbam-600001"] == "u1"

    # Mapped id absent from Chadwick still pairs.
    other = tmp_path / "other_reg"
    other.mkdir()
    empty_reg = _write_register(other, [])
    t2 = generate_transfers(register_path=empty_reg, snapshot_dir=tmp_path)
    assert len(t2) == 1


def test_cohort_carries_direction_and_age_source(tmp_path: Path) -> None:
    reg = _write_register(
        tmp_path,
        [{"key_uuid": "u1", "key_mlbam": 600001, "name_last": "Smith", "name_given": "John"}],
    )
    _write_snapshots(
        tmp_path,
        [
            _line("mlbam-600001", "John Smith", 2023, "AAA", age_source="birth_year"),
            _line("kbo-55123", "Jonseu Seumiseu", 2024, "KBO", age_source="imputed"),
        ],
    )
    (tmp_path / "kbo_foreign_map.csv").write_text(
        "kbo_player_id,mlbam_id,name_en,source\n55123,600001,John Smith,test\n"
    )
    t = generate_transfers(register_path=reg, snapshot_dir=tmp_path)
    t.to_csv(tmp_path / "transfers.csv", index=False)

    cohort = build_cohort(tmp_path, chadwick_path=reg)
    assert len(cohort) == 1
    row = cohort.iloc[0]
    assert row["direction"] == "import"
    assert row["pre_age_source"] == "birth_year"
    assert row["post_age_source"] == "imputed"


def test_cohort_age_source_defaults_blank_and_direction_derived(tmp_path: Path) -> None:
    """Old transfers.csv (no direction) and old seasons (no age_source) still build."""
    reg = _write_register(tmp_path, [])
    _write_snapshots(
        tmp_path,
        [_line("npb-1", "X", 2022, "NPB"), _line("npb-1", "X", 2023, "MLB")],
    )
    pd.DataFrame(
        [["npb-1", "X", "batter", "NPB", "MLB", 2022, 2023, 27.0, 28.0, False]],
        columns=[
            "player_id", "player_name", "role", "from_league", "to_league",
            "from_season", "to_season", "age_from", "age_to", "holdout",
        ],
    ).to_csv(tmp_path / "transfers.csv", index=False)
    cohort = build_cohort(tmp_path, chadwick_path=reg)
    assert len(cohort) == 1
    assert cohort.iloc[0]["direction"] == "export"
    assert cohort.iloc[0]["pre_age_source"] == ""
    assert cohort.iloc[0]["post_age_source"] == ""
