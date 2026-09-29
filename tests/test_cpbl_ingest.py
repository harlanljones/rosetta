from __future__ import annotations

from pathlib import Path

from rosetta.ingest.cpbl import normalize_batting, parse_recordall_table

FIXTURE = Path(__file__).parent / "fixtures" / "cpbl_recordall_bat_2024.html"


def test_parse_cpbl_recordall_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    rows = parse_recordall_table(html)
    assert len(rows) >= 10
    bat = normalize_batting(rows, 2024)
    assert (bat["league"] == "CPBL").all()
    assert bat["is_synthetic"].eq(False).all()
    assert bat["woba"].between(0.2, 0.6).all()
