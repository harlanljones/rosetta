"""Optional live scrapers.

Network scrapers are intentionally thin stubs with clear extension points.
CI and `make backtest` use committed snapshots only — never this module.
"""

from __future__ import annotations

from typing import Any


def scrape_mlb_fangraphs(season: int) -> list[dict[str, Any]]:
    """Pull MLB seasonal lines via pybaseball when installed.

    Raises RuntimeError with install guidance if optional deps are missing.
    """
    try:
        from pybaseball import batting_stats, pitching_stats
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install optional scrape extras: pip install -e '.[scrape]'") from exc

    bat = batting_stats(season, qual=50)
    pit = pitching_stats(season, qual=20)
    return [{"role": "batter", "frame": bat}, {"role": "pitcher", "frame": pit}]


def scrape_league(league: str, season: int) -> list[dict[str, Any]]:
    """Placeholder for NPB/KBO/CPBL/BR scrapers.

    Real implementations should:
    1. Fetch pages with rate limits + User-Agent
    2. Normalize into schema.SeasonLine columns
    3. Write parquet/CSV under data/snapshots/
    """
    raise NotImplementedError(
        f"Live scraper for {league} {season} not implemented. "
        "Use committed snapshots or contribute a scraper under rosetta.ingest."
    )
