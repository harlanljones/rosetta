"""League ingestion: snapshot loaders and optional live scrapers."""

from rosetta.ingest.loaders import load_all_seasons, load_league_seasons

__all__ = ["load_all_seasons", "load_league_seasons"]
