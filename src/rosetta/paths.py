"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PKG_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
OUTPUT_DIR = DATA_DIR / "outputs"


def ensure_data_dirs() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
