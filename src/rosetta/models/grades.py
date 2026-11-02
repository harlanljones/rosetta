"""wRC+, FIP-, 20-80 grades, WAR and dollars from translated wOBA / FIP.

Reference constants come from ``data/reference/mlb_constants.csv`` (see
``load_constants``). Nothing here bakes in a published constant: a missing
constant yields ``None`` and every dependent output is ``None`` too.

Every transform is monotone in its input (wOBA or FIP), so an 80% band on the
input maps endpoint-for-endpoint to a band on the output. Transforms that
decrease in FIP (grade, WAR, dollars for pitchers) swap the endpoints so the
reported ``low <= high`` in the output metric's own orientation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rosetta.paths import DATA_DIR, SNAPSHOT_DIR

DEFAULT_REFERENCE_PATH = DATA_DIR / "reference" / "mlb_constants.csv"

CONSTANT_COLUMNS: tuple[str, ...] = (
    "woba_scale",
    "lg_woba",
    "lg_r_pa",
    "runs_per_win",
    "lg_fip",
    "fip_constant",
    "dollars_per_war",
    "pitcher_repl_r9",
)

# Spec D4 playing time and D3 qualification thresholds.
DEFAULT_PA = 600.0
DEFAULT_IP_STARTER = 170.0
DEFAULT_IP_RELIEVER = 65.0
RELIEVER_GS_SHARE = 0.5
QUALIFIED_PA = 300.0
QUALIFIED_IP = 100.0

# Spec WP6: positional adjustment (runs per 162 games), scaled by PA / 700.
POS_ADJ_PER_162: dict[str, float] = {
    "C": 12.5,
    "SS": 7.5,
    "2B": 2.5,
    "3B": 2.5,
    "CF": 2.5,
    "LF": -7.5,
    "RF": -7.5,
    "1B": -12.5,
    "DH": -17.5,
}
POS_ADJ_PA = 700.0
REPL_RUNS_PER_600_PA = 20.0


@dataclass
class MlbConstants:
    """Reference constants for one MLB season. ``None`` means unavailable."""

    season: int
    woba_scale: float | None = None
    lg_woba: float | None = None
    lg_r_pa: float | None = None
    runs_per_win: float | None = None
    lg_fip: float | None = None
    fip_constant: float | None = None
    dollars_per_war: float | None = None
    pitcher_repl_r9: float | None = None
    source: str = ""
    # Qualified-player reference inputs for the 20-80 grade (raw wOBA / FIP).
    ref_woba: list[float] = field(default_factory=list)
    ref_fip: list[float] = field(default_factory=list)
    ref_ip: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _clean(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _real_rows(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.loc[df["season"] == season]
    if "is_synthetic" in out.columns:
        flag = out["is_synthetic"].astype(str).str.strip().str.lower().isin({"true", "1"})
        out = out.loc[~flag]
    return out


def _read_snapshot(snapshot_dir: Path, name: str) -> pd.DataFrame | None:
    path = snapshot_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_constants(
    season: int,
    snapshot_dir: Path | str | None = None,
    reference_path: Path | str | None = None,
) -> MlbConstants:
    """Read the constants table row for ``season`` and fill what it lacks.

    Missing ``lg_woba`` is computed as the PA-weighted mean wOBA of real
    (``is_synthetic`` False) MLB batters; missing ``lg_fip`` as the IP-weighted
    mean FIP of real MLB pitchers. All other missing constants stay ``None``.
    """
    snap = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    ref = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
    c = MlbConstants(season=int(season))

    if ref.exists():
        table = pd.read_csv(ref)
        if "season" in table.columns:
            rows = table.loc[table["season"] == season]
            if not rows.empty:
                row = rows.iloc[-1]
                for col in CONSTANT_COLUMNS:
                    if col in table.columns:
                        setattr(c, col, _clean(row[col]))
                if "source" in table.columns and isinstance(row["source"], str):
                    c.source = row["source"]

    batters = _read_snapshot(snap, "mlb_batters.csv")
    if batters is not None:
        b = _real_rows(batters, season)
        b = b.dropna(subset=["woba", "pa"])
        b = b.loc[b["pa"] > 0]
        if c.lg_woba is None and not b.empty:
            c.lg_woba = float(np.average(b["woba"], weights=b["pa"]))
            c.notes.append("lg_woba computed from MLB snapshot (PA-weighted).")
        q = b.loc[b["pa"] >= QUALIFIED_PA]
        c.ref_woba = [float(x) for x in q["woba"]]

    pitchers = _read_snapshot(snap, "mlb_pitchers.csv")
    if pitchers is not None:
        p = _real_rows(pitchers, season)
        p = p.dropna(subset=["fip", "ip"])
        p = p.loc[p["ip"] > 0]
        if c.lg_fip is None and not p.empty:
            c.lg_fip = float(np.average(p["fip"], weights=p["ip"]))
            c.notes.append("lg_fip computed from MLB snapshot (IP-weighted).")
        q = p.loc[p["ip"] >= QUALIFIED_IP]
        c.ref_fip = [float(x) for x in q["fip"]]

    return c


def wrc_plus(woba: float | None, c: MlbConstants) -> float | None:
    if woba is None or c.lg_woba is None or c.woba_scale is None or c.lg_r_pa is None:
        return None
    if c.woba_scale == 0 or c.lg_r_pa == 0:
        return None
    return 100.0 * (((woba - c.lg_woba) / c.woba_scale) + c.lg_r_pa) / c.lg_r_pa


def fip_minus(fip: float | None, c: MlbConstants) -> float | None:
    if fip is None or c.lg_fip is None or c.lg_fip == 0:
        return None
    return 100.0 * fip / c.lg_fip


def _round_to_5(x: float) -> float:
    return math.floor(x / 5.0 + 0.5) * 5.0


def grade_20_80(
    value: float | None,
    reference_values: Sequence[float],
    higher_is_better: bool = True,
) -> float | None:
    """20-80 grade: 50 at the reference median, 10 points per SD (D3).

    Clipped to [20, 80] and rounded to the nearest 5. ``None`` when the
    reference has fewer than two finite values or zero spread.
    """
    if value is None or math.isnan(value):
        return None
    ref = np.asarray([v for v in reference_values if v is not None and not math.isnan(v)], float)
    if ref.size < 2:
        return None
    sd = float(ref.std(ddof=1))
    if sd <= 0:
        return None
    z = (value - float(np.median(ref))) / sd
    if not higher_is_better:
        z = -z
    return _round_to_5(min(80.0, max(20.0, 50.0 + 10.0 * z)))


def batter_reference(c: MlbConstants) -> tuple[list[float], str]:
    """Qualified-batter reference values and the metric they are in.

    wRC+ when computable from the constants, else raw wOBA.
    """
    if wrc_plus(0.320, c) is not None:
        vals = [v for v in (wrc_plus(w, c) for w in c.ref_woba) if v is not None]
        return vals, "wrc_plus"
    return list(c.ref_woba), "woba"


def pitcher_reference(c: MlbConstants) -> list[float]:
    """Qualified-pitcher FIP- values (empty when lg_fip is unavailable)."""
    return [v for v in (fip_minus(f, c) for f in c.ref_fip) if v is not None]


def grade_batter(woba: float | None, c: MlbConstants) -> float | None:
    if woba is None:
        return None
    vals, metric = batter_reference(c)
    value = wrc_plus(woba, c) if metric == "wrc_plus" else woba
    return grade_20_80(value, vals, higher_is_better=True)


def grade_pitcher(fip: float | None, c: MlbConstants) -> float | None:
    value = fip_minus(fip, c)
    return grade_20_80(value, pitcher_reference(c), higher_is_better=False)


def war_batter(
    woba: float | None,
    pa: float,
    position: str | None,
    def_runs: float | None,
    c: MlbConstants,
    notes: list[str] | None = None,
) -> float | None:
    """Batter WAR at ``pa`` plate appearances. ``None`` if constants are missing."""
    if woba is None or c.lg_woba is None or c.woba_scale is None or c.runs_per_win is None:
        return None
    if c.woba_scale == 0 or c.runs_per_win == 0:
        return None
    batting_runs = (woba - c.lg_woba) / c.woba_scale * pa

    pos = (position or "").strip().upper()
    if pos in POS_ADJ_PER_162:
        pos_adj = POS_ADJ_PER_162[pos] * pa / POS_ADJ_PA
    else:
        pos_adj = 0.0
        if notes is not None:
            notes.append(f"Unknown position {position!r}; positional adjustment taken as 0.")

    if def_runs is None or math.isnan(def_runs):
        def_runs = 0.0
        if notes is not None:
            notes.append("def_runs missing; fielding runs taken as 0.")

    repl = REPL_RUNS_PER_600_PA * pa / 600.0
    return (batting_runs + pos_adj + def_runs + repl) / c.runs_per_win


def war_pitcher(
    fip: float | None,
    ip: float,
    c: MlbConstants,
    repl_runs_per_9: float | None = None,
) -> float | None:
    """FIP-based pitcher WAR.

    ``repl_runs_per_9`` is runs above replacement per 9 IP, added to the
    run-prevention term; it falls back to ``c.pitcher_repl_r9`` and has no
    built-in default, so WAR is ``None`` until the table supplies it.
    """
    r9 = repl_runs_per_9 if repl_runs_per_9 is not None else c.pitcher_repl_r9
    if fip is None or c.lg_fip is None or c.runs_per_win is None or r9 is None:
        return None
    if c.runs_per_win == 0:
        return None
    repl_p = ip / 9.0 * r9
    return ((c.lg_fip - fip) / 9.0 * ip + repl_p) / c.runs_per_win


def dollars(war: float | None, c: MlbConstants) -> float | None:
    if war is None or c.dollars_per_war is None:
        return None
    return war * c.dollars_per_war


def pitcher_innings(
    gs: float | None, g: float | None, target_pitch_role: str | None = None
) -> tuple[float, str]:
    """Default IP per D4 and the role used ("SP" or "RP")."""
    if target_pitch_role == "SP":
        return DEFAULT_IP_STARTER, "SP"
    if target_pitch_role == "RP":
        return DEFAULT_IP_RELIEVER, "RP"
    if gs is not None and g is not None and not math.isnan(gs) and not math.isnan(g) and g > 0:
        if gs / g < RELIEVER_GS_SHARE:
            return DEFAULT_IP_RELIEVER, "RP"
    return DEFAULT_IP_STARTER, "SP"


@dataclass
class GradeBundle:
    """Point estimate with low/high for each grade-family metric."""

    wrc_plus: tuple[float | None, float | None, float | None] = (None, None, None)
    fip_minus: tuple[float | None, float | None, float | None] = (None, None, None)
    grade: tuple[float | None, float | None, float | None] = (None, None, None)
    war: tuple[float | None, float | None, float | None] = (None, None, None)
    dollars: tuple[float | None, float | None, float | None] = (None, None, None)
    position: str = ""
    notes: list[str] = field(default_factory=list)


def _ordered(
    point: float | None, a: float | None, b: float | None
) -> tuple[float | None, float | None, float | None]:
    """Return (point, low, high) given the two transformed band endpoints."""
    if a is None or b is None:
        return (point, None, None)
    return (point, min(a, b), max(a, b))


def grade_batter_line(
    woba: float,
    woba_low: float,
    woba_high: float,
    pa: float,
    position: str | None,
    def_runs: float | None,
    c: MlbConstants,
) -> GradeBundle:
    out = GradeBundle(position=(position or "").strip().upper())
    out.wrc_plus = _ordered(
        wrc_plus(woba, c), wrc_plus(woba_low, c), wrc_plus(woba_high, c)
    )
    out.grade = _ordered(
        grade_batter(woba, c), grade_batter(woba_low, c), grade_batter(woba_high, c)
    )
    # Notes are recorded once, from the point estimate.
    out.war = _ordered(
        war_batter(woba, pa, position, def_runs, c, out.notes),
        war_batter(woba_low, pa, position, def_runs, c),
        war_batter(woba_high, pa, position, def_runs, c),
    )
    out.dollars = _ordered(
        dollars(out.war[0], c), dollars(out.war[1], c), dollars(out.war[2], c)
    )
    return out


def grade_pitcher_line(
    fip: float,
    fip_low: float,
    fip_high: float,
    ip: float,
    c: MlbConstants,
) -> GradeBundle:
    out = GradeBundle()
    out.fip_minus = _ordered(
        fip_minus(fip, c), fip_minus(fip_low, c), fip_minus(fip_high, c)
    )
    out.grade = _ordered(
        grade_pitcher(fip, c), grade_pitcher(fip_low, c), grade_pitcher(fip_high, c)
    )
    out.war = _ordered(
        war_pitcher(fip, ip, c), war_pitcher(fip_low, ip, c), war_pitcher(fip_high, ip, c)
    )
    out.dollars = _ordered(
        dollars(out.war[0], c), dollars(out.war[1], c), dollars(out.war[2], c)
    )
    return out
