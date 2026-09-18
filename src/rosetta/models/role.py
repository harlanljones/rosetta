"""Pitcher role adjustments (SP ↔ RP).

NPB/KBO aces often become MLB relievers. Rate translation still applies; this
module applies a transparent additive shift to rate components when a role
change is declared. Defaults are conservative placeholders — replace with
empirical SP→RP deltas from your org sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PitchRole = Literal["SP", "RP", "unknown"]


@dataclass(frozen=True)
class RoleShift:
    """Additive shifts applied after league translation (MLB-scale rates)."""

    k_pct: float = 0.0
    bb_pct: float = 0.0
    hr_fb: float = 0.0
    fip: float = 0.0
    era: float = 0.0


# Literature-shaped defaults: relievers miss bats more, walk a bit more, shorter leverage
SP_TO_RP = RoleShift(k_pct=0.03, bb_pct=0.005, hr_fb=-0.005, fip=-0.25, era=-0.20)
RP_TO_SP = RoleShift(k_pct=-0.025, bb_pct=-0.003, hr_fb=0.008, fip=0.30, era=0.35)


def detect_role(gs: float | None, g: float | None) -> PitchRole:
    if gs is None or g is None or g <= 0:
        return "unknown"
    share = gs / g
    if share >= 0.6:
        return "SP"
    if share <= 0.2:
        return "RP"
    return "unknown"


def role_shift(from_role: PitchRole, to_role: PitchRole) -> RoleShift:
    if from_role == to_role or "unknown" in (from_role, to_role):
        return RoleShift()
    if from_role == "SP" and to_role == "RP":
        return SP_TO_RP
    if from_role == "RP" and to_role == "SP":
        return RP_TO_SP
    return RoleShift()


def apply_role_shift(rates: dict[str, float], shift: RoleShift) -> dict[str, float]:
    out = dict(rates)
    for key, delta in (
        ("k_pct", shift.k_pct),
        ("bb_pct", shift.bb_pct),
        ("hr_fb", shift.hr_fb),
        ("fip", shift.fip),
        ("era", shift.era),
    ):
        if key in out and delta:
            out[key] = out[key] + delta
    return out
