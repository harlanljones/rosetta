"""translate(player, from_league) → MLB-equivalent statline + uncertainty."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.factors import HITTER_STATS, PITCHER_STATS, LeagueFactorModel
from rosetta.schema import League, Role, TranslationResult


def _recombine_woba(rates: dict[str, float]) -> float:
    bb = rates.get("bb_pct", 0.08)
    iso = rates.get("iso", 0.15)
    babip = rates.get("babip", 0.30)
    hr = rates.get("hr_pct", 0.03)
    return float(max(0.2, min(0.5, 0.12 + 0.7 * bb + 0.35 * babip + 0.55 * iso + 0.9 * hr)))


def _recombine_fip(rates: dict[str, float]) -> float:
    k = rates.get("k_pct", 0.2)
    bb = rates.get("bb_pct", 0.08)
    hr_fb = rates.get("hr_fb", 0.12)
    return float(max(1.5, min(8.0, 3.2 + 12.0 * bb - 8.0 * k + 10.0 * hr_fb)))


def translate_line(
    line: pd.Series | dict,
    *,
    model: LeagueFactorModel,
    from_league: str | None = None,
    to_league: str = "MLB",
) -> TranslationResult:
    """Translate one normalized season line to the target league."""
    if isinstance(line, dict):
        line = pd.Series(line)
    role = str(line.get("role", "batter"))
    src = from_league or str(line["league"])
    stats = HITTER_STATS if role == "batter" else PITCHER_STATS
    path = model.path(src, to_league)

    rates: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    notes: list[str] = []

    for stat in stats:
        if stat not in line or pd.isna(line[stat]):
            continue
        raw = float(line[stat])
        point, lo_f, hi_f, resid_sd = model.get_interval(role, src, to_league, stat)
        # Factor uncertainty + mover residual noise (~80% ~ 1.28σ)
        center = raw * point
        factor_band = max(abs(raw * hi_f - center), abs(center - raw * lo_f))
        half = max(factor_band, 1.28 * resid_sd)
        rates[stat] = center
        # Keep rate-like stats non-negative in reported bands
        floor = 0.0 if stat not in {"era", "fip"} else 0.5
        lower[stat] = max(floor, center - half)
        upper[stat] = center + half

    # Count movers along path for transparency
    n_movers = 0
    for a, b in zip(path[:-1], path[1:], strict=False):
        key = model.link_key(a, b)
        node = model.links.get(role, {}).get(key, {})
        if node:
            n_movers += int(min(v.get("n", 0) for v in node.values()) if node else 0)

    if src != to_league and n_movers < 15:
        notes.append("Thin mover sample along path — widen interpretation of bands.")
    if src in {"CUBA", "CPBL"}:
        notes.append("Long/thin international chain; uncertainty intentionally wide.")

    def _coerce_league(value: str, default: League) -> League:
        try:
            return League(value)
        except ValueError:
            notes.append(f"Unknown league {value}; treated as {default.value}.")
            return default

    result = TranslationResult(
        player_id=str(line.get("player_id", "")),
        player_name=str(line.get("player_name", "")),
        season=int(line.get("season", 0)),
        from_league=_coerce_league(src, League.AAA),
        to_league=_coerce_league(to_league, League.MLB),
        role=Role.BATTER if role == "batter" else Role.PITCHER,
        age=float(line.get("age", 27)),
        path=path,
        rates=rates,
        lower=lower,
        upper=upper,
        sample_pa=float(line.get("pa", 0) or 0),
        sample_ip=float(line.get("ip", 0) or 0),
        n_movers_path=n_movers,
        notes=notes,
    )

    if role == "batter":
        result.woba = rates.get("woba", _recombine_woba(rates))
        result.woba_low = lower.get("woba", _recombine_woba(lower) if lower else result.woba)
        result.woba_high = upper.get("woba", _recombine_woba(upper) if upper else result.woba)
    else:
        result.fip = rates.get("fip", _recombine_fip(rates))
        result.fip_low = lower.get("fip", _recombine_fip(lower) if lower else result.fip)
        result.fip_high = upper.get("fip", _recombine_fip(upper) if upper else result.fip)

    return result


def translate(
    player: str,
    from_league: str,
    *,
    season: int | None = None,
    role: str = "batter",
    model: LeagueFactorModel | None = None,
    factors_path: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    to_league: str = "MLB",
) -> TranslationResult:
    """Public API: translate a player-season from `from_league` to MLB (default).

    `player` may be a player_id or exact player_name.
    """
    if model is None:
        if factors_path is None:
            raise ValueError("Provide model= or factors_path=")
        model = LeagueFactorModel.load(factors_path)

    seasons = load_all_seasons(role=role, snapshot_dir=Path(snapshot_dir) if snapshot_dir else None)
    mask = (seasons["league"] == from_league) & (
        (seasons["player_id"] == player) | (seasons["player_name"] == player)
    )
    sub = seasons.loc[mask]
    if sub.empty:
        raise LookupError(f"No {role} seasons for {player!r} in {from_league}")
    if season is not None:
        sub = sub.loc[sub["season"] == season]
        if sub.empty:
            raise LookupError(f"No season {season} for {player!r} in {from_league}")
    line = sub.sort_values("season").iloc[-1]
    return translate_line(line, model=model, from_league=from_league, to_league=to_league)


def translate_frame(
    df: pd.DataFrame,
    *,
    model: LeagueFactorModel,
    from_league_col: str = "league",
    to_league: str = "MLB",
) -> pd.DataFrame:
    """Vectorized convenience: translate each row, return flat DataFrame."""
    records = []
    for _, row in df.iterrows():
        tr = translate_line(row, model=model, from_league=str(row[from_league_col]), to_league=to_league)
        flat = tr.model_dump()
        rates = flat.pop("rates")
        lower = flat.pop("lower")
        upper = flat.pop("upper")
        for k, v in rates.items():
            flat[f"mle_{k}"] = v
            flat[f"mle_{k}_low"] = lower.get(k)
            flat[f"mle_{k}_high"] = upper.get(k)
        flat["path"] = "→".join(flat["path"])
        flat["notes"] = "; ".join(flat["notes"])
        records.append(flat)
    return pd.DataFrame(records)
