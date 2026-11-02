"""Park factors from home/road splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR


def estimate_park_factors(
    seasons: pd.DataFrame,
    *,
    stat: str = "woba",
    min_pa: float = 200.0,
) -> pd.DataFrame:
    """Estimate team-season park factors as home/road ratio, regressed to 1.0.

    For pitchers pass stat='era' (factors inverted so >1 means hitter-friendly).
    """
    home_col = f"home_{stat}"
    road_col = f"road_{stat}"
    if home_col not in seasons.columns or road_col not in seasons.columns:
        return pd.DataFrame(columns=["league", "team", "season", "park_factor", "n"])

    df = seasons.dropna(subset=[home_col, road_col]).copy()
    df = df[df["home_pa"] + df["road_pa"] >= min_pa]
    if df.empty:
        return pd.DataFrame(columns=["league", "team", "season", "park_factor", "n"])

    grouped = (
        df.groupby(["league", "team", "season"], as_index=False)
        .agg(
            home=(home_col, "mean"),
            road=(road_col, "mean"),
            n=("home_pa", "sum"),
        )
    )
    raw = grouped["home"] / grouped["road"].replace(0, pd.NA)
    # Shrink toward 1.0 with weight rising in sample size
    k = 500.0
    weight = grouped["n"] / (grouped["n"] + k)
    grouped["park_factor"] = 1.0 + weight * (raw.fillna(1.0) - 1.0)
    return grouped[["league", "team", "season", "park_factor", "n"]]


def neutralize(rate: float, park_factor: float, *, center: float = 1.0) -> float:
    """Strip park effect toward league-neutral."""
    if park_factor <= 0:
        return rate
    return rate / park_factor * center


# ---------------------------------------------------------------------------
# Runs park factors for NPB / KBO from game logs
# ---------------------------------------------------------------------------

GAME_LOG_COLUMNS = [
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_runs",
    "away_runs",
    "park_id",
    "source",
    "is_synthetic",
]
RUNS_PARK_COLUMNS = ["league", "team", "season", "park_factor", "games"]
RUNS_PARK_LEAGUES = ("NPB", "KBO")

# Regression constant, in games. A team's pooled home+road game count G is
# shrunk toward 1.0 with weight G / (G + K). K = 300 puts a full 3-season
# window (~430 games in NPB/KBO's 143-144 game schedules) at ~59% weight on
# the observed ratio and a single season (~143 games) at ~32%. Single-park
# runs ratios have a large noise floor (a 3-season team sample is still only
# ~215 home games), so we deliberately trust the data less than half at one
# season and a bit over half at three, rather than the ~500-PA-style constant
# used for Statcast splits.
DEFAULT_RUNS_PARK_K = 300.0
DEFAULT_RUNS_PARK_WINDOW = 3

# Stats that move with run environment (park effect applied at half strength).
# bb_pct, k_pct and babip are left alone: walks and strikeouts are plate-
# discipline outcomes that parks barely move, and BABIP's park signal is small
# and mostly defense/surface, so dividing it by a runs ratio would add noise.
_PARK_BAT_STATS = ("woba", "iso", "hr_pct")
_PARK_PIT_STATS = ("era", "fip", "hr_fb")


def _empty_runs_park() -> pd.DataFrame:
    return pd.DataFrame(columns=RUNS_PARK_COLUMNS)


def _real_logs(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    if "is_synthetic" in df.columns:
        flag = df["is_synthetic"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        df = df.loc[~flag]
    for col in ("home_runs", "away_runs", "season"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["home_team", "away_team", "home_runs", "away_runs", "season"])


def estimate_runs_park_factors(
    logs: pd.DataFrame,
    *,
    window: int = DEFAULT_RUNS_PARK_WINDOW,
    k: float = DEFAULT_RUNS_PARK_K,
) -> pd.DataFrame:
    """Runs park factors per league-team-season from game logs.

    Only real rows (``is_synthetic`` False) are used. The home team is the park
    proxy. When ``park_id`` is present, only home games at the team's modal
    park for that season count as home games (neutral-site or relocated games
    are ignored). For season S the sample pools seasons S-window+1..S:

        raw = (runs per game in the team's home games, both teams)
              / (runs per game in the team's road games, both teams)
        park_factor = 1 + G / (G + k) * (raw - 1),   G = home + road games

    Output columns: league, team, season, park_factor, games (G). A team-season
    appears only if the team played in that season itself and the pooled window
    has both home and road games.
    """
    if logs is None or logs.empty:
        return _empty_runs_park()
    df = _real_logs(logs)
    if df.empty:
        return _empty_runs_park()
    df["total_runs"] = df["home_runs"] + df["away_runs"]
    df["season"] = df["season"].astype(int)

    has_park = "park_id" in df.columns
    if has_park:
        df["park_id"] = df["park_id"].astype(object).where(df["park_id"].notna(), "")
        df["park_id"] = df["park_id"].astype(str).str.strip()

    out: list[dict] = []
    for league, lg in df.groupby("league"):
        # per-season home-park filter
        if has_park:
            modal = (
                lg[lg["park_id"] != ""]
                .groupby(["home_team", "season"])["park_id"]
                .agg(lambda s: s.value_counts().index[0])
            )
            keep = []
            for team, season, park in zip(
                lg["home_team"], lg["season"], lg["park_id"], strict=True
            ):
                m = modal.get((team, season))
                keep.append(park == "" or m is None or park == m)
            home_df = lg.loc[keep]
        else:
            home_df = lg
        home_by = home_df.groupby(["home_team", "season"])["total_runs"].agg(["sum", "count"])
        road_by = lg.groupby(["away_team", "season"])["total_runs"].agg(["sum", "count"])
        teams = sorted(set(home_df["home_team"]) | set(lg["away_team"]))
        seasons = sorted(lg["season"].unique())
        for team in teams:
            for season in seasons:
                if (team, season) not in home_by.index and (team, season) not in road_by.index:
                    continue
                hr = hg = rr = rg = 0.0
                for s in range(season - window + 1, season + 1):
                    if (team, s) in home_by.index:
                        hr += float(home_by.loc[(team, s), "sum"])
                        hg += float(home_by.loc[(team, s), "count"])
                    if (team, s) in road_by.index:
                        rr += float(road_by.loc[(team, s), "sum"])
                        rg += float(road_by.loc[(team, s), "count"])
                if hg <= 0 or rg <= 0 or rr <= 0:
                    continue
                raw = (hr / hg) / (rr / rg)
                games = hg + rg
                weight = games / (games + k)
                out.append(
                    {
                        "league": league,
                        "team": team,
                        "season": int(season),
                        "park_factor": 1.0 + weight * (raw - 1.0),
                        "games": int(games),
                    }
                )
    if not out:
        return _empty_runs_park()
    return pd.DataFrame(out, columns=RUNS_PARK_COLUMNS)


def load_runs_park_factors(
    snapshot_dir: Path | str | None = None,
    *,
    window: int = DEFAULT_RUNS_PARK_WINDOW,
    k: float = DEFAULT_RUNS_PARK_K,
) -> pd.DataFrame:
    """Build the runs park-factor table from ``game_logs_{npb,kbo}.csv``.

    Returns an empty frame (with the output columns) when the files are missing
    or have no rows, so callers can treat "no logs" as "no park adjustment".
    """
    root = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    frames: list[pd.DataFrame] = []
    for league in RUNS_PARK_LEAGUES:
        path = root / f"game_logs_{league.lower()}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return _empty_runs_park()
    return estimate_runs_park_factors(pd.concat(frames, ignore_index=True), window=window, k=k)


def lookup_runs_park_factor(
    park_factors: pd.DataFrame | None, league: object, team: object, season: object
) -> float | None:
    """Park factor for a league-team-season, or None when unavailable."""
    if park_factors is None or park_factors.empty:
        return None
    if str(league) not in RUNS_PARK_LEAGUES:
        return None
    try:
        season_i = int(season)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    hit = park_factors[
        (park_factors["league"] == str(league))
        & (park_factors["team"] == str(team))
        & (park_factors["season"] == season_i)
    ]
    if hit.empty:
        return None
    pf = float(hit.iloc[0]["park_factor"])
    return pf if pf > 0 and pd.notna(pf) else None


def neutralize_line_runs(line: pd.Series, park_factor: float | None, role: str) -> pd.Series:
    """Remove half the runs park effect from a line's run-environment stats.

    divisor = (1 + park_factor) / 2. Batters: woba, iso, hr_pct. Pitchers: era,
    fip, hr_fb. bb_pct, k_pct and babip are left unchanged (see module notes
    above). No-op when ``park_factor`` is missing or non-positive.
    """
    if park_factor is None or pd.isna(park_factor) or float(park_factor) <= 0:
        return line
    divisor = (1.0 + float(park_factor)) / 2.0
    out = line.copy()
    stats = _PARK_BAT_STATS if role == "batter" else _PARK_PIT_STATS
    for stat in stats:
        if stat in out.index and pd.notna(out[stat]):
            out[stat] = float(out[stat]) / divisor
    return out
