"""Minor/winter-league ingest via the MLB Stats API (no auth, JSON).

Fills the AAA bridge — the most trusted chain link — plus any other sportId
(AA/A/winter/indy) with one client. Writes normalized snapshot CSVs compatible
with `rosetta.ingest.loaders`. Explicit fetch step only; never CI-live.

Endpoint shape (verified live 2026-09-18):
  GET /api/v1/stats?stats=season&group=hitting|pitching&season=YYYY
      &sportId=11&leagueIds=..&limit=N&offset=M
  -> {"stats": [{"splits": [{"season", "stat": {...counting stats...},
      "team": {"id","name"}, "player": {"id","fullName"},
      "league": {"id","name"}, "position": {...}}]}]}

  Coverage caveat: the season endpoint returns a limited pool per sport
  (AAA-2024: 136 hitting / 22 pitching splits — effectively qualified
  players), not every rostered player. Treat it as a high-signal bridge
  sample plus live-update feed, not a census; BR register exports (#2)
  remain the depth source.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

BASE_URL = "https://statsapi.mlb.com/api/v1"

# sportId → Rosetta league label. Winter (17) and indy (23) levels need
# leagueIds to separate (e.g. LIDOM=131); the label can be overridden per call.
SPORT_LEAGUE = {
    1: "MLB",
    11: "AAA",
    12: "AA",
    13: "A+",
    14: "A",
    16: "Rookie",
}

# Fixed wOBA weights (2024-scale) for rebuilding wOBA from counting stats.
WOBA_W = {"ubb": 0.69, "hbp": 0.72, "1b": 0.89, "2b": 1.27, "3b": 1.62, "hr": 2.10}
FIP_CONSTANT = 3.10


def _get_json(params: dict[str, Any], *, timeout: int) -> dict:
    """GET the Stats API with stdlib only (no extra dependency)."""
    url = f"{BASE_URL}/stats?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-mle/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_splits(
    group: str,
    season: int,
    sport_id: int,
    *,
    league_ids: str | None = None,
    limit: int = 200,
    timeout: int = 60,
) -> list[dict]:
    """Fetch all season-stat splits for a group/sport, following pagination."""
    splits: list[dict] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "stats": "season",
            "group": group,
            "season": season,
            "sportId": sport_id,
            "playerPool": "ALL",
            "limit": limit,
            "offset": offset,
        }
        if league_ids:
            params["leagueIds"] = league_ids
        body = _get_json(params, timeout=timeout)
        stats = body.get("stats") or []
        if not stats:
            break
        page = stats[0].get("splits") or []
        if not page:
            break
        splits.extend(page)
        total = stats[0].get("totalSplits", len(page))
        offset += len(page)
        if offset >= total or len(page) < limit:
            break
    return splits


def parse_ip(ip: Any) -> float:
    """Parse baseball IP notation ('178.2' → 178 + 2/3)."""
    try:
        whole, _, frac = str(ip).partition(".")
        return float(whole or 0) + float(frac or 0) / 3.0
    except (ValueError, TypeError):
        return 0.0


def _num(stat: dict, *names: str) -> float:
    for n in names:
        v = stat.get(n)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return 0.0


def normalize_hitting(splits: list[dict], season: int, league_label: str) -> pd.DataFrame:
    """Map Stats API hitting splits → Rosetta hitter schema."""
    rows: list[dict] = []
    for sp in splits:
        st = sp.get("stat") or {}
        player = sp.get("player") or {}
        team = sp.get("team") or {}
        pa = _num(st, "plateAppearances")
        if pa <= 0:
            continue
        ab = _num(st, "atBats")
        so = _num(st, "strikeOuts")
        bb = _num(st, "baseOnBalls")
        ibb = _num(st, "intentionalWalks")
        hbp = _num(st, "hitByPitch")
        doubles = _num(st, "doubles")
        triples = _num(st, "triples")
        hr = _num(st, "homeRuns")
        hits = _num(st, "hits")
        singles = hits - doubles - triples - hr
        denom = pa if pa else 1.0
        bb_pct = (bb + ibb) / denom
        k_pct = so / denom
        iso = ((doubles + 2 * triples + 3 * hr) / ab) if ab else 0.0
        bip = ab - so - hr
        babip = ((hits - hr) / bip) if bip else 0.0
        hr_pct = hr / denom
        woba = (
            WOBA_W["ubb"] * (bb - ibb)
            + WOBA_W["hbp"] * hbp
            + WOBA_W["1b"] * singles
            + WOBA_W["2b"] * doubles
            + WOBA_W["3b"] * triples
            + WOBA_W["hr"] * hr
        ) / denom
        try:
            age = float(st.get("age", 27))
        except (ValueError, TypeError):
            age = 27.0
        g = _num(st, "gamesPlayed")
        rows.append(
            {
                "player_id": f"mlbam-{player.get('id', '')}",
                "player_name": player.get("fullName", ""),
                "season": season,
                "league": league_label,
                "team": team.get("name", ""),
                "role": "batter",
                "age": age,
                "pa": pa,
                "g": g,
                "bb_pct": bb_pct,
                "k_pct": k_pct,
                "iso": iso,
                "babip": babip,
                "hr_pct": hr_pct,
                "avg": _num(st, "avg"),
                "obp": _num(st, "obp"),
                "slg": _num(st, "slg"),
                "woba": woba,
                # No season-level home/road splits in this endpoint (see §3 of
                # docs/research/data-sources.md); neutral placeholders so the
                # park module treats these rows as league-average context.
                "home_pa": pa * 0.5,
                "road_pa": pa * 0.5,
                "home_woba": woba,
                "road_woba": woba,
                "park_id": team.get("name", ""),
                "source": "statsapi",
                "is_synthetic": False,
            }
        )
    out = pd.DataFrame(rows, columns=HITTER_COLUMNS)
    return out


def normalize_pitching(splits: list[dict], season: int, league_label: str) -> pd.DataFrame:
    """Map Stats API pitching splits → Rosetta pitcher schema."""
    rows: list[dict] = []
    for sp in splits:
        st = sp.get("stat") or {}
        player = sp.get("player") or {}
        team = sp.get("team") or {}
        bf = _num(st, "battersFaced")
        ip = parse_ip(st.get("inningsPitched", 0))
        if bf <= 0 or ip <= 0:
            continue
        so = _num(st, "strikeOuts")
        bb = _num(st, "baseOnBalls")
        ibb = _num(st, "intentionalWalks")
        hbp = _num(st, "hitByPitch")
        hr = _num(st, "homeRuns")
        air_outs = _num(st, "airOuts")
        k_pct = so / bf
        bb_pct = (bb + ibb) / bf
        # Endpoint has no batted-ball split; approximate HR/FB as HR over
        # (HR + air outs). Documented approximation — prefer Savant for
        # precision (see docs/research/data-sources.md §3.4).
        hr_fb = hr / (hr + air_outs) if (hr + air_outs) else 0.12
        era = _num(st, "era")
        fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
        try:
            age = float(st.get("age", 27))
        except (ValueError, TypeError):
            age = 27.0
        rows.append(
            {
                "player_id": f"mlbam-{player.get('id', '')}",
                "player_name": player.get("fullName", ""),
                "season": season,
                "league": league_label,
                "team": team.get("name", ""),
                "role": "pitcher",
                "age": age,
                "ip": ip,
                "g": _num(st, "gamesPitched"),
                "gs": _num(st, "gamesStarted"),
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "hr_fb": hr_fb,
                "era": era,
                "fip": fip,
                "home_pa": bf * 0.5,
                "road_pa": bf * 0.5,
                "home_era": era,
                "road_era": era,
                "park_id": team.get("name", ""),
                "source": "statsapi",
                "is_synthetic": False,
            }
        )
    out = pd.DataFrame(rows, columns=PITCHER_COLUMNS)
    return out


def fetch_sport_players(season: int, sport_id: int, *, timeout: int = 60) -> list[dict]:
    """Full player pool for a sport+season (ids, birthdates, handedness, teams)."""
    url = f"{BASE_URL}/sports/{sport_id}/players?season={season}"
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-mle/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    return body.get("people") or []


def fetch_person(person_id: int | str, *, timeout: int = 60) -> dict:
    """Single player record (birthDate, bats/throws, positions)."""
    url = f"{BASE_URL}/people/{person_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-mle/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    people = body.get("people") or []
    return people[0] if people else {}


def age_at_season(birth_date: str | None, season: int) -> float | None:
    """Mid-season age from an ISO birthdate; None when unknown."""
    try:
        return float(season - int(str(birth_date)[:4]))
    except (ValueError, TypeError):
        return None


def attach_ages(
    df: pd.DataFrame, season: int, sport_id: int, *, people: list[dict] | None = None
) -> pd.DataFrame:
    """Overwrite `age` with people-pool birth years where IDs match."""
    out = df.copy()
    if out.empty:
        return out
    pool = people if people is not None else fetch_sport_players(season, sport_id)
    ages: dict[str, float] = {}
    for p in pool:
        age = age_at_season(p.get("birthDate"), season)
        if age is not None:
            ages[f"mlbam-{p.get('id', '')}"] = age
    if ages:
        mapped = out["player_id"].map(ages)
        out["age"] = mapped.where(mapped.notna(), out["age"])
    return out


def fetch_league_season(
    season: int,
    sport_id: int,
    *,
    league_label: str | None = None,
    league_ids: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch + normalize one season for a sportId (default label from map)."""
    label = league_label or SPORT_LEAGUE.get(sport_id, f"sport-{sport_id}")
    hitting = fetch_splits("hitting", season, sport_id, league_ids=league_ids)
    pitching = fetch_splits("pitching", season, sport_id, league_ids=league_ids)
    return (
        normalize_hitting(hitting, season, label),
        normalize_pitching(pitching, season, label),
    )


def write_statsapi_snapshots(
    seasons: list[int],
    sport_id: int,
    *,
    league_label: str | None = None,
    league_ids: str | None = None,
    out_dir: Path | None = None,
    append: bool = True,
    with_ages: bool = True,
) -> Path:
    """Fetch seasons via the Stats API and merge into league snapshot CSVs."""
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    label = league_label or SPORT_LEAGUE.get(sport_id, f"sport-{sport_id}")
    slug = label.lower()
    bats: list[pd.DataFrame] = []
    pits: list[pd.DataFrame] = []
    for season in seasons:
        b, p = fetch_league_season(season, sport_id, league_label=label, league_ids=league_ids)
        if with_ages and (not b.empty or not p.empty):
            try:
                pool = fetch_sport_players(season, sport_id)
            except Exception:
                pool = []
            b = attach_ages(b, season, sport_id, people=pool)
            p = attach_ages(p, season, sport_id, people=pool)
        bats.append(b)
        pits.append(p)
    bat_df = pd.concat(bats, ignore_index=True) if bats else pd.DataFrame(columns=HITTER_COLUMNS)
    pit_df = pd.concat(pits, ignore_index=True) if pits else pd.DataFrame(columns=PITCHER_COLUMNS)
    for df, name in ((bat_df, f"{slug}_batters.csv"), (pit_df, f"{slug}_pitchers.csv")):
        path = root / name
        if append and path.exists() and not df.empty:
            old = pd.read_csv(path)
            # Merge semantics: synthetic fixture rows are always preserved;
            # real rows are replaced only for seasons being written now.
            written_keys = {(row["league"], row["season"]) for row in df.to_dict("records")}
            old = old[
                (old["is_synthetic"].astype(str) == "True")
                | ~old.apply(lambda r, wk=written_keys: (r["league"], r["season"]) in wk, axis=1)
            ]
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                ["player_id", "season", "league"], keep="last"
            )
        if not df.empty:
            df.to_csv(path, index=False)
    return root
