"""NPB league ingest via npb.jp BIS English pages (no auth, HTML).

Covers the NPB leg of the league chain where the Stats API returns nothing:
season leaderboards per league (Central/Pacific) for batters and pitchers,
merged per player, with birth years from player-card pages for age curves.

Verified live 2026-09-19. Explicit fetch step only; never CI-live.
Be polite: ~1s between requests, snapshot everything.

Structure (static HTML):
  /bis/eng/{season}/stats/bat_c.html   Central batting  (top-30 qualified)
  /bis/eng/{season}/stats/bat_p.html   Pacific batting
  /bis/eng/{season}/stats/pit_c.html   Central pitching (top ~qualified)
  /bis/eng/{season}/stats/pit_p.html   Pacific pitching
  /bis/eng/players/active/index_{a..z}.html   name -> 8-digit BIS player id
  /bis/eng/players/{id}.html           player card ("Born: October 18, 1990")

Caveats:
  - Leaderboards are qualified players only (~30/league), not a full census.
  - 8-digit BIS ids are the Chadwick `key_npb` space, so `npb-{id}` joins the
    crosswalk directly — that is what unlocks real NPB->MLB transfer pairs.
  - No GS column for pitchers -> gs=0. No home/road splits -> neutral
    placeholders. NPB innings may render as fractions ("155 2/3").
  - NPB BB column is treated the same as the KBO module treats BB+IBB
    (documented approximation — wide bootstrap bands).
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from rosetta.ingest.kbo import FIP_CONSTANT, WOBA_W, _row_get
from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

BASE = "https://npb.jp"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}

TEAM_NAMES = {
    "C": "Hiroshima Carp",
    "D": "Chunichi Dragons",
    "G": "Yomiuri Giants",
    "T": "Hanshin Tigers",
    "S": "Yakult Swallows",
    "DB": "DeNA BayStars",
    "L": "Seibu Lions",
    "H": "SoftBank Hawks",
    "E": "Rakuten Eagles",
    "M": "Chiba Lotte Marines",
    "Bs": "Orix Buffaloes",
    "F": "Nippon-Ham Fighters",
}


def stat_url(season: int, kind: str) -> str:
    """Stat-table URL for a season; kind in {bat_c, bat_p, pit_c, pit_p}."""
    return f"{BASE}/bis/eng/{season}/stats/{kind}.html"


ID_INDEX_URLS = [
    f"{BASE}/bis/eng/players/active/index_{c}.html" for c in "abcdefghijklmnopqrstuvwxyz"
]
PLAYER_URL = BASE + "/bis/eng/players/{}.html"


class NPBSession:
    """Polite HTTP session for npb.jp (sequential GETs, ~1s delay)."""

    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay

    def get(self, url: str) -> str:
        time.sleep(self.delay)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")

    def get_season(self, url: str, season: int) -> str:
        """KBO-session-compatible alias (NPB seasons live in the URL)."""
        return self.get(url)


def _clean(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", " ").strip()


def parse_table(html: str) -> tuple[list[str], list[dict]]:
    """Parse an NPB stat table -> (headers, rows with player_name/team/values)."""
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
    table = next((t for t in tables if 'class="ststats"' in t), None)
    if not table:
        return [], []
    header_row = re.search(r"<tr><th.*?</tr>", table, re.S)
    if not header_row:
        return [], []
    headers = [_clean(t) for t in re.findall(r"<th[^>]*>(.*?)</th>", header_row.group(0), re.S)]
    # The Player column header has colspan=2 on some pages (name + team cells),
    # leaving headers one short of row cells — reinsert the Team label.
    first_row = re.search(r'<tr class="ststats">.*?</tr>', table, re.S)
    if first_row:
        n_cells = len(re.findall(r"<td[^>]*>", first_row.group(0)))
        if n_cells == len(headers) + 1 and "Team" not in headers:
            headers = headers[:2] + ["Team"] + headers[2:]
    rows: list[dict] = []
    for tr in re.findall(r'<tr class="ststats">.*?</tr>', table, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(cells) != len(headers):
            continue
        name_m = re.search(r'<td class="stplayer">(.*?)</td>', tr, re.S)
        team_m = re.search(r'<td class="stteam">(.*?)</td>', tr, re.S)
        rows.append(
            {
                "player_name": _clean(name_m.group(1)) if name_m else "",
                "team": _clean(team_m.group(1)).strip("()") if team_m else "",
                "values": [_clean(c) for c in cells],
            }
        )
    return headers, rows


def parse_id_index(html: str) -> dict[str, str]:
    """One alphabetical index page -> {\"Last, First\": 8-digit BIS id}."""
    out: dict[str, str] = {}
    # Blocks nest a player card per anchor — parse anchor-by-anchor.
    for m in re.finditer(
        r'<a href="/bis/eng/players/(\d{8})\.html" class="player_unit[^"]*">(.*?)</a>',
        html,
        re.S,
    ):
        pid, block = m.group(1), m.group(2)
        name_m = re.search(r'<dd class="name">\s*([^<]+?)\s*</dd>', block, re.S)
        if name_m:
            out[" ".join(name_m.group(1).split())] = pid
    return out


def fetch_id_map(*, session: NPBSession | None = None, delay: float = 1.0) -> dict[str, str]:
    """All active NPB players -> BIS ids (26 alphabetical index pages)."""
    sess = session or NPBSession(delay=delay)
    out: dict[str, str] = {}
    for url in ID_INDEX_URLS:
        try:
            out.update(parse_id_index(sess.get(url)))
        except Exception:
            continue
    return out


def fetch_birth_year(html: str) -> int | None:
    """Birth year from a player-card page ('Born: October 18, 1990')."""
    m = re.search(r"<th>Born</th>\s*<td>[A-Za-z]+ \d{1,2}, (\d{4})</td>", html)
    return int(m.group(1)) if m else None


def _to_float(x: str) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _parse_ip(x: str) -> float:
    """NPB innings: decimal ('157') or fraction notation ('155 2/3')."""
    s = str(x).strip()
    m = re.match(r"(\d+)\s+(\d)\s*/\s*(\d)", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    m = re.match(r"(\d+)\.(\d)", s)
    if m and m.group(2) in "012":  # .1/.2 innings-out notation
        return float(m.group(1)) + float(m.group(2)) / 3.0
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _frame(headers: list[str], rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = dict(zip(headers, r["values"], strict=False))
        rec["player_name"] = r["player_name"]
        rec["team"] = r["team"]
        rec["_npb_id"] = ""
        recs.append(rec)
    return pd.DataFrame(recs)


def _merge_leagues(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Stack Central + Pacific tables (same schema, name is the join key)."""
    if d1.empty:
        return d2
    if d2.empty:
        return d1
    return pd.concat([d1, d2], ignore_index=True)


def fetch_hitting(
    season: int, *, session: NPBSession | None = None, id_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Central + Pacific batting tables for a season (raw columns + ids)."""
    sess = session or NPBSession()
    id_map = id_map or {}
    h1, r1 = parse_table(sess.get_season(stat_url(season, "bat_c"), season))
    h2, r2 = parse_table(sess.get_season(stat_url(season, "bat_p"), season))
    df = _merge_leagues(_frame(h1, r1), _frame(h2, r2))
    if not df.empty:
        df["_npb_id"] = df["player_name"].map(id_map).fillna("")
    return df


def fetch_pitching(
    season: int, *, session: NPBSession | None = None, id_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Central + Pacific pitching tables for a season (raw columns + ids)."""
    sess = session or NPBSession()
    id_map = id_map or {}
    h1, r1 = parse_table(sess.get_season(stat_url(season, "pit_c"), season))
    h2, r2 = parse_table(sess.get_season(stat_url(season, "pit_p"), season))
    df = _merge_leagues(_frame(h1, r1), _frame(h2, r2))
    if not df.empty:
        df["_npb_id"] = df["player_name"].map(id_map).fillna("")
    return df


def fetch_birth_years(
    player_ids: list[str], *, session: NPBSession | None = None
) -> dict[str, int]:
    """Birth years keyed by 8-digit BIS id (player-card pages; polite)."""
    sess = session or NPBSession()
    out: dict[str, int] = {}
    for pid in dict.fromkeys(p for p in player_ids if p):
        try:
            year = fetch_birth_year(sess.get(PLAYER_URL.format(pid)))
        except Exception:
            continue
        if year:
            out[pid] = year
    return out


def normalize_hitting(
    df: pd.DataFrame, season: int, *, birth_years: dict[str, int] | None = None
) -> pd.DataFrame:
    """Raw NPB batting table -> Rosetta hitter schema."""
    birth_years = birth_years or {}
    rows = []
    for _, r in df.iterrows():
        get = lambda *ns, _r=r: _row_get(df, _r, *ns)  # noqa: E731
        pa = get("PA")
        if pa <= 0:
            continue
        ab = get("AB")
        h = get("H")
        doubles = get("2B")
        triples = get("3B")
        hr = get("HR")
        bb = get("BB")
        ibb = get("IBB")
        hbp = get("HP")
        so = get("SO")
        singles = h - doubles - triples - hr
        bb_pct = (bb + ibb) / pa
        k_pct = so / pa
        iso = ((doubles + 2 * triples + 3 * hr) / ab) if ab else 0.0
        bip = ab - so - hr
        babip = ((h - hr) / bip) if bip else 0.0
        woba = (
            WOBA_W["ubb"] * (bb - ibb)
            + WOBA_W["hbp"] * hbp
            + WOBA_W["1b"] * singles
            + WOBA_W["2b"] * doubles
            + WOBA_W["3b"] * triples
            + WOBA_W["hr"] * hr
        ) / pa
        npb_id = str(r.get("_npb_id", "") or "")
        by = birth_years.get(npb_id)
        name = str(r.get("player_name", ""))
        team_ab = str(r.get("team", ""))
        rows.append(
            {
                "player_id": f"npb-{npb_id}" if npb_id else f"npb-name-{name}",
                "player_name": name,
                "season": season,
                "league": "NPB",
                "team": TEAM_NAMES.get(team_ab, team_ab),
                "role": "batter",
                "age": float(season - by) if by else 27.0,
                "pa": pa,
                "g": get("G"),
                "bb_pct": bb_pct,
                "k_pct": k_pct,
                "iso": iso,
                "babip": babip,
                "hr_pct": hr / pa,
                "avg": get("AVG"),
                "obp": get("OBP"),
                "slg": get("SLG"),
                "woba": woba,
                "home_pa": pa * 0.5,
                "road_pa": pa * 0.5,
                "home_woba": woba,
                "road_woba": woba,
                "park_id": team_ab,
                "source": "npb-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=HITTER_COLUMNS)


def normalize_pitching(
    df: pd.DataFrame, season: int, *, birth_years: dict[str, int] | None = None
) -> pd.DataFrame:
    """Raw NPB pitching table -> Rosetta pitcher schema."""
    birth_years = birth_years or {}
    rows = []
    for _, r in df.iterrows():
        get = lambda *ns, _r=r: _row_get(df, _r, *ns)  # noqa: E731
        bf = get("BF")
        ip = _parse_ip(str(r.get("IP", 0)))
        if bf <= 0 or ip <= 0:
            continue
        so = get("SO")
        bb = get("BB")
        ibb = get("IBB")
        hbp = get("HBP")
        hr = get("HR")
        k_pct = so / bf
        bb_pct = (bb + ibb) / bf
        hits_allowed = get("H")
        # No batted-ball split on leaderboard tables; HR per hit allowed is a
        # crude HR/FB proxy (same approximation as the KBO module).
        hr_fb = (hr / hits_allowed) if hits_allowed else 0.12
        era = get("ERA")
        fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
        npb_id = str(r.get("_npb_id", "") or "")
        by = birth_years.get(npb_id)
        name = str(r.get("player_name", ""))
        team_ab = str(r.get("team", ""))
        rows.append(
            {
                "player_id": f"npb-{npb_id}" if npb_id else f"npb-name-{name}",
                "player_name": name,
                "season": season,
                "league": "NPB",
                "team": TEAM_NAMES.get(team_ab, team_ab),
                "role": "pitcher",
                "age": float(season - by) if by else 27.0,
                "ip": ip,
                "g": get("G"),
                "gs": 0.0,  # not published on leaderboard tables
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "hr_fb": min(max(hr_fb, 0.0), 1.0),
                "era": era,
                "fip": fip,
                "home_pa": bf * 0.5,
                "road_pa": bf * 0.5,
                "home_era": era,
                "road_era": era,
                "park_id": team_ab,
                "source": "npb-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=PITCHER_COLUMNS)


def fetch_npb_season(
    season: int, *, session: NPBSession | None = None, with_ages: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch + normalize one NPB season (hitting, pitching)."""
    sess = session or NPBSession()
    id_map = fetch_id_map(session=sess)
    hit_raw = fetch_hitting(season, session=sess, id_map=id_map)
    pit_raw = fetch_pitching(season, session=sess, id_map=id_map)
    by_b: dict[str, int] = {}
    by_p: dict[str, int] = {}
    if with_ages:
        if not hit_raw.empty:
            by_b = fetch_birth_years(hit_raw["_npb_id"].tolist(), session=sess)
        if not pit_raw.empty:
            by_p = fetch_birth_years(pit_raw["_npb_id"].tolist(), session=sess)
    return (
        normalize_hitting(hit_raw, season, birth_years=by_b),
        normalize_pitching(pit_raw, season, birth_years=by_p),
    )


def write_npb_snapshots(
    seasons: list[int],
    *,
    out_dir: Path | None = None,
    append: bool = True,
    with_ages: bool = True,
    delay: float = 1.0,
) -> Path:
    """Fetch NPB seasons and merge into npb_batters/npb_pitchers snapshots."""
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    sess = NPBSession(delay=delay)
    bats: list[pd.DataFrame] = []
    pits: list[pd.DataFrame] = []
    for season in seasons:
        b, p = fetch_npb_season(season, session=sess, with_ages=with_ages)
        bats.append(b)
        pits.append(p)
    bat_df = pd.concat(bats, ignore_index=True) if bats else pd.DataFrame(columns=HITTER_COLUMNS)
    pit_df = pd.concat(pits, ignore_index=True) if pits else pd.DataFrame(columns=PITCHER_COLUMNS)
    for fname, frame in (("npb_batters.csv", bat_df), ("npb_pitchers.csv", pit_df)):
        path = root / fname
        if append and path.exists() and not frame.empty:
            old = pd.read_csv(path)
            # Merge semantics: synthetic fixture rows are always preserved;
            # real rows are replaced only for seasons being written now.
            written_keys = {(row["league"], row["season"]) for row in frame.to_dict("records")}
            keys = written_keys
            is_real = old["is_synthetic"].astype(str) != "True"
            rewritten = old.apply(lambda r, wk=keys: (r["league"], r["season"]) in wk, axis=1)
            old = old[~(is_real & rewritten)]
            frame = pd.concat([old, frame], ignore_index=True).drop_duplicates(
                ["player_id", "season", "league"], keep="last"
            )
        if not frame.empty:
            frame.to_csv(path, index=False)
    return root
