"""KBO league ingest via koreabaseball.com record pages (no auth, HTML).

Covers the foreign-league depth gap where the Stats API returns nothing:
season leaderboards back to 1982, Basic1 (counting) + Basic2 (BB/SO/OBP/SLG)
merged per player, with birth years from detail pages for age curves.

Verified live 2026-09-18. Explicit fetch step only; never CI-live.
Be polite: ~1s between requests, snapshot everything.

Structure (ASP.NET):
  Record/Player/HitterBasic/Basic1.aspx   rank/name/team/AVG/G/PA/AB/R/H/2B/3B/HR/TB/RBI/SAC/SF
  Record/Player/HitterBasic/Basic2.aspx   rank/name/team/AVG/BB/IBB/HBP/SO/GDP/SLG/OBP/OPS/...
  Record/Player/PitcherBasic/Basic1.aspx  rank/name/team/ERA/G/W/L/SV/HLD/WPCT/IP/H/HR/BB/HBP/SO/R/ER/WHIP
  Record/Player/PitcherBasic/Basic2.aspx  .../TBF/NP/.../IBB/...
  Record/Player/{Hitter,Pitcher}Detail/Basic.aspx?playerId=N  profile (birthdate) + playerId keys

Caveats:
  - Leaderboards are top-30 (hitting) / top ~18 (pitching) qualified players,
    not a full census. Depth source for stars + movers, not population rates.
  - No GS column for pitchers → gs=0, role detection stays "unknown".
  - No season-level home/road splits → neutral placeholders (league-average).
"""

from __future__ import annotations

import http.cookiejar
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

HANGUL_RE = re.compile(r"[\uac00-\ud7af]")


def romanize_name(name: str) -> str:
    """Romanize Hangul names/teams to RR (pass through anything else)."""
    if not HANGUL_RE.search(name):
        return name
    try:
        from korean_romanizer.romanizer import Romanizer

        return str(Romanizer(name).romanize())
    except ImportError:
        return name


BASE = "https://www.koreabaseball.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}

HIT_BASIC1 = f"{BASE}/Record/Player/HitterBasic/Basic1.aspx"
HIT_BASIC2 = f"{BASE}/Record/Player/HitterBasic/Basic2.aspx"
PIT_BASIC1 = f"{BASE}/Record/Player/PitcherBasic/Basic1.aspx"
PIT_BASIC2 = f"{BASE}/Record/Player/PitcherBasic/Basic2.aspx"
HIT_DETAIL = f"{BASE}/Record/Player/HitterDetail/Basic.aspx"
PIT_DETAIL = f"{BASE}/Record/Player/PitcherDetail/Basic.aspx"

WOBA_W = {"ubb": 0.69, "hbp": 0.72, "1b": 0.89, "2b": 1.27, "3b": 1.62, "hr": 2.10}
FIP_CONSTANT = 3.10


def _clean(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).strip()


def parse_table(html: str) -> tuple[list[str], list[dict]]:
    """Parse a KBO record table → (headers, rows with values + player_id)."""
    table = re.search(r'<table class="tData01 tt".*?</table>', html, re.S)
    if not table:
        return [], []
    body = table.group(0)
    headers = [_clean(t) for t in re.findall(r"<th[^>]*>(.*?)</th>", body, re.S)]
    rows: list[dict] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        if "<td" not in tr:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        pid = re.search(r"(?:Hitter|Pitcher)Detail/Basic\.aspx\?playerId=(\d+)", tr)
        rows.append(
            {
                "values": [_clean(c) for c in cells],
                "player_id": pid.group(1) if pid else "",
            }
        )
    return headers, rows


def parse_birth_year(html: str) -> int | None:
    """Extract birth year from a player detail page (lblBirthday)."""
    m = re.search(r"lblBirthday[^>]*>(\d{4})", html)
    return int(m.group(1)) if m else None


class KBOSession:
    """Cookie-preserving session for ASP.NET viewstate postbacks."""

    def __init__(self, *, delay: float = 1.0) -> None:
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self.delay = delay

    def get(self, url: str) -> str:
        time.sleep(self.delay)
        req = urllib.request.Request(url, headers=UA)
        with self.opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")

    def get_season(self, url: str, season: int) -> str:
        """GET the page, then post the season dropdown (seasons back to 1982)."""
        html = self.get(url)

        def field(name: str) -> str:
            for pat in (r'name="%s"[^>]*value="([^"]*)"', r'id="%s"[^>]*value="([^"]*)"'):
                m = re.search(pat % re.escape(name), html)
                if m:
                    return m.group(1)
            return ""

        form: dict[str, str] = {}
        for n in re.findall(r'<input type="hidden" name="([^"]+)"', html):
            form[n] = field(n)
        season_field = ""
        for n in re.findall(r'<select name="([^"]+)"', html):
            m = re.search(f'<select name="{re.escape(n)}".*?</select>', html, re.S)
            sel = (
                re.findall(r'<option selected="selected" value="([^"]*)"', m.group(0)) if m else []
            )
            form[n] = sel[0] if sel else ""
            if "ddlSeason" in n:
                season_field = n
        if not season_field:
            raise RuntimeError("season dropdown not found — page layout changed")
        form[season_field] = str(season)
        event_target = next(
            (n for n in re.findall(r'<select name="([^"]+)"', html) if "ddlSeason" in n), ""
        )
        form["__EVENTTARGET"] = event_target
        form["__EVENTARGUMENT"] = ""
        time.sleep(self.delay)
        data = urllib.parse.urlencode(form).encode()
        req = urllib.request.Request(
            url, data=data, headers={**UA, "Content-Type": "application/x-www-form-urlencoded"}
        )
        with self.opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")


def _to_float(x: str) -> float:
    try:
        return float(x.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _row_get(df: pd.DataFrame, row: pd.Series, *names: str) -> float:
    for n in names:
        if n in df.columns and pd.notna(row[n]):
            return _to_float(str(row[n]))
    return 0.0


def _frame(headers: list[str], rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = dict(zip(headers, r["values"], strict=False))
        rec["_kbo_id"] = r["player_id"]
        recs.append(rec)
    return pd.DataFrame(recs)


def _merge_tables(d1: pd.DataFrame, d2: pd.DataFrame, keep2: list[str]) -> pd.DataFrame:
    """Merge Basic1+Basic2 on stable keys (id, else name — never empty strings)."""
    if d1.empty:
        return d1
    d2 = d2[[c for c in keep2 if c in d2.columns]]
    for d in (d1, d2):
        name = d.get("선수명", pd.Series([""] * len(d))).fillna("").astype(str)
        kid = d.get("_kbo_id", pd.Series([""] * len(d))).fillna("").astype(str)
        d["_mkey"] = ("kbo:" + kid).where(kid != "", "name:" + name)
    merged = d1.merge(d2, on="_mkey", how="left", suffixes=("", "_b2"))
    return merged.drop(columns=["_mkey"])


def fetch_hitting(season: int, *, session: KBOSession | None = None) -> pd.DataFrame:
    """Merged Basic1+Basic2 hitting table for a season (raw columns)."""
    sess = session or KBOSession()
    h1, r1 = parse_table(sess.get_season(HIT_BASIC1, season))
    h2, r2 = parse_table(sess.get_season(HIT_BASIC2, season))
    d1, d2 = _frame(h1, r1), _frame(h2, r2)
    return _merge_tables(
        d1, d2, ["선수명", "_kbo_id", "BB", "IBB", "HBP", "SO", "SLG", "OBP", "OPS"]
    )


def fetch_pitching(season: int, *, session: KBOSession | None = None) -> pd.DataFrame:
    """Merged Basic1+Basic2 pitching table for a season (raw columns)."""
    sess = session or KBOSession()
    h1, r1 = parse_table(sess.get_season(PIT_BASIC1, season))
    h2, r2 = parse_table(sess.get_season(PIT_BASIC2, season))
    d1, d2 = _frame(h1, r1), _frame(h2, r2)
    return _merge_tables(d1, d2, ["선수명", "_kbo_id", "TBF", "NP", "IBB"])


def fetch_birth_years(
    player_ids: list[str], role: str, *, session: KBOSession | None = None
) -> dict[str, int]:
    """Birth years keyed by KBO playerId (detail pages; polite delays)."""
    sess = session or KBOSession()
    base = HIT_DETAIL if role == "batter" else PIT_DETAIL
    out: dict[str, int] = {}
    for pid in dict.fromkeys(p for p in player_ids if p):
        try:
            year = parse_birth_year(sess.get(f"{base}?playerId={pid}"))
        except Exception:
            continue
        if year:
            out[pid] = year
    return out


def normalize_hitting(
    df: pd.DataFrame, season: int, *, birth_years: dict[str, int] | None = None
) -> pd.DataFrame:
    """Raw merged KBO hitting table → Rosetta hitter schema."""
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
        hbp = get("HBP")
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
        kbo_id = str(r.get("_kbo_id", "") or "")
        by = birth_years.get(kbo_id)
        name = romanize_name(str(r.get("선수명", "")))
        rows.append(
            {
                "player_id": f"kbo-{kbo_id}" if kbo_id else f"kbo-name-{name}",
                "player_name": name,
                "season": season,
                "league": "KBO",
                "team": romanize_name(str(r.get("팀명", ""))),
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
                "park_id": str(r.get("팀명", "")),
                "source": "kbo-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=HITTER_COLUMNS)


def _parse_ip_kbo(x: str) -> float:
    """KBO IP renders as decimal ('155') or with fractions; be lenient."""
    try:
        return float(str(x).replace(",", "").split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0


def normalize_pitching(
    df: pd.DataFrame, season: int, *, birth_years: dict[str, int] | None = None
) -> pd.DataFrame:
    """Raw merged KBO pitching table → Rosetta pitcher schema."""
    birth_years = birth_years or {}
    rows = []
    for _, r in df.iterrows():
        get = lambda *ns, _r=r: _row_get(df, _r, *ns)  # noqa: E731
        tbf = get("TBF")
        ip = _parse_ip_kbo(str(r.get("IP", 0)))
        if tbf <= 0 or ip <= 0:
            continue
        so = get("SO")
        bb = get("BB")
        ibb = get("IBB")
        hbp = get("HBP")
        hr = get("HR")
        k_pct = so / tbf
        bb_pct = (bb + ibb) / tbf
        # No batted-ball split on leaderboard tables; HR per hit allowed is a
        # crude HR/FB proxy. Documented approximation — treat with wide bands.
        hits_allowed = get("H")
        hr_fb = (hr / hits_allowed) if hits_allowed else 0.12
        era = get("ERA")
        fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
        kbo_id = str(r.get("_kbo_id", "") or "")
        by = birth_years.get(kbo_id)
        name = romanize_name(str(r.get("선수명", "")))
        rows.append(
            {
                "player_id": f"kbo-{kbo_id}" if kbo_id else f"kbo-name-{name}",
                "player_name": name,
                "season": season,
                "league": "KBO",
                "team": romanize_name(str(r.get("팀명", ""))),
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
                "home_pa": tbf * 0.5,
                "road_pa": tbf * 0.5,
                "home_era": era,
                "road_era": era,
                "park_id": str(r.get("팀명", "")),
                "source": "kbo-official",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows, columns=PITCHER_COLUMNS)


def fetch_kbo_season(
    season: int, *, session: KBOSession | None = None, with_ages: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch + normalize one KBO season (hitting, pitching)."""
    sess = session or KBOSession()
    hit_raw = fetch_hitting(season, session=sess)
    pit_raw = fetch_pitching(season, session=sess)
    by_b: dict[str, int] = {}
    by_p: dict[str, int] = {}
    if with_ages and not hit_raw.empty:
        by_b = fetch_birth_years(hit_raw["_kbo_id"].tolist(), "batter", session=sess)
    if with_ages and not pit_raw.empty:
        by_p = fetch_birth_years(pit_raw["_kbo_id"].tolist(), "pitcher", session=sess)
    return (
        normalize_hitting(hit_raw, season, birth_years=by_b),
        normalize_pitching(pit_raw, season, birth_years=by_p),
    )


def write_kbo_snapshots(
    seasons: list[int],
    *,
    out_dir: Path | None = None,
    append: bool = True,
    with_ages: bool = True,
    delay: float = 1.0,
) -> Path:
    """Fetch KBO seasons and merge into kbo_batters/kbo_pitchers snapshots."""
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    sess = KBOSession(delay=delay)
    bats: list[pd.DataFrame] = []
    pits: list[pd.DataFrame] = []
    for season in seasons:
        b, p = fetch_kbo_season(season, session=sess, with_ages=with_ages)
        bats.append(b)
        pits.append(p)
    bat_df = pd.concat(bats, ignore_index=True) if bats else pd.DataFrame(columns=HITTER_COLUMNS)
    pit_df = pd.concat(pits, ignore_index=True) if pits else pd.DataFrame(columns=PITCHER_COLUMNS)
    for fname, frame in (("kbo_batters.csv", bat_df), ("kbo_pitchers.csv", pit_df)):
        path = root / fname
        if append and path.exists() and not frame.empty:
            old = pd.read_csv(path)
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
