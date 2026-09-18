"""Generate reproducible offline snapshot CSVs for development and CI.

These are *labeled synthetic* cohorts shaped like real league environments and
transfer patterns. Live scrapers can refresh real data later; `make backtest`
must work without network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rosetta.paths import SNAPSHOT_DIR, ensure_data_dirs
from rosetta.schema import HITTER_COLUMNS, PITCHER_COLUMNS

# League offensive environments (illustrative anchors, not published factors)
LEAGUE_ENV = {
    "MLB": {"bb": 0.085, "k": 0.220, "iso": 0.165, "babip": 0.300, "hr": 0.035, "woba": 0.320},
    "AAA": {"bb": 0.090, "k": 0.230, "iso": 0.180, "babip": 0.310, "hr": 0.040, "woba": 0.335},
    "NPB": {"bb": 0.080, "k": 0.195, "iso": 0.140, "babip": 0.295, "hr": 0.028, "woba": 0.315},
    "KBO": {"bb": 0.095, "k": 0.185, "iso": 0.155, "babip": 0.305, "hr": 0.032, "woba": 0.330},
    "CPBL": {"bb": 0.088, "k": 0.175, "iso": 0.145, "babip": 0.308, "hr": 0.030, "woba": 0.325},
    "CUBA": {"bb": 0.070, "k": 0.160, "iso": 0.130, "babip": 0.295, "hr": 0.025, "woba": 0.305},
}

PITCH_ENV = {
    "MLB": {"k": 0.220, "bb": 0.080, "hr_fb": 0.120, "era": 4.20, "fip": 4.15},
    "AAA": {"k": 0.230, "bb": 0.085, "hr_fb": 0.125, "era": 4.50, "fip": 4.40},
    "NPB": {"k": 0.200, "bb": 0.075, "hr_fb": 0.100, "era": 3.60, "fip": 3.70},
    "KBO": {"k": 0.190, "bb": 0.078, "hr_fb": 0.110, "era": 4.00, "fip": 4.05},
    "CPBL": {"k": 0.180, "bb": 0.082, "hr_fb": 0.115, "era": 4.10, "fip": 4.15},
    "CUBA": {"k": 0.170, "bb": 0.070, "hr_fb": 0.095, "era": 3.80, "fip": 3.90},
}

# True difficulty multipliers applied when a player moves toward MLB (talent view)
# values < 1 on offense rates that should fall (iso/woba), >1 on K% for hitters, etc.
TRUE_LINK = {
    ("AAA", "MLB"): {
        "bb_pct": 0.95,
        "k_pct": 1.08,
        "iso": 0.82,
        "babip": 0.97,
        "hr_pct": 0.80,
        "woba": 0.88,
        "p_k_pct": 0.92,
        "p_bb_pct": 1.05,
        "hr_fb": 1.05,
        "era": 1.12,
        "fip": 1.10,
    },
    ("NPB", "AAA"): {
        "bb_pct": 0.97,
        "k_pct": 1.05,
        "iso": 0.90,
        "babip": 0.98,
        "hr_pct": 0.88,
        "woba": 0.93,
        "p_k_pct": 0.95,
        "p_bb_pct": 1.03,
        "hr_fb": 1.04,
        "era": 1.08,
        "fip": 1.07,
    },
    ("KBO", "NPB"): {
        "bb_pct": 0.96,
        "k_pct": 1.04,
        "iso": 0.92,
        "babip": 0.98,
        "hr_pct": 0.90,
        "woba": 0.94,
        "p_k_pct": 0.96,
        "p_bb_pct": 1.02,
        "hr_fb": 1.03,
        "era": 1.06,
        "fip": 1.05,
    },
    ("CPBL", "KBO"): {
        "bb_pct": 0.97,
        "k_pct": 1.03,
        "iso": 0.93,
        "babip": 0.99,
        "hr_pct": 0.92,
        "woba": 0.95,
        "p_k_pct": 0.97,
        "p_bb_pct": 1.02,
        "hr_fb": 1.02,
        "era": 1.05,
        "fip": 1.04,
    },
}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _woba_from_parts(bb: float, iso: float, babip: float, hr: float) -> float:
    # Lightweight recombination for synthetic lines (not linear weights)
    return _clip(0.12 + 0.7 * bb + 0.35 * babip + 0.55 * iso + 0.9 * hr, 0.200, 0.500)


def _fip_from_parts(k: float, bb: float, hr_fb: float) -> float:
    # Toy FIP-like scale for synthetic pitchers
    return _clip(3.2 + 12.0 * bb - 8.0 * k + 10.0 * hr_fb, 1.5, 7.5)


def _player_id(prefix: str, i: int) -> str:
    return f"{prefix}-{i:04d}"


def generate_league_pool(rng: np.random.Generator, seasons: range) -> dict[str, list[dict]]:
    """Create stable player talent draws reused across league stints."""
    hitters: list[dict] = []
    pitchers: list[dict] = []

    # Core MLB/AAA population
    for i in range(1, 221):
        talent = {
            "player_id": _player_id("H", i),
            "player_name": f"Hitter {i}",
            "bb": rng.normal(0.085, 0.025),
            "k": rng.normal(0.22, 0.05),
            "iso": rng.normal(0.165, 0.05),
            "babip": rng.normal(0.300, 0.025),
            "hr": rng.normal(0.035, 0.015),
            "base_age": rng.uniform(22, 34),
            "tier": "mlb_system",
        }
        hitters.append(talent)

    for i in range(1, 181):
        talent = {
            "player_id": _player_id("P", i),
            "player_name": f"Pitcher {i}",
            "k": rng.normal(0.22, 0.05),
            "bb": rng.normal(0.08, 0.02),
            "hr_fb": rng.normal(0.12, 0.03),
            "base_age": rng.uniform(22, 34),
            "tier": "mlb_system",
        }
        pitchers.append(talent)

    # Foreign stars (smaller pools)
    for i, league in enumerate(["NPB", "KBO", "CPBL", "CUBA"], start=1):
        for j in range(1, 41):
            idx = 300 + i * 50 + j
            hitters.append(
                {
                    "player_id": _player_id("H", idx),
                    "player_name": f"{league} Hitter {j}",
                    "bb": rng.normal(0.09, 0.02),
                    "k": rng.normal(0.18, 0.04),
                    "iso": rng.normal(0.170, 0.04),
                    "babip": rng.normal(0.305, 0.02),
                    "hr": rng.normal(0.034, 0.012),
                    "base_age": rng.uniform(24, 32),
                    "tier": league,
                }
            )
        for j in range(1, 31):
            idx = 300 + i * 50 + j
            pitchers.append(
                {
                    "player_id": _player_id("P", idx),
                    "player_name": f"{league} Pitcher {j}",
                    "k": rng.normal(0.21, 0.04),
                    "bb": rng.normal(0.075, 0.02),
                    "hr_fb": rng.normal(0.11, 0.03),
                    "base_age": rng.uniform(24, 32),
                    "tier": league,
                }
            )

    return {"hitters": hitters, "pitchers": pitchers, "seasons": list(seasons)}


def _apply_env_hitter(talent: dict, league: str, age: float, rng: np.random.Generator) -> dict:
    env = LEAGUE_ENV[league]
    # Age curve: peak ~27
    age_delta = -0.003 * (age - 27) ** 2

    def noise(s: float) -> float:
        return float(rng.normal(0, s))

    bb = _clip(talent["bb"] * (env["bb"] / 0.085) + noise(0.01) + age_delta, 0.02, 0.25)
    k = _clip(talent["k"] * (env["k"] / 0.22) + noise(0.015) - 0.5 * age_delta, 0.05, 0.45)
    iso = _clip(talent["iso"] * (env["iso"] / 0.165) + noise(0.02) + age_delta, 0.02, 0.40)
    babip = _clip(talent["babip"] * (env["babip"] / 0.300) + noise(0.015), 0.22, 0.40)
    hr = _clip(talent["hr"] * (env["hr"] / 0.035) + noise(0.008) + 0.5 * age_delta, 0.005, 0.12)
    woba = _woba_from_parts(bb, iso, babip, hr)
    pa = float(rng.integers(180, 650))
    home_pa = pa * rng.uniform(0.45, 0.55)
    road_pa = pa - home_pa
    park_bump = rng.normal(0, 0.015)
    return {
        "bb_pct": bb,
        "k_pct": k,
        "iso": iso,
        "babip": babip,
        "hr_pct": hr,
        "avg": _clip(babip * (1 - k - bb) + 0.15 * hr, 0.15, 0.40),
        "obp": _clip(bb + (1 - k - bb) * babip * 0.9, 0.25, 0.50),
        "slg": _clip(0.300 + iso + 0.1 * babip, 0.25, 0.70),
        "woba": woba,
        "pa": pa,
        "g": pa / rng.uniform(3.5, 4.2),
        "home_pa": home_pa,
        "road_pa": road_pa,
        "home_woba": _clip(woba + park_bump, 0.2, 0.5),
        "road_woba": _clip(woba - park_bump, 0.2, 0.5),
    }


def _apply_env_pitcher(talent: dict, league: str, age: float, rng: np.random.Generator) -> dict:
    env = PITCH_ENV[league]
    age_delta = 0.004 * (age - 27) ** 2  # pitchers worsen away from peak

    def noise(s: float) -> float:
        return float(rng.normal(0, s))

    k = _clip(talent["k"] * (env["k"] / 0.22) + noise(0.015) - 0.3 * age_delta, 0.08, 0.45)
    bb = _clip(talent["bb"] * (env["bb"] / 0.08) + noise(0.01) + 0.2 * age_delta, 0.03, 0.20)
    hr_fb = _clip(talent["hr_fb"] * (env["hr_fb"] / 0.12) + noise(0.01) + 0.2 * age_delta, 0.04, 0.25)
    fip = _fip_from_parts(k, bb, hr_fb)
    era = _clip(fip + rng.normal(0.1, 0.35), 1.5, 8.0)
    ip = float(rng.integers(40, 190))
    g = float(rng.integers(15, 35))
    gs = float(rng.integers(0, int(min(g, 32))))
    park_bump = rng.normal(0, 0.25)
    return {
        "k_pct": k,
        "bb_pct": bb,
        "hr_fb": hr_fb,
        "era": era,
        "fip": fip,
        "ip": ip,
        "g": g,
        "gs": gs,
        "home_pa": ip * 2.2,
        "road_pa": ip * 2.1,
        "home_era": _clip(era + park_bump, 1.0, 9.0),
        "road_era": _clip(era - park_bump, 1.0, 9.0),
    }


def _chain_factors(from_league: str, to_league: str) -> dict[str, float]:
    order = ["CPBL", "KBO", "NPB", "AAA", "MLB"]
    if from_league not in order or to_league not in order:
        raise ValueError(f"unsupported path {from_league}->{to_league}")
    i = order.index(from_league)
    j = order.index(to_league)
    if i >= j:
        raise ValueError("only upward chain supported in synthetic generator")
    acc: dict[str, float] = {}
    for a, b in zip(order[i:j], order[i + 1 : j + 1], strict=False):
        link = TRUE_LINK[(a, b)]
        if not acc:
            acc = dict(link)
        else:
            acc = {k: acc[k] * link[k] for k in acc}
    return acc


def _post_move_hitter(pre: dict, factors: dict, rng: np.random.Generator) -> dict:
    out = dict(pre)
    for key in ("bb_pct", "k_pct", "iso", "babip", "hr_pct", "woba"):
        out[key] = _clip(pre[key] * factors[key] * rng.normal(1.0, 0.04), 0.01, 0.55)
    out["woba"] = _woba_from_parts(out["bb_pct"], out["iso"], out["babip"], out["hr_pct"])
    out["avg"] = _clip(out["babip"] * (1 - out["k_pct"] - out["bb_pct"]) + 0.15 * out["hr_pct"], 0.15, 0.4)
    out["obp"] = _clip(out["bb_pct"] + (1 - out["k_pct"] - out["bb_pct"]) * out["babip"] * 0.9, 0.25, 0.5)
    out["slg"] = _clip(0.30 + out["iso"] + 0.1 * out["babip"], 0.25, 0.7)
    return out


def _post_move_pitcher(pre: dict, factors: dict, rng: np.random.Generator) -> dict:
    out = dict(pre)
    out["k_pct"] = _clip(pre["k_pct"] * factors["p_k_pct"] * rng.normal(1.0, 0.04), 0.05, 0.5)
    out["bb_pct"] = _clip(pre["bb_pct"] * factors["p_bb_pct"] * rng.normal(1.0, 0.04), 0.02, 0.25)
    out["hr_fb"] = _clip(pre["hr_fb"] * factors["hr_fb"] * rng.normal(1.0, 0.05), 0.03, 0.3)
    out["fip"] = _fip_from_parts(out["k_pct"], out["bb_pct"], out["hr_fb"])
    out["era"] = _clip(out["fip"] + rng.normal(0.15, 0.4), 1.5, 9.0)
    return out


def build_snapshots(out_dir: Path | None = None, seed: int = 42) -> Path:
    ensure_data_dirs()
    root = Path(out_dir) if out_dir else SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seasons = range(2015, 2026)
    pool = generate_league_pool(rng, seasons)

    hitter_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    transfer_rows: list[dict] = []

    # Assign primary leagues
    assignments = {
        "MLB": (pool["hitters"][:120], pool["pitchers"][:100]),
        "AAA": (pool["hitters"][80:200], pool["pitchers"][60:160]),
        "NPB": (
            [h for h in pool["hitters"] if h["tier"] == "NPB"] + pool["hitters"][200:220],
            [p for p in pool["pitchers"] if p["tier"] == "NPB"] + pool["pitchers"][160:175],
        ),
        "KBO": (
            [h for h in pool["hitters"] if h["tier"] == "KBO"] + pool["hitters"][210:230],
            [p for p in pool["pitchers"] if p["tier"] == "KBO"] + pool["pitchers"][165:180],
        ),
        "CPBL": (
            [h for h in pool["hitters"] if h["tier"] == "CPBL"],
            [p for p in pool["pitchers"] if p["tier"] == "CPBL"],
        ),
        "CUBA": (
            [h for h in pool["hitters"] if h["tier"] == "CUBA"],
            [p for p in pool["pitchers"] if p["tier"] == "CUBA"],
        ),
    }

    # Regular seasons per league
    for league, (hs, ps) in assignments.items():
        for season in seasons:
            for h in hs:
                age = h["base_age"] + (season - 2018) * 0.15
                if age < 20 or age > 40:
                    continue
                if rng.random() < 0.15:
                    continue
                stats = _apply_env_hitter(h, league, age, rng)
                hitter_rows.append(
                    {
                        "player_id": h["player_id"],
                        "player_name": h["player_name"],
                        "season": season,
                        "league": league,
                        "team": f"{league}-T{rng.integers(1, 8)}",
                        "role": "batter",
                        "age": round(age, 1),
                        **stats,
                        "park_id": f"{league}-P{rng.integers(1, 6)}",
                        "source": "synthetic",
                        "is_synthetic": True,
                    }
                )
            for p in ps:
                age = p["base_age"] + (season - 2018) * 0.15
                if age < 20 or age > 40:
                    continue
                if rng.random() < 0.18:
                    continue
                stats = _apply_env_pitcher(p, league, age, rng)
                pitcher_rows.append(
                    {
                        "player_id": p["player_id"],
                        "player_name": p["player_name"],
                        "season": season,
                        "league": league,
                        "team": f"{league}-T{rng.integers(1, 8)}",
                        "role": "pitcher",
                        "age": round(age, 1),
                        **stats,
                        "park_id": f"{league}-P{rng.integers(1, 6)}",
                        "source": "synthetic",
                        "is_synthetic": True,
                    }
                )

    # Build transfer pairs with known ground-truth multipliers
    links = [("AAA", "MLB"), ("NPB", "AAA"), ("NPB", "MLB"), ("KBO", "NPB"), ("KBO", "MLB"), ("CPBL", "KBO")]
    for from_lg, to_lg in links:
        factors = _chain_factors(from_lg, to_lg)
        n_hit = {("AAA", "MLB"): 90, ("NPB", "MLB"): 35, ("NPB", "AAA"): 40, ("KBO", "MLB"): 25, ("KBO", "NPB"): 30, ("CPBL", "KBO"): 20}[
            (from_lg, to_lg)
        ]
        n_pit = max(12, n_hit // 2)

        cand_h = [h for h in pool["hitters"] if h["tier"] in {from_lg, "mlb_system", to_lg}]
        for h in rng.choice(cand_h, size=min(n_hit, len(cand_h)), replace=False):
            season_from = int(rng.integers(2015, 2024))
            season_to = season_from + 1
            age_from = h["base_age"] + (season_from - 2018) * 0.15
            age_to = age_from + 1.0
            pre = _apply_env_hitter(h, from_lg, age_from, rng)
            post = _post_move_hitter(pre, factors, rng)
            # Ensure both seasons exist in tables
            for season, age, lg, stats in (
                (season_from, age_from, from_lg, pre),
                (season_to, age_to, to_lg, post),
            ):
                hitter_rows.append(
                    {
                        "player_id": h["player_id"],
                        "player_name": h["player_name"],
                        "season": season,
                        "league": lg,
                        "team": f"{lg}-TX",
                        "role": "batter",
                        "age": round(age, 1),
                        **stats,
                        "park_id": f"{lg}-P1",
                        "source": "synthetic_transfer",
                        "is_synthetic": True,
                    }
                )
            transfer_rows.append(
                {
                    "player_id": h["player_id"],
                    "player_name": h["player_name"],
                    "role": "batter",
                    "from_league": from_lg,
                    "to_league": to_lg,
                    "from_season": season_from,
                    "to_season": season_to,
                    "age_from": round(age_from, 1),
                    "age_to": round(age_to, 1),
                    "holdout": season_to >= 2023,
                }
            )

        cand_p = [p for p in pool["pitchers"] if p["tier"] in {from_lg, "mlb_system", to_lg}]
        for p in rng.choice(cand_p, size=min(n_pit, len(cand_p)), replace=False):
            season_from = int(rng.integers(2015, 2024))
            season_to = season_from + 1
            age_from = p["base_age"] + (season_from - 2018) * 0.15
            age_to = age_from + 1.0
            pre = _apply_env_pitcher(p, from_lg, age_from, rng)
            post = _post_move_pitcher(pre, factors, rng)
            for season, age, lg, stats in (
                (season_from, age_from, from_lg, pre),
                (season_to, age_to, to_lg, post),
            ):
                pitcher_rows.append(
                    {
                        "player_id": p["player_id"],
                        "player_name": p["player_name"],
                        "season": season,
                        "league": lg,
                        "team": f"{lg}-TX",
                        "role": "pitcher",
                        "age": round(age, 1),
                        **stats,
                        "park_id": f"{lg}-P1",
                        "source": "synthetic_transfer",
                        "is_synthetic": True,
                    }
                )
            transfer_rows.append(
                {
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "role": "pitcher",
                    "from_league": from_lg,
                    "to_league": to_lg,
                    "from_season": season_from,
                    "to_season": season_to,
                    "age_from": round(age_from, 1),
                    "age_to": round(age_to, 1),
                    "holdout": season_to >= 2023,
                }
            )

    hit_df = pd.DataFrame(hitter_rows)
    pit_df = pd.DataFrame(pitcher_rows)
    # Dedup player-season-league
    hit_df = hit_df.drop_duplicates(["player_id", "season", "league"], keep="last")
    pit_df = pit_df.drop_duplicates(["player_id", "season", "league"], keep="last")

    for league in LEAGUE_ENV:
        hdf = hit_df[hit_df["league"] == league].reindex(columns=HITTER_COLUMNS)
        pdf = pit_df[pit_df["league"] == league].reindex(columns=PITCHER_COLUMNS)
        hdf.to_csv(root / f"{league.lower()}_batters.csv", index=False)
        pdf.to_csv(root / f"{league.lower()}_pitchers.csv", index=False)

    pd.DataFrame(transfer_rows).to_csv(root / "transfers.csv", index=False)

    meta = {
        "seed": seed,
        "note": "Synthetic snapshots for offline CI. Replace via scrapers when available.",
        "n_hitter_seasons": int(len(hit_df)),
        "n_pitcher_seasons": int(len(pit_df)),
        "n_transfers": int(len(transfer_rows)),
    }
    pd.Series(meta).to_json(root / "manifest.json")
    return root


def main() -> None:
    root = build_snapshots()
    print(f"Wrote snapshots to {root}")


if __name__ == "__main__":
    main()
