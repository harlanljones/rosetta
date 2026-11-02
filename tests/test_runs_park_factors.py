"""Runs park factors from game logs (NPB/KBO): estimator, application, no-op."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rosetta.cli import app
from rosetta.cohort.builder import build_cohort, pair_seasons
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons, load_transfers
from rosetta.models.factors import fit_factor_model
from rosetta.models.park import (
    DEFAULT_RUNS_PARK_K,
    GAME_LOG_COLUMNS,
    estimate_runs_park_factors,
    load_runs_park_factors,
    lookup_runs_park_factor,
    neutralize_line_runs,
)
from rosetta.translate.api import translate, translate_line

REPO_SNAPSHOTS = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def _logs(rows: list[tuple], league: str = "NPB") -> pd.DataFrame:
    """rows: (season, home, away, home_runs, away_runs[, park_id])."""
    recs = []
    for i, r in enumerate(rows):
        season, home, away, hr, ar = r[:5]
        park = r[5] if len(r) > 5 else ""
        recs.append(
            {
                "league": league,
                "season": season,
                "date": f"{season}-04-{(i % 28) + 1:02d}",
                "home_team": home,
                "away_team": away,
                "home_runs": hr,
                "away_runs": ar,
                "park_id": park,
                "source": "test",
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(recs, columns=GAME_LOG_COLUMNS)


def _two_team_season(season: int, n: int = 100, home_total: int = 10, road_total: int = 8):
    """A hitter's park (A) and a neutral-ish park (B), n home games each."""
    rows = []
    for _ in range(n):
        rows.append((season, "A", "B", home_total // 2, home_total - home_total // 2))
        rows.append((season, "B", "A", road_total // 2, road_total - road_total // 2))
    return rows


def test_header_only_snapshot_files_match_schema() -> None:
    for league in ("npb", "kbo"):
        df = pd.read_csv(REPO_SNAPSHOTS / f"game_logs_{league}.csv")
        assert list(df.columns) == GAME_LOG_COLUMNS
        assert df.empty


def test_estimator_known_ratio_and_regression() -> None:
    logs = _logs(_two_team_season(2020))
    out = estimate_runs_park_factors(logs, window=3, k=300.0)
    a = out[(out["team"] == "A") & (out["season"] == 2020)].iloc[0]
    # A: home 10 rpg, road 8 rpg -> raw 1.25; 200 games -> weight 0.4
    assert a["games"] == 200
    assert a["park_factor"] == pytest.approx(1.0 + 0.4 * 0.25)
    b = out[(out["team"] == "B") & (out["season"] == 2020)].iloc[0]
    # B: home 8, road 10 -> raw 0.8
    assert b["park_factor"] == pytest.approx(1.0 + 0.4 * (0.8 - 1.0))
    assert list(out.columns) == ["league", "team", "season", "park_factor", "games"]


def test_estimator_pools_trailing_window_and_k_zero_is_raw() -> None:
    rows = _two_team_season(2020, n=50, home_total=10, road_total=8)
    rows += _two_team_season(2021, n=50, home_total=12, road_total=8)
    rows += _two_team_season(2022, n=50, home_total=14, road_total=8)
    rows += _two_team_season(2023, n=50, home_total=16, road_total=8)
    out = estimate_runs_park_factors(_logs(rows), window=3, k=0.0)
    a23 = out[(out["team"] == "A") & (out["season"] == 2023)].iloc[0]
    # 2021-2023 only: home mean 14, road mean 8
    assert a23["park_factor"] == pytest.approx(14 / 8)
    assert a23["games"] == 300
    a21 = out[(out["team"] == "A") & (out["season"] == 2021)].iloc[0]
    # 2019-2021 window: only 2020 and 2021 exist
    assert a21["park_factor"] == pytest.approx(11 / 8)
    # window=1 uses just the season
    one = estimate_runs_park_factors(_logs(rows), window=1, k=0.0)
    a23_1 = one[(one["team"] == "A") & (one["season"] == 2023)].iloc[0]
    assert a23_1["park_factor"] == pytest.approx(16 / 8)


def test_estimator_ignores_synthetic_rows_and_empty() -> None:
    logs = _logs(_two_team_season(2020))
    fake = logs.copy()
    fake["is_synthetic"] = True
    fake["home_runs"] = 50
    both = pd.concat([logs, fake], ignore_index=True)
    out = estimate_runs_park_factors(both)
    ref = estimate_runs_park_factors(logs)
    pd.testing.assert_frame_equal(out, ref)
    assert estimate_runs_park_factors(fake).empty
    assert estimate_runs_park_factors(logs.iloc[0:0]).empty


def test_estimator_park_id_excludes_non_home_park_games() -> None:
    rows = _two_team_season(2020, n=40)
    # A's primary park "PA" (all rows above lack park ids; give them one)
    rows = [(*r, "PA" if r[1] == "A" else "PB") for r in rows]
    # 5 extreme neutral-site games with A as nominal home team at another park
    rows += [(2020, "A", "B", 30, 30, "TOKYO")] * 5
    out = estimate_runs_park_factors(_logs(rows), k=0.0)
    a = out[out["team"] == "A"].iloc[0]
    assert a["park_factor"] == pytest.approx(10 / 8)


def test_load_runs_park_factors_empty_and_populated(tmp_path: Path) -> None:
    empty = load_runs_park_factors(REPO_SNAPSHOTS)
    assert empty.empty
    assert list(empty.columns) == ["league", "team", "season", "park_factor", "games"]
    assert load_runs_park_factors(tmp_path / "nope").empty

    pd.DataFrame(columns=GAME_LOG_COLUMNS).to_csv(tmp_path / "game_logs_npb.csv", index=False)
    _logs(_two_team_season(2020), league="KBO").to_csv(tmp_path / "game_logs_kbo.csv", index=False)
    pf = load_runs_park_factors(tmp_path)
    assert set(pf["league"]) == {"KBO"}
    assert not pf.empty


def test_neutralize_line_runs_half_effect_and_untouched_stats() -> None:
    bat = pd.Series(
        {"woba": 0.360, "iso": 0.200, "hr_pct": 0.040, "bb_pct": 0.09, "k_pct": 0.2, "babip": 0.3}
    )
    out = neutralize_line_runs(bat, 1.2, "batter")
    div = (1 + 1.2) / 2
    assert out["woba"] == pytest.approx(0.360 / div)
    assert out["iso"] == pytest.approx(0.200 / div)
    assert out["hr_pct"] == pytest.approx(0.040 / div)
    for s in ("bb_pct", "k_pct", "babip"):
        assert out[s] == bat[s]
    assert bat["woba"] == 0.360  # input not mutated

    pit = pd.Series({"era": 3.0, "fip": 3.2, "hr_fb": 0.1, "k_pct": 0.25, "bb_pct": 0.07})
    outp = neutralize_line_runs(pit, 0.8, "pitcher")
    assert outp["era"] == pytest.approx(3.0 / 0.9)
    assert outp["fip"] == pytest.approx(3.2 / 0.9)
    assert outp["hr_fb"] == pytest.approx(0.1 / 0.9)
    assert outp["k_pct"] == 0.25 and outp["bb_pct"] == 0.07


@pytest.mark.parametrize("pf", [None, float("nan"), 0.0, -1.0])
def test_neutralize_line_runs_noop_without_factor(pf) -> None:
    bat = pd.Series({"woba": 0.36, "iso": 0.2})
    assert neutralize_line_runs(bat, pf, "batter").equals(bat)


def test_lookup_only_npb_kbo_exact_team_season() -> None:
    pf = pd.DataFrame(
        [
            {"league": "NPB", "team": "A", "season": 2020, "park_factor": 1.1, "games": 100},
            {"league": "KBO", "team": "A", "season": 2020, "park_factor": 0.9, "games": 100},
        ]
    )
    assert lookup_runs_park_factor(pf, "NPB", "A", 2020) == 1.1
    assert lookup_runs_park_factor(pf, "KBO", "A", 2020) == 0.9
    assert lookup_runs_park_factor(pf, "NPB", "A", 2021) is None
    assert lookup_runs_park_factor(pf, "NPB", "Z", 2020) is None
    assert lookup_runs_park_factor(pf, "AAA", "A", 2020) is None
    assert lookup_runs_park_factor(None, "NPB", "A", 2020) is None


def test_default_k_documented_value() -> None:
    assert DEFAULT_RUNS_PARK_K == 300.0


# ---------------------------------------------------------------------------
# Integration: translate_line, cohort, regression with empty logs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory):
    root = build_snapshots(out_dir=tmp_path_factory.mktemp("snap") / "snap", seed=3)
    cohort = build_cohort(root, min_pa=40, min_ip=15)
    model = fit_factor_model(cohort[~cohort["holdout"].astype(bool)], n_boot=20, seed=1)
    return root, model


def _factor_frame(league: str, team: str, season: int, pf: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"league": league, "team": team, "season": season, "park_factor": pf, "games": 400}]
    )


def test_translate_line_applies_park_before_league_factors(synth) -> None:
    _, model = synth
    line = {
        "player_id": "x",
        "player_name": "X",
        "season": 2022,
        "league": "NPB",
        "team": "Tokyo Test",
        "role": "batter",
        "age": 27.0,
        "pa": 500,
        "woba": 0.360,
        "iso": 0.200,
        "hr_pct": 0.040,
        "bb_pct": 0.09,
        "k_pct": 0.18,
        "babip": 0.31,
    }
    base = translate_line(line, model=model)
    pf = _factor_frame("NPB", "Tokyo Test", 2022, 1.2)
    adj = translate_line(line, model=model, park_factors=pf)
    # hitter's park -> lowered rates; untouched stats identical
    assert adj.rates["woba"] < base.rates["woba"]
    assert adj.rates["woba"] == pytest.approx(base.rates["woba"] / 1.1)
    assert adj.rates["bb_pct"] == base.rates["bb_pct"]
    assert adj.rates["babip"] == base.rates["babip"]
    # different team-season or league -> unchanged
    other = translate_line(
        line, model=model, park_factors=_factor_frame("NPB", "Elsewhere", 2022, 1.2)
    )
    assert other.rates == base.rates
    kbo_pf = _factor_frame("KBO", "Tokyo Test", 2022, 1.2)
    assert translate_line(line, model=model, park_factors=kbo_pf).rates == base.rates
    # AAA source is never park-adjusted by this path
    aaa = {**line, "league": "AAA"}
    aaa_pf = _factor_frame("AAA", "Tokyo Test", 2022, 1.2)
    assert (
        translate_line(aaa, model=model, park_factors=aaa_pf).rates
        == translate_line(aaa, model=model).rates
    )


def test_pair_seasons_applies_to_npb_endpoints_only(synth) -> None:
    root, _ = synth
    transfers = load_transfers(root)
    bat = load_all_seasons(role="batter", snapshot_dir=root)
    pit = load_all_seasons(role="pitcher", snapshot_dir=root)
    base = pair_seasons(transfers, bat, pit)
    # factor for every NPB/KBO team-season
    teams = (
        pd.concat([bat, pit])
        .query("league in ['NPB','KBO']")[["league", "team", "season"]]
        .drop_duplicates()
    )
    teams["park_factor"] = 1.3
    teams["games"] = 400
    adj = pair_seasons(transfers, bat, pit, park_factors=teams)
    assert len(adj) == len(base)
    bt = base[base["role"] == "batter"]
    at = adj[adj["role"] == "batter"]
    npb_from = bt["from_league"].isin(["NPB", "KBO"])
    assert npb_from.any()
    div = (1 + 1.3) / 2
    assert (
        (at.loc[npb_from, "pre_woba"] / bt.loc[npb_from, "pre_woba"])
        .round(9)
        .eq(round(1 / div, 9))
        .all()
    )
    assert at.loc[npb_from, "pre_bb_pct"].equals(bt.loc[npb_from, "pre_bb_pct"])
    # MLB/AAA post endpoints untouched
    non_intl_post = ~bt["to_league"].isin(["NPB", "KBO"])
    assert at.loc[non_intl_post, "post_woba"].equals(bt.loc[non_intl_post, "post_woba"])
    non_intl_pre = ~bt["from_league"].isin(["NPB", "KBO"])
    assert at.loc[non_intl_pre, "pre_woba"].equals(bt.loc[non_intl_pre, "pre_woba"])


def test_outputs_unchanged_when_game_logs_empty(synth, tmp_path: Path) -> None:
    root, model = synth
    # Snapshot copy WITH header-only game logs vs one WITHOUT the files at all.
    with_logs = tmp_path / "with_logs"
    without = tmp_path / "without"
    shutil.copytree(root, with_logs)
    shutil.copytree(root, without)
    for league in ("npb", "kbo"):
        shutil.copy(
            REPO_SNAPSHOTS / f"game_logs_{league}.csv", with_logs / f"game_logs_{league}.csv"
        )
        assert not (without / f"game_logs_{league}.csv").exists()

    c1 = build_cohort(with_logs, min_pa=40, min_ip=15)
    c2 = build_cohort(without, min_pa=40, min_ip=15)
    pd.testing.assert_frame_equal(c1, c2)

    model.save(tmp_path / "factors.json")
    bat = load_all_seasons(role="batter", snapshot_dir=root)
    pit = load_all_seasons(role="pitcher", snapshot_dir=root)
    for frame, role in ((bat, "batter"), (pit, "pitcher")):
        for league in ("NPB", "KBO"):
            row = frame[(frame["league"] == league)].iloc[0]
            for snap in (with_logs, without):
                tr = translate(
                    row["player_id"],
                    league,
                    season=int(row["season"]),
                    role=role,
                    factors_path=tmp_path / "factors.json",
                    snapshot_dir=snap,
                )
                direct = translate_line(row, model=model)
                assert tr.rates == direct.rates
                assert tr.lower == direct.lower
                assert tr.upper == direct.upper


def test_translate_loads_factors_from_snapshot_dir(synth, tmp_path: Path) -> None:
    root, model = synth
    snap = tmp_path / "snap"
    shutil.copytree(root, snap)
    bat = load_all_seasons(role="batter", snapshot_dir=snap)
    row = bat[bat["league"] == "NPB"].iloc[0]
    # one game per side is enough for the loader; k tiny so effect is visible
    g = _logs(
        [(int(row["season"]), row["team"], "ZZZ", 8, 6)] * 30
        + [(int(row["season"]), "ZZZ", row["team"], 3, 3)] * 30
    )
    g.to_csv(snap / "game_logs_npb.csv", index=False)
    model.save(tmp_path / "f.json")
    kw = {
        "season": int(row["season"]),
        "role": "batter",
        "factors_path": tmp_path / "f.json",
    }
    plain = translate(row["player_id"], "NPB", snapshot_dir=root, **kw)
    adj = translate(row["player_id"], "NPB", snapshot_dir=snap, **kw)
    assert adj.rates["woba"] < plain.rates["woba"]  # park (14/6) is hitter-friendly
    assert adj.rates["bb_pct"] == plain.rates["bb_pct"]


def test_cli_derive_runs_park_factors(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    _logs(_two_team_season(2020)).to_csv(snap / "game_logs_npb.csv", index=False)
    out = tmp_path / "out" / "pf.csv"
    res = CliRunner().invoke(
        app, ["derive-runs-park-factors", "--snapshots", str(snap), "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(out)
    assert list(df.columns) == ["league", "team", "season", "park_factor", "games"]
    assert set(df["team"]) == {"A", "B"}

    # empty logs -> empty table, still exit 0
    out2 = tmp_path / "pf2.csv"
    res = CliRunner().invoke(
        app, ["derive-runs-park-factors", "--snapshots", str(REPO_SNAPSHOTS), "--out", str(out2)]
    )
    assert res.exit_code == 0, res.output
    assert pd.read_csv(out2).empty
