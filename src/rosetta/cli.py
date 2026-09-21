"""Rosetta CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from rosetta.backtest.evaluate import run_backtest
from rosetta.cohort.builder import build_cohort
from rosetta.ingest.generate_snapshots import build_snapshots
from rosetta.ingest.loaders import load_all_seasons
from rosetta.models.factors import fit_factor_model
from rosetta.paths import OUTPUT_DIR, SNAPSHOT_DIR, ensure_data_dirs
from rosetta.translate.api import translate, translate_frame

app = typer.Typer(
    help="Rosetta MLE — translate foreign/minor league stats to MLB scale.", no_args_is_help=True
)


@app.command("historical-backtest")
def historical_backtest_cmd(
    snapshots: Path = typer.Option(SNAPSHOT_DIR, help="Real normalized snapshot directory"),
    target_season: int = typer.Option(2025, help="Completed MLB target season"),
    boot: int = typer.Option(250, help="Bootstrap resamples per fitted factor"),
    seed: int = typer.Option(42, help="Deterministic bootstrap seed"),
    min_pa: float = typer.Option(80.0, help="Minimum source and target batter PA"),
    min_ip: float = typer.Option(30.0, help="Minimum source and target pitcher IP"),
    estimator: str = typer.Option("ratio_of_means", help="Link estimator: ratio_of_means (default) or mean_ratio"),
    no_era_floor: bool = typer.Option(
        False, "--no-era-floor", help="Disable the AAA 2019 ball-standardization era floor (comparison runs only)"
    ),
    out_dir: Path = typer.Option(OUTPUT_DIR / "historical-backtest", help="Artifact directory"),
) -> None:
    """Evaluate adjacent real AAA-to-MLB seasons and write a readable report."""
    from rosetta.backtest.historical import run_historical_backtest
    from rosetta.backtest.report import write_historical_report
    from rosetta.models.factors import LEAGUE_ERA_FLOOR

    try:
        payload = run_historical_backtest(
            snapshots,
            target_season=target_season,
            n_boot=boot,
            seed=seed,
            min_pa=min_pa,
            min_ip=min_ip,
            estimator=estimator,
            era_floor=None if no_era_floor else LEAGUE_ERA_FLOOR,
        )
        paths = write_historical_report(payload, out_dir)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        rprint(f"[red]Historical backtest unavailable:[/red] {exc}")
        raise typer.Exit(1) from exc

    metadata = payload.get("metadata", {})
    counts = metadata.get("counts", {})
    training_count = counts.get("training_pairs")
    if training_count is None:
        training_count = len(payload.get("training_pairs", []))
    rprint(
        f"[green]Historical backtest:[/green] target {metadata.get('target_season', target_season)}; "
        f"training pairs {training_count}; "
        f"scored detail rows {len(payload.get('detail', []))}"
    )
    headline = [row for row in payload.get("metrics", []) if str(row.get("stat", "")).lower() in {"woba", "fip"}]
    if headline:
        table = Table(title="Historical headline metrics")
        for column in ("role", "stat", "n", "mae", "baseline_mae", "coverage_80"):
            table.add_column(column)
        for row in headline:
            table.add_row(
                str(row.get("role", "")), str(row.get("stat", "")), str(row.get("n", "")),
                f"{float(row['mae']):.4f}", f"{float(row['baseline_mae']):.4f}",
                f"{float(row['coverage_80']) * 100:.1f}%",
            )
        rprint(table)
    for kind, path in paths.items():
        rprint(f"  {kind}: {path}")


@app.command("historical-rolling")
def historical_rolling_cmd(
    snapshots: Path = typer.Option(SNAPSHOT_DIR, help="Real normalized snapshot directory"),
    target_seasons: str = typer.Option("2022,2023,2024,2025", help="Comma-separated target MLB seasons"),
    boot: int = typer.Option(250, help="Bootstrap resamples per fitted factor"),
    seed: int = typer.Option(42, help="Deterministic bootstrap seed"),
    min_pa: float = typer.Option(80.0, help="Minimum source and target batter PA"),
    min_ip: float = typer.Option(30.0, help="Minimum source and target pitcher IP"),
    estimator: str = typer.Option("ratio_of_means", help="Link estimator: ratio_of_means (default) or mean_ratio"),
    no_era_floor: bool = typer.Option(
        False, "--no-era-floor", help="Disable the AAA 2019 ball-standardization era floor (comparison runs only)"
    ),
    out_dir: Path = typer.Option(OUTPUT_DIR / "historical-rolling", help="Artifact directory"),
) -> None:
    """Fit and score each target season independently (no 2025-only tuning)."""
    from rosetta.backtest.historical import run_rolling_backtest
    from rosetta.backtest.report import write_rolling_report
    from rosetta.models.factors import LEAGUE_ERA_FLOOR

    seasons = [int(part.strip()) for part in target_seasons.split(",") if part.strip()]
    try:
        payload = run_rolling_backtest(
            snapshots,
            target_seasons=seasons,
            n_boot=boot,
            seed=seed,
            min_pa=min_pa,
            min_ip=min_ip,
            estimator=estimator,
            era_floor=None if no_era_floor else LEAGUE_ERA_FLOOR,
        )
        paths = write_rolling_report(payload, out_dir)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        rprint(f"[red]Historical rolling backtest unavailable:[/red] {exc}")
        raise typer.Exit(1) from exc

    metadata = payload.get("metadata", {})
    skipped = metadata.get("skipped", [])
    rprint(
        f"[green]Historical rolling backtest:[/green] targets {metadata.get('target_seasons', seasons)}; "
        f"estimator {metadata.get('parameters', {}).get('estimator', estimator)}"
    )
    if skipped:
        for item in skipped:
            rprint(f"  [yellow]skipped {item.get('target_season')}:[/yellow] {item.get('reason')}")

    pooled = [row for row in payload.get("pooled", []) if str(row.get("stat", "")).lower() in {"woba", "fip"}]
    if pooled:
        table = Table(title="Pooled headline metrics")
        for column in ("role", "stat", "n", "mae", "baseline_mae", "coverage_80"):
            table.add_column(column)
        for row in pooled:
            table.add_row(
                str(row.get("role", "")), str(row.get("stat", "")), str(row.get("n", "")),
                f"{float(row['mae']):.4f}", f"{float(row['baseline_mae']):.4f}",
                f"{float(row['coverage_80']) * 100:.1f}%",
            )
        rprint(table)

    per_target = [row for row in payload.get("summary", []) if str(row.get("stat", "")).lower() in {"woba", "fip"}]
    if per_target:
        table = Table(title="Per-target MAE vs baseline (wOBA / FIP)")
        for column in ("target_season", "role", "stat", "n", "mae", "baseline_mae"):
            table.add_column(column)
        for row in sorted(per_target, key=lambda r: (r.get("target_season", 0), str(r.get("role", "")), str(r.get("stat", "")))):
            table.add_row(
                str(row.get("target_season", "")), str(row.get("role", "")), str(row.get("stat", "")),
                str(row.get("n", "")), f"{float(row['mae']):.4f}", f"{float(row['baseline_mae']):.4f}",
            )
        rprint(table)

    for kind, path in paths.items():
        rprint(f"  {kind}: {path}")


@app.command("generate-snapshots")
def generate_snapshots_cmd(seed: int = 42) -> None:
    """Write synthetic offline snapshots for CI/demo."""
    root = build_snapshots(seed=seed)
    rprint(f"[green]Snapshots written to[/green] {root}")


@app.command("fit-factors")
def fit_factors(
    snapshots: Path = typer.Option(SNAPSHOT_DIR, help="Snapshot directory"),
    out: Path = typer.Option(OUTPUT_DIR / "factors.json", help="Output factors JSON"),
    boot: int = typer.Option(250, help="Bootstrap resamples per link"),
    chadwick: Path | None = typer.Option(None, help="Chadwick register for cross-ID linking"),
) -> None:
    """Estimate league-link factors from the transferred-player cohort."""
    ensure_data_dirs()
    cohort = build_cohort(snapshots, chadwick_path=chadwick)
    # Fit on non-holdout only
    train = cohort[~cohort["holdout"].astype(bool)] if "holdout" in cohort.columns else cohort
    model = fit_factor_model(train, n_boot=boot)
    model.save(out)
    rprint(f"[green]Wrote factors[/green] → {out}")
    for role, links in model.links.items():
        rprint(f"  {role}: {', '.join(links)}")


@app.command("translate")
def translate_cmd(
    player: str = typer.Argument(..., help="player_id or exact name"),
    from_league: str = typer.Argument(..., help="Source league, e.g. NPB"),
    season: int | None = typer.Option(None, help="Season year"),
    role: str = typer.Option("batter", help="batter|pitcher"),
    factors: Path = typer.Option(OUTPUT_DIR / "factors.json"),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    target_role: str | None = typer.Option(
        None, help="Pitcher target role for SP<->RP adjustment: SP|RP"
    ),
) -> None:
    """Translate one player-season to MLB-equivalent rates."""
    if not factors.exists():
        rprint("[yellow]Factors missing — fitting now…[/yellow]")
        fit_factors(snapshots=snapshots, out=factors)
    tr = translate(
        player,
        from_league,
        season=season,
        role=role,
        factors_path=factors,
        snapshot_dir=snapshots,
        target_pitch_role=target_role,
    )
    rprint(tr.model_dump(mode="json"))


@app.command("backtest")
def backtest_cmd(
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    factors: Path = typer.Option(OUTPUT_DIR / "factors.json"),
    out: Path = typer.Option(OUTPUT_DIR / "backtest.json"),
    boot: int = 300,
    chadwick: Path | None = typer.Option(None, help="Chadwick register for cross-ID linking"),
) -> None:
    """Hold out recent transfers and score translations."""
    ensure_data_dirs()
    if not factors.exists():
        fit_factors(snapshots=snapshots, out=factors, boot=boot, chadwick=chadwick)
    metrics = run_backtest(
        snapshots, factors_path=factors, n_boot=boot, out_path=out, chadwick_path=chadwick
    )
    table = Table(title="Backtest by stat")
    table.add_column("stat")
    table.add_column("n", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("80% cov", justify="right")
    for stat, m in sorted(metrics.get("by_stat", {}).items()):
        table.add_row(
            stat,
            str(m["n"]),
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['coverage_80']:.1%}",
        )
    rprint(table)
    rprint(f"Wrote {out}")


@app.command("leaderboard")
def leaderboard_cmd(
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    factors: Path = typer.Option(OUTPUT_DIR / "factors.json"),
    out: Path = typer.Option(OUTPUT_DIR / "leaderboard.json"),
    season: int = typer.Option(2025, help="Season to rank"),
    min_pa: float = 200,
    min_ip: float = 60,
    include_synthetic: bool = typer.Option(
        False,
        help="Include synthetic fixture rows (off by default; live rows are preferred).",
    ),
) -> None:
    """Export current foreign-league stars with MLB-equivalent projections.

    Snapshots can contain synthetic offline fixtures alongside live ingests. The
    public leaderboard is real-data-only unless ``--include-synthetic`` is
    explicitly requested.
    """
    ensure_data_dirs()
    if not factors.exists():
        fit_factors(snapshots=snapshots, out=factors)
    from rosetta.models.factors import LeagueFactorModel

    model = LeagueFactorModel.load(factors)
    boards = {}
    source_counts = {}
    for role, min_pt, pt_col in (("batter", min_pa, "pa"), ("pitcher", min_ip, "ip")):
        df = load_all_seasons(role=role, snapshot_dir=snapshots)
        if not include_synthetic and "is_synthetic" in df.columns:
            df = df[~df["is_synthetic"].astype(str).str.lower().eq("true")]
        df = df[(df["season"] == season) & (df["league"] != "MLB") & (df[pt_col] >= min_pt)]
        source_counts[role] = (
            df["source"].fillna("unknown").value_counts().to_dict()
            if "source" in df.columns
            else {}
        )
        if df.empty:
            boards[role] = []
            continue
        translated = translate_frame(df, model=model)
        for column in ("source", "is_synthetic", "team"):
            if column in df.columns:
                translated[column] = df[column].to_numpy()
        if role == "batter":
            translated = translated.sort_values("woba", ascending=False)
        else:
            translated = translated.sort_values("fip", ascending=True)
        boards[role] = json.loads(translated.head(50).to_json(orient="records"))
    out.write_text(
        json.dumps(
            {
                "season": season,
                "data_mode": "all" if include_synthetic else "real",
                "source_counts": source_counts,
                "boards": boards,
            },
            indent=2,
        )
    )
    rprint(f"[green]Leaderboard →[/green] {out}")


@app.command("fetch-chadwick")
def fetch_chadwick(
    out: Path = typer.Option(SNAPSHOT_DIR / "chadwick_people.csv"),
) -> None:
    """Download the Chadwick register (sharded) for cross-league ID linking."""
    from rosetta.ingest.chadwick import download_register

    path = download_register(out)
    rprint(f"[green]Chadwick register →[/green] {path}")


@app.command("fetch-statsapi")
def fetch_statsapi(
    seasons: str = typer.Option("2024", help="Comma-separated seasons, e.g. 2022,2023,2024"),
    sport: int = typer.Option(11, help="Stats API sportId (11=AAA, 12=AA, 17=winter)"),
    league_label: str | None = typer.Option(
        None, help="Override league label (default from sportId)"
    ),
    league_ids: str | None = typer.Option(None, help="Comma-separated leagueIds filter"),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    with_ages: bool = typer.Option(True, help="Backfill ages from the sport player pool"),
) -> None:
    """Fetch season lines via the MLB Stats API into league snapshots."""
    from rosetta.ingest.statsapi import write_statsapi_snapshots

    years = [int(s) for s in seasons.split(",") if s.strip()]
    root = write_statsapi_snapshots(
        years,
        sport,
        league_label=league_label,
        league_ids=league_ids,
        out_dir=snapshots,
        with_ages=with_ages,
    )
    rprint(f"[green]Stats API snapshots →[/green] {root}")


@app.command("fetch-kbo")
def fetch_kbo(
    seasons: str = typer.Option("2025", help="Comma-separated seasons, e.g. 2023,2024,2025"),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    with_ages: bool = typer.Option(True, help="Fetch detail pages for birth years"),
    delay: float = typer.Option(1.0, help="Seconds between requests"),
) -> None:
    """Fetch KBO season lines via koreabaseball.com into snapshots."""
    from rosetta.ingest.kbo import write_kbo_snapshots

    years = [int(s) for s in seasons.split(",") if s.strip()]
    root = write_kbo_snapshots(years, out_dir=snapshots, with_ages=with_ages, delay=delay)
    rprint(f"[green]KBO snapshots →[/green] {root}")


@app.command("fetch-npb")
def fetch_npb(
    seasons: str = typer.Option("2025", help="Comma-separated seasons, e.g. 2024,2025"),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    with_ages: bool = typer.Option(True, help="Fetch player cards for birth years"),
    delay: float = typer.Option(1.0, help="Seconds between requests"),
) -> None:
    """Fetch NPB season lines via npb.jp BIS into snapshots."""
    from rosetta.ingest.npb import write_npb_snapshots

    years = [int(s) for s in seasons.split(",") if s.strip()]
    root = write_npb_snapshots(years, out_dir=snapshots, with_ages=with_ages, delay=delay)
    rprint(f"[green]NPB snapshots →[/green] {root}")


@app.command("fetch-savant")
def fetch_savant(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD"),
    season: int = typer.Option(2024, help="Season year for snapshot labeling"),
    league: str = typer.Option("AAA", help="League label for snapshot files"),
    minors: bool = typer.Option(True, help="Fetch minors (vs MLB) pitch data"),
    delay: float = typer.Option(2.0, help="Seconds between day requests"),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    raw_dir: Path | None = typer.Option(None, help="Raw CSV cache dir (default data/raw/savant)"),
) -> None:
    """Download Savant pitch data and aggregate into league snapshots."""
    from rosetta.ingest.savant import write_savant_snapshots

    root = write_savant_snapshots(
        start_date,
        end_date,
        season,
        league,
        minors=minors,
        raw_dir=raw_dir,
        out_dir=snapshots,
        delay=delay,
    )
    rprint(f"[green]Savant snapshots →[/green] {root}")


@app.command("derive-park-factors")
def derive_park_factors_cmd(
    raw_dir: Path = typer.Option(default=..., help="Dir with savant CSV files"),
    min_pa: int = typer.Option(200, help="Minimum PA per park"),
    out: Path = typer.Option(OUTPUT_DIR / "park_factors.csv"),
) -> None:
    """Derive regressed park factors from Savant pitch CSVs."""
    from rosetta.ingest.savant import derive_park_factors, load_pitches

    raw = Path(raw_dir)
    csvs = sorted(raw.glob("*.csv"))
    if not csvs:
        rprint("[red]No CSVs found in raw_dir[/red]")
        raise typer.Exit(1)
    rprint(f"Loading {len(csvs)} CSV files…")
    pitches = load_pitches(csvs)
    if pitches.empty:
        rprint("[red]No pitch data loaded[/red]")
        raise typer.Exit(1)
    rprint(f"Loaded {len(pitches):,} pitch events")
    pf = derive_park_factors(pitches, min_pa=min_pa)
    out.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out, index=False)
    rprint(f"[green]Park factors ({len(pf)} parks) →[/green] {out}")


@app.command("generate-transfers")
def generate_transfers_cmd(
    register: Path = typer.Option(
        SNAPSHOT_DIR / "chadwick_people.csv", help="Chadwick register CSV"
    ),
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    out: Path = typer.Option(SNAPSHOT_DIR / "transfers.csv"),
    max_gap: int = typer.Option(1, help="Max season gap for a transfer pair"),
) -> None:
    """Generate real transferred-player pairs from Chadwick + season data."""
    from rosetta.ingest.transfers import write_transfer_snapshot

    path = write_transfer_snapshot(
        register_path=register,
        snapshot_dir=snapshots,
        max_season_gap=max_gap,
        out_path=out,
    )
    rprint(f"[green]Transfer pairs →[/green] {path}")


@app.command("cohort-summary")
def cohort_summary(snapshots: Path = typer.Option(SNAPSHOT_DIR)) -> None:
    """Show transferred-player cohort counts by link."""
    cohort = build_cohort(snapshots)
    g = cohort.groupby(["role", "from_league", "to_league"]).size().reset_index(name="n")
    rprint(g.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    app()
