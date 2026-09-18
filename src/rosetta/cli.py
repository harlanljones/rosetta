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

app = typer.Typer(help="Rosetta MLE — translate foreign/minor league stats to MLB scale.", no_args_is_help=True)


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
) -> None:
    """Estimate league-link factors from the transferred-player cohort."""
    ensure_data_dirs()
    cohort = build_cohort(snapshots)
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
    )
    rprint(tr.model_dump(mode="json"))


@app.command("backtest")
def backtest_cmd(
    snapshots: Path = typer.Option(SNAPSHOT_DIR),
    factors: Path = typer.Option(OUTPUT_DIR / "factors.json"),
    out: Path = typer.Option(OUTPUT_DIR / "backtest.json"),
    boot: int = 300,
) -> None:
    """Hold out recent transfers and score translations."""
    ensure_data_dirs()
    if not factors.exists():
        fit_factors(snapshots=snapshots, out=factors, boot=boot)
    metrics = run_backtest(snapshots, factors_path=factors, n_boot=boot, out_path=out)
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
) -> None:
    """Export current foreign-league stars with MLB-equivalent projections."""
    ensure_data_dirs()
    if not factors.exists():
        fit_factors(snapshots=snapshots, out=factors)
    from rosetta.models.factors import LeagueFactorModel

    model = LeagueFactorModel.load(factors)
    boards = {}
    for role, min_pt, pt_col in (("batter", min_pa, "pa"), ("pitcher", min_ip, "ip")):
        df = load_all_seasons(role=role, snapshot_dir=snapshots)
        df = df[(df["season"] == season) & (df["league"] != "MLB") & (df[pt_col] >= min_pt)]
        if df.empty:
            boards[role] = []
            continue
        translated = translate_frame(df, model=model)
        if role == "batter":
            translated = translated.sort_values("woba", ascending=False)
        else:
            translated = translated.sort_values("fip", ascending=True)
        boards[role] = json.loads(translated.head(50).to_json(orient="records"))
    out.write_text(json.dumps({"season": season, "boards": boards}, indent=2))
    rprint(f"[green]Leaderboard →[/green] {out}")


@app.command("cohort-summary")
def cohort_summary(snapshots: Path = typer.Option(SNAPSHOT_DIR)) -> None:
    """Show transferred-player cohort counts by link."""
    cohort = build_cohort(snapshots)
    g = cohort.groupby(["role", "from_league", "to_league"]).size().reset_index(name="n")
    rprint(g.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    app()
