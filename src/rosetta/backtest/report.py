"""Human- and machine-readable artifacts for the historical backtest."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

_DETAIL_COLUMNS = (
    "player_id", "player_name", "role", "from_season", "to_season",
    "from_league", "to_league", "pre_sample", "post_sample", "pre_source",
    "post_source", "stat", "prior", "prediction", "actual", "error",
    "absolute_error", "lower", "upper", "covered_80",
)


def _number(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _coverage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value) if value is not None else ""


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _display(key: str, value: Any) -> str:
    if value is None or value == "":
        return ""
    if key == "coverage_80" or key == "covered_80":
        if isinstance(value, bool):
            return "yes" if value else "no"
        return _coverage(value)
    if key in {"mae", "rmse", "baseline_mae", "baseline_rmse", "prior", "prediction", "actual", "error", "absolute_error", "lower", "upper"}:
        return _number(value)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    keys = list(columns or ())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)


def _metric_table(metrics: list[dict[str, Any]]) -> str:
    lines = ["| Role | Stat | n | MAE | RMSE | Baseline MAE | Baseline RMSE | 80% coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(
            "| {role} | {stat} | {n} | {mae} | {rmse} | {bmae} | {brmse} | {coverage} |".format(
                role=row.get("role", ""), stat=row.get("stat", ""), n=row.get("n", ""),
                mae=_number(row.get("mae")), rmse=_number(row.get("rmse")),
                bmae=_number(row.get("baseline_mae")), brmse=_number(row.get("baseline_rmse")),
                coverage=_coverage(row.get("coverage_80")),
            )
        )
    return "\n".join(lines)


def _markdown(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    metrics = list(payload.get("metrics", []))
    detail = list(payload.get("detail", []))
    training = list(payload.get("training_pairs", []))
    lines = [
        "# Historical backtest",
        "",
        f"Target season: **{metadata.get('target_season', '')}**  ",
        f"Source season: **{metadata.get('source_season', '')}**  ",
        f"Data mode: **{metadata.get('data_mode', 'real')}**",
        "",
        "This report compares a translated prior-season rate with the observed MLB rate. "
        "MAE is the average absolute miss; RMSE penalizes large misses more heavily. "
        "Baseline columns use the prior source rate without translation.",
        "",
        "## Metrics",
        "",
        _metric_table(metrics),
        "",
        "## Named examples",
        "",
    ]
    # Stable, objective examples: first sorted batter and pitcher detail rows.
    ordered = sorted(
        detail,
        key=lambda r: (
            str(r.get("role", "")),
            str(r.get("player_name", "")).casefold(),
            str(r.get("player_id", "")),
            str(r.get("stat", "")),
        ),
    )
    for role in ("batter", "pitcher"):
        rows = [r for r in ordered if r.get("role") == role]
        preferred = "woba" if role == "batter" else "fip"
        preferred_rows = [r for r in rows if str(r.get("stat", "")).lower() == preferred]
        rows = preferred_rows or rows
        if rows:
            for row in rows[:5]:
                lines.append(f"- **{row.get('player_name', row.get('player_id', 'Unknown'))}** ({role}, {row.get('stat', '')}): predicted {_number(row.get('prediction'))}, actual {_number(row.get('actual'))}; 80% interval [{_number(row.get('lower'))}, {_number(row.get('upper'))}].")
    if not any(r.get("role") in ("batter", "pitcher") for r in ordered):
        lines.append("No scored players were available.")
    counts = metadata.get("counts", {})
    lines += ["", "## Selection and cutoff", "",
              f"- Training pairs: **{counts.get('training_pairs', len(training))}**; "
              f"evaluation pairs: **{counts.get('evaluation_pairs', '')}**.",
              f"- Training destination seasons are strictly before target; latest training season: **{metadata.get('latest_training_to_season', '')}**.",
              f"- Parameters: `{json.dumps(metadata.get('parameters', {}), sort_keys=True)}`.",
              "", "## All-player detail", "",
              f"Rows: **{len(detail)}**. Complete detail is available in [historical-backtest.csv](historical-backtest.csv).", ""]
    lines += ["", "## Training and provenance", "", f"Training pairs: **{len(training)}**", ""]
    for key, value in metadata.items():
        if key in {"limitations", "input_sha256", "source_counts", "checks", "provenance"}:
            lines.append(f"- **{key}:** `{json.dumps(value, sort_keys=True)}`")
    limitations = metadata.get("limitations", [])
    if limitations:
        lines += ["", "## Limitations", ""]
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def _html(payload: dict[str, Any]) -> str:
    # Render the already complete Markdown-adjacent content without a template dependency.
    metadata = payload.get("metadata", {})
    metrics = payload.get("metrics", [])
    detail = sorted(
        payload.get("detail", []),
        key=lambda r: (
            str(r.get("role", "")),
            str(r.get("player_name", "")).casefold(),
            str(r.get("player_id", "")),
            str(r.get("stat", "")),
        ),
    )
    esc = _escape
    metric_keys = ("role", "stat", "n", "mae", "rmse", "baseline_mae", "baseline_rmse", "coverage_80")
    metric_rows = "".join(
        "<tr>" + "".join(f"<td>{esc(_display(key, row.get(key, '')))}</td>" for key in metric_keys) + "</tr>"
        for row in metrics
    )
    detail_rows = "".join(
        "<tr>" + "".join(f"<td>{esc(_display(key, row.get(key, '')))}</td>" for key in _DETAIL_COLUMNS) + "</tr>"
        for row in detail
    )
    limitations = metadata.get("limitations", [])
    limitation_html = "".join(f"<li>{esc(item)}</li>" for item in limitations)
    counts = metadata.get("counts", {})
    summary_values = (
        ("trained through", metadata.get("latest_training_to_season", "")),
        ("training pairs", counts.get("training_pairs", "")),
        ("scored players", counts.get("evaluation_pairs", "")),
        ("evaluation", f"{metadata.get('source_season', '')} → {metadata.get('target_season', '')}"),
    )
    summary_html = "".join(f"<div class=\"card\"><b>{esc(label)}</b><br>{esc(value)}</div>" for label, value in summary_values)
    compact_keys = ("player_name", "role", "stat", "prior", "prediction", "actual", "error")
    example_rows = []
    for role in ("batter", "pitcher"):
        preferred = "woba" if role == "batter" else "fip"
        role_rows = [row for row in detail if row.get("role") == role and str(row.get("stat", "")).lower() == preferred]
        if not role_rows:
            role_rows = [row for row in detail if row.get("role") == role]
        example_rows.extend(
            sorted(
                role_rows,
                key=lambda row: (str(row.get("player_name", "")).casefold(), str(row.get("player_id", ""))),
            )[:5]
        )
    compact_rows = "".join(
        "<tr>" + "".join(f"<td>{esc(_display(key, row.get(key, '')))}</td>" for key in compact_keys) + "</tr>"
        for row in example_rows
    )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rosetta historical backtest</title>
<style>body{font:15px system-ui,sans-serif;line-height:1.45;color:#202124;margin:2rem;max-width:1400px}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;white-space:nowrap}th{background:#f3f5f7}section{overflow-x:auto}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.6rem}.card{background:#f7f9fb;padding:.7rem;border-radius:.4rem}</style></head><body>
<h1>Historical backtest</h1><div class="meta"><div class="card"><b>Target season</b><br>""" + esc(metadata.get("target_season", "")) + """</div><div class="card"><b>Source season</b><br>""" + esc(metadata.get("source_season", "")) + """</div><div class="card"><b>Data mode</b><br>""" + esc(metadata.get("data_mode", "real")) + """</div>""" + summary_html + """</div>
<p>MAE is the average absolute miss. RMSE penalizes large misses more heavily. Baseline metrics use the prior source rate without translation.</p><h2>Named examples</h2><section><table><thead><tr>""" + "".join(f"<th>{esc(k.replace('_', ' '))}</th>" for k in compact_keys) + """</tr></thead><tbody>""" + compact_rows + """</tbody></table></section><h2>Metrics</h2><section><table><thead><tr>""" + "".join(f"<th>{esc(k.replace('_', ' '))}</th>" for k in metric_keys) + """</tr></thead><tbody>""" + metric_rows + """</tbody></table></section><h2>Limitations</h2><ul>""" + limitation_html + """</ul><details><summary>Provenance, checksums, and complete audit detail</summary><p><a href="historical-backtest.json">Download JSON</a> · <a href="historical-backtest.csv">Download detail CSV</a> · <a href="historical-backtest-training.csv">Download training CSV</a></p><dl>""" + "".join(f"<dt><b>{esc(key)}</b></dt><dd>{esc(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)}</dd>" for key, value in metadata.items() if key in {"training_cutoff", "input_sha256", "source_counts", "counts", "latest_training_to_season"}) + """</dl><section><table><thead><tr>""" + "".join(f"<th>{esc(k.replace('_', ' '))}</th>" for k in _DETAIL_COLUMNS) + """</tr></thead><tbody>""" + detail_rows + """</tbody></table></section></details></body></html>"""


_ROLLING_DETAIL_COLUMNS = ("target_season", *_DETAIL_COLUMNS)
_ROLLING_SUMMARY_COLUMNS = (
    "target_season", "role", "stat", "n", "mae", "rmse",
    "baseline_mae", "baseline_rmse", "coverage_80",
)


def _rolling_pooled_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Role | Stat | n | MAE | RMSE | Baseline MAE | Baseline RMSE | 80% coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {role} | {stat} | {n} | {mae} | {rmse} | {bmae} | {brmse} | {coverage} |".format(
                role=row.get("role", ""), stat=row.get("stat", ""), n=row.get("n", ""),
                mae=_number(row.get("mae")), rmse=_number(row.get("rmse")),
                bmae=_number(row.get("baseline_mae")), brmse=_number(row.get("baseline_rmse")),
                coverage=_coverage(row.get("coverage_80")),
            )
        )
    return "\n".join(lines)


def _rolling_summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Target | Role | Stat | n | MAE | RMSE | Baseline MAE | Baseline RMSE | 80% coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r.get("target_season", 0), str(r.get("role", "")), str(r.get("stat", "")))):
        lines.append(
            "| {target} | {role} | {stat} | {n} | {mae} | {rmse} | {bmae} | {brmse} | {coverage} |".format(
                target=row.get("target_season", ""), role=row.get("role", ""), stat=row.get("stat", ""),
                n=row.get("n", ""), mae=_number(row.get("mae")), rmse=_number(row.get("rmse")),
                bmae=_number(row.get("baseline_mae")), brmse=_number(row.get("baseline_rmse")),
                coverage=_coverage(row.get("coverage_80")),
            )
        )
    return "\n".join(lines)


def _rolling_markdown(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    pooled = list(payload.get("pooled", []))
    summary = list(payload.get("summary", []))
    skipped = list(metadata.get("skipped", []))
    lines = [
        "# Historical rolling backtest",
        "",
        f"Target seasons: **{metadata.get('target_seasons', '')}**  ",
        f"Data mode: **{metadata.get('data_mode', 'real')}**  ",
        f"Parameters: `{json.dumps(metadata.get('parameters', {}), sort_keys=True)}`",
        "",
        "Each target season is fit independently on real AAA-to-MLB pairs with a "
        "destination season strictly before that target, then scored against the "
        "immediately following season. MAE is the average absolute miss between the "
        "translated prediction and the observed MLB rate; RMSE penalizes large misses "
        "more heavily. Baseline columns use the untranslated prior source rate.",
        "",
        "## Skipped targets",
        "",
    ]
    if skipped:
        for item in skipped:
            lines.append(f"- **{item.get('target_season', '')}**: {item.get('reason', '')}")
    else:
        lines.append("None — every requested target season had sufficient real training and evaluation pairs.")
    lines += [
        "",
        "## Pooled metrics (all scored targets combined)",
        "",
        _rolling_pooled_table(pooled),
        "",
        "## Per-target metrics",
        "",
        _rolling_summary_table(summary),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_rolling_report(payload: dict[str, Any], out_dir: Path | str) -> dict[str, Path]:
    """Write the multi-season rolling historical report and return artifact paths."""
    if not isinstance(payload, dict):
        raise TypeError("historical rolling payload must be a dictionary")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = [dict(row) for row in payload.get("summary", [])]
    detail_rows: list[dict[str, Any]] = []
    for season_key, target_payload in payload.get("by_target", {}).items():
        for row in target_payload.get("detail", []):
            merged = {"target_season": row.get("to_season", season_key)}
            merged.update(row)
            detail_rows.append(merged)
    paths = {
        "json": out / "historical-rolling.json",
        "summary_csv": out / "historical-rolling-summary.csv",
        "detail_csv": out / "historical-rolling-detail.csv",
        "markdown": out / "historical-rolling.md",
    }
    paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(paths["summary_csv"], summary_rows, _ROLLING_SUMMARY_COLUMNS)
    _write_csv(paths["detail_csv"], detail_rows, _ROLLING_DETAIL_COLUMNS)
    paths["markdown"].write_text(_rolling_markdown(payload), encoding="utf-8")
    return paths


def write_historical_report(payload: dict[str, Any], out_dir: Path | str) -> dict[str, Path]:
    """Write the complete historical report and return artifact paths."""
    if not isinstance(payload, dict):
        raise TypeError("historical backtest payload must be a dictionary")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail = [dict(row) for row in payload.get("detail", [])]
    training = [dict(row) for row in payload.get("training_pairs", [])]
    paths = {
        "json": out / "historical-backtest.json", "csv": out / "historical-backtest.csv",
        "markdown": out / "historical-backtest.md", "html": out / "historical-backtest.html",
        "training_csv": out / "historical-backtest-training.csv",
    }
    paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(paths["csv"], detail, _DETAIL_COLUMNS)
    _write_csv(paths["training_csv"], training)
    paths["markdown"].write_text(_markdown(payload), encoding="utf-8")
    paths["html"].write_text(_html(payload), encoding="utf-8")
    return paths
