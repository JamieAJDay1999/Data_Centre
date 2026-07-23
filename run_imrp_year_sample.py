"""Run the existing paper model over an evenly spaced sample of IMRP days.

This is intentionally a thin wrapper around ``sensitivity_sweep.py``.  It
reuses that module's isolated Scenario 1 nominal calculation, Scenario 2
optimisation, component accounting, per-case checkpoints, and JSON outputs.

Default trial (50 evenly spaced standard days from 2025):

    python run_imrp_year_sample.py

Useful supporting commands:

    python run_imrp_year_sample.py --dry-run
    python run_imrp_year_sample.py --sample-days 2
    python run_imrp_year_sample.py --report-only
    python run_imrp_year_sample.py --sample-days all --floor-negative-prices

Full standard-day year after the trial is accepted:

    python run_imrp_year_sample.py --sample-days all
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "static" / "data" / "imrp_actuals.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "static" / "data" / "imrp_annual_sample_outputs"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "imrp_annual_sample"
REQUIRED_COLUMNS = {"IMRP_Date", "Settlement_Period", "IMRP_Amount"}

METRICS = {
    "base_cost_gbp": ("Scenario 1 base cost", "GBP/day"),
    "opt_cost_gbp": ("Scenario 2 optimised cost", "GBP/day"),
    "saving_gbp": ("Absolute cost saving", "GBP/day"),
    "saving_pct": ("Relative cost saving", "%"),
    "saving_it_gbp": ("IT cost saving", "GBP/day"),
    "saving_cooling_gbp": ("Cooling cost saving", "GBP/day"),
    "saving_ups_charge_gbp": ("UPS charging cost difference", "GBP/day"),
    "saving_other_gbp": ("Other-load cost saving", "GBP/day"),
}


def parse_sample_days(value: str) -> int | None:
    if str(value).strip().lower() == "all":
        return None
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--sample-days must be a positive integer or 'all'")
    return count


def load_observed_horizons(
    csv_path: Path,
    year: int,
    floor_negative_prices: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return 27-hour price horizons for standard 24-period days in ``year``.

    Each horizon contains the selected day's 24 hourly prices followed by the
    next settlement day's first three prices.  Fixed-horizon runs exclude the
    23- and 25-period daylight-saving dates.
    """
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    data["IMRP_Date"] = pd.to_datetime(data["IMRP_Date"], errors="raise").dt.normalize()
    data["Settlement_Period"] = pd.to_numeric(data["Settlement_Period"], errors="raise").astype(int)
    data["IMRP_Amount"] = pd.to_numeric(data["IMRP_Amount"], errors="raise")
    data = data.sort_values(["IMRP_Date", "Settlement_Period"], kind="stable")

    if data.duplicated(["IMRP_Date", "Settlement_Period"]).any():
        raise ValueError("Duplicate IMRP date/settlement-period keys found")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Null values found in required IMRP columns")

    by_date = {date: group for date, group in data.groupby("IMRP_Date", sort=True)}
    target_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    horizons: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []

    for date in target_dates:
        group = by_date.get(date)
        next_date = date + pd.Timedelta(days=1)
        next_group = by_date.get(next_date)
        periods = [] if group is None else group["Settlement_Period"].tolist()
        next_periods = [] if next_group is None else next_group["Settlement_Period"].tolist()
        standard_day = periods == list(range(1, 25))
        extension_available = next_periods[:3] == [1, 2, 3]
        reason = "included"
        if not standard_day:
            reason = f"excluded: {len(periods)} settlement periods"
        elif not extension_available:
            reason = "excluded: following-day extension unavailable"

        audit_row = {
            "date": date.date().isoformat(),
            "settlement_periods": len(periods),
            "next_date": next_date.date().isoformat(),
            "extension_periods_available": int(extension_available),
            "status": reason,
            "negative_prices_in_27h_original": None,
            "negative_prices_clipped_27h": 0,
        }
        if reason != "included":
            audit_rows.append(audit_row)
            continue

        core = group["IMRP_Amount"].to_numpy(dtype=float)
        extension = next_group.loc[next_group["Settlement_Period"].isin([1, 2, 3]), "IMRP_Amount"].to_numpy(dtype=float)
        if len(core) != 24 or len(extension) != 3:
            raise AssertionError(f"Unexpected horizon length for {date:%Y-%m-%d}")
        horizon = np.concatenate([core, extension])
        negative_count = int((horizon < 0).sum())
        clipped_count = negative_count if floor_negative_prices else 0
        if clipped_count:
            horizon = np.maximum(horizon, 0.0)
        audit_row["negative_prices_in_27h_original"] = negative_count
        audit_row["negative_prices_clipped_27h"] = clipped_count
        audit_rows.append(audit_row)
        horizons[date.date().isoformat()] = horizon

    if not horizons:
        raise ValueError(f"No valid 24-period days found for {year}")
    return horizons, pd.DataFrame(audit_rows)


def select_evenly_spaced_dates(available_dates: list[str], sample_days: int | None) -> list[str]:
    dates = sorted(available_dates)
    if sample_days is None:
        return dates
    if sample_days > len(dates):
        raise ValueError(f"Requested {sample_days} days but only {len(dates)} standard days are available")
    indices = np.rint(np.linspace(0, len(dates) - 1, sample_days)).astype(int)
    if len(set(indices.tolist())) != sample_days:
        raise AssertionError("Even-spacing produced duplicate sample positions")
    return [dates[index] for index in indices]


def case_id_for_date(date: str) -> str:
    return f"imrp_{date.replace('-', '')}"


def build_cases(
    selected_dates: list[str],
    clipped_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    clipped_counts = clipped_counts or {}
    return [
        {
            "case_id": case_id_for_date(date),
            "param": "price_day",
            "level": order,
            "price_scenario": date,
            "negative_prices_clipped_27h": int(clipped_counts.get(date, 0)),
            "price_floor_gbp_per_mwh": 0.0 if clipped_counts.get(date, 0) else None,
        }
        for order, date in enumerate(selected_dates, start=1)
    ]


def build_manifest(selected_dates: list[str], horizons: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    previous: pd.Timestamp | None = None
    for order, date_text in enumerate(selected_dates, start=1):
        date = pd.Timestamp(date_text)
        core = horizons[date_text][:24]
        rows.append(
            {
                "sample_order": order,
                "case_id": case_id_for_date(date_text),
                "date": date_text,
                "next_date_extension": (date + pd.Timedelta(days=1)).date().isoformat(),
                "gap_from_previous_days": None if previous is None else int((date - previous).days),
                "price_mean_gbp_per_mwh": float(core.mean()),
                "price_min_gbp_per_mwh": float(core.min()),
                "price_max_gbp_per_mwh": float(core.max()),
                "price_std_gbp_per_mwh": float(core.std(ddof=0)),
                "negative_price_hours": int((core < 0).sum()),
            }
        )
        previous = date
    return pd.DataFrame(rows)


def tariff_from_observed_horizon(hourly_27: np.ndarray, num_steps: int, dt_seconds: float) -> np.ndarray:
    """Expand 24 core hours + three actual extension hours to model timesteps."""
    hourly_27 = np.asarray(hourly_27, dtype=float)
    if hourly_27.shape != (27,):
        raise ValueError("Observed price horizon must contain exactly 27 hourly values")
    if 3600 % dt_seconds != 0:
        raise ValueError("dt_seconds must divide evenly into one hour")
    expanded = np.repeat(hourly_27, int(3600 // dt_seconds))
    if num_steps > len(expanded):
        raise ValueError(f"Model requested {num_steps} steps but observed horizon provides {len(expanded)}")
    return np.insert(expanded[:num_steps], 0, 0.0)


def configure_existing_pipeline(horizons: dict[str, np.ndarray], output_root: Path) -> ModuleType:
    """Point the existing sensitivity wrapper at observed prices and trial paths."""
    import sensitivity_sweep as sweep

    sweep.SWEEP_ROOT = output_root

    def install_observed_tariff(scenario: str) -> None:
        if scenario not in horizons:
            raise KeyError(f"No observed IMRP horizon registered for {scenario}")
        hourly_27 = horizons[scenario]

        def observed_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
            return tariff_from_observed_horizon(hourly_27, num_steps, dt_seconds)

        sweep.nom.generate_tariff = observed_tariff
        sweep.opt.generate_tariff = observed_tariff
        sweep.fd.generate_tariff = observed_tariff

    def observed_price_summary(scenario: str, path: Path | None = None) -> dict[str, Any]:
        del path
        core = horizons[scenario][:24]
        extension = horizons[scenario][24:]
        return {
            "price_scenario": scenario,
            "price_mean_gbp_per_mwh": round(float(core.mean()), 3),
            "price_min_gbp_per_mwh": round(float(core.min()), 3),
            "price_max_gbp_per_mwh": round(float(core.max()), 3),
            "price_std_gbp_per_mwh": round(float(core.std(ddof=0)), 3),
            "negative_price_hours": int((core < 0).sum()),
            "extension_price_mean_gbp_per_mwh": round(float(extension.mean()), 3),
        }

    sweep.install_price_scenario = install_observed_tariff
    sweep.price_scenario_summary = observed_price_summary
    return sweep


def solver_available() -> bool:
    import pyomo.environ as pyo

    return bool(pyo.SolverFactory("scip").available(exception_flag=False))


def execute_cases(
    sweep: ModuleType,
    cases: list[dict[str, Any]],
    force: bool,
    fail_fast: bool,
) -> dict[str, str]:
    """Run Scenario 1/2 cases serially because redirection uses module-level hooks."""
    errors: dict[str, str] = {}
    total = len(cases)
    for position, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        root = sweep.case_dirs(case_id)["root"]
        print(f"\n[{position}/{total}] {case['price_scenario']} ({case_id})")
        try:
            checkpoint = root / "tier1_result.json"
            if checkpoint.exists() and not force:
                print(f"  [resume] Scenario 1/2 checkpoint exists: {checkpoint}")
            else:
                sweep.run_tier1_case(case)
        except Exception:
            errors[case_id] = traceback.format_exc()
            print(f"  [failed] {case_id}\n{errors[case_id]}")
            if fail_fast:
                break
    return errors


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_results(
    cases: list[dict[str, Any]],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_index = pd.DataFrame(
        [
            {
                "case_id": case["case_id"],
                "date": case["price_scenario"],
                "sample_order": case["level"],
                "negative_prices_clipped_27h": case.get("negative_prices_clipped_27h", 0),
                "price_floor_gbp_per_mwh": case.get("price_floor_gbp_per_mwh"),
            }
            for case in cases
        ]
    )
    tier1_rows = []
    for case in cases:
        root = output_root / case["case_id"]
        tier1_path = root / "tier1_result.json"
        if tier1_path.exists():
            tier1_rows.append(_read_json(tier1_path))

    daily = case_index.copy()
    if tier1_rows:
        tier1 = pd.DataFrame(tier1_rows)
        daily = daily.merge(tier1, on="case_id", how="left", suffixes=("", "_result"))
        for component in ("it", "cooling", "ups_charge", "other"):
            base_col = f"base_cost_{component}_gbp"
            opt_col = f"opt_cost_{component}_gbp"
            if base_col in daily and opt_col in daily:
                daily[f"saving_{component}_gbp"] = daily[base_col] - daily[opt_col]

    daily = daily.sort_values("sample_order").reset_index(drop=True)
    summary_rows = []
    for column, (label, unit) in METRICS.items():
        if column not in daily:
            continue
        values = pd.to_numeric(daily[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append(
            {
                "metric": column,
                "label": label,
                "unit": unit,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "maximum": float(values.max()),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}"
    return html.escape(str(value))


def dataframe_html(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return '<p class="muted">No completed results are available for this section yet.</p>'
    shown = frame if max_rows is None else frame.head(max_rows)
    formatters = {column: format_number for column in shown.columns}
    return '<div class="table-wrap">' + shown.to_html(index=False, border=0, classes="data", formatters=formatters) + "</div>"


def outcome_figure_html(daily: pd.DataFrame) -> str:
    """Create one self-contained figure covering the recommended outcome views."""
    cost_columns = ["saving_pct", "saving_it_gbp", "saving_cooling_gbp", "saving_ups_charge_gbp", "saving_other_gbp"]
    has_cost = any(column in daily and daily[column].notna().any() for column in cost_columns)
    if not has_cost:
        return '<p class="muted">Figures will be embedded here after model results are available.</p>'

    plotted = daily.copy()
    plotted["date"] = pd.to_datetime(plotted["date"], errors="coerce")
    plotted = plotted.sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), constrained_layout=True)

    ax = axes[0]
    if "saving_pct" in plotted and plotted["saving_pct"].notna().any():
        values = pd.to_numeric(plotted["saving_pct"], errors="coerce")
        valid = plotted["date"].notna() & values.notna()
        q05, q25, median, q75, q95 = values[valid].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        ax.axhspan(q05, q95, color="#9ecae1", alpha=0.25, label="5th-95th percentile")
        ax.axhspan(q25, q75, color="#3182bd", alpha=0.22, label="Interquartile range")
        ax.axhline(median, color="#08519c", linewidth=1.8, label=f"Median ({median:.2f}%)")
        ax.plot(plotted.loc[valid, "date"], values[valid], "o-", color="#2f855a", markersize=4, linewidth=1)
        ax.set_ylabel("Cost saving (%)")
        ax.legend(frameon=False, ncol=3, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Cost-saving results not yet available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Scenario 2 saving across sampled dates")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    components = [
        ("saving_it_gbp", "IT"),
        ("saving_cooling_gbp", "Cooling"),
        ("saving_ups_charge_gbp", "UPS charge"),
        ("saving_other_gbp", "Other load"),
    ]
    component_data = []
    component_labels = []
    for column, label in components:
        if column in plotted:
            values = pd.to_numeric(plotted[column], errors="coerce").dropna()
            if not values.empty:
                component_data.append(values.to_numpy())
                component_labels.append(label)
    if component_data:
        box = ax.boxplot(component_data, tick_labels=component_labels, showmeans=True, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#bee3f8")
        ax.axhline(0, color="#4a5568", linewidth=0.8)
        ax.set_ylabel("Daily cost difference (GBP)")
    else:
        ax.text(0.5, 0.5, "Component results not yet available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Distribution of component contributions to saving")
    ax.grid(axis="y", alpha=0.25)

    axes[1].set_xlabel("Cost component")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<figure><img src="{uri}" alt="Cost saving and component contribution results">'
        '<figcaption>Daily points preserve seasonal variation. Shading on the saving panel shows the '
        'interquartile and 5th-95th percentile ranges; the tables below provide the exact values.</figcaption></figure>'
    )


def write_html_report(
    path: Path,
    year: int,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    report_only: bool,
    floor_negative_prices: bool,
    errors: dict[str, str],
) -> None:
    complete_cost = int(daily["saving_pct"].notna().sum()) if "saving_pct" in daily else 0
    excluded = audit.loc[audit["status"] != "included"]
    summary_columns = [
        "label", "unit", "count", "median", "iqr", "p05", "p95", "mean", "minimum", "maximum"
    ]
    summary_view = summary.reindex(columns=[column for column in summary_columns if column in summary])
    daily_columns = [
        "sample_order", "date", "price_mean_gbp_per_mwh", "price_min_gbp_per_mwh",
        "price_max_gbp_per_mwh", "base_cost_gbp", "opt_cost_gbp", "saving_gbp",
        "saving_pct", "negative_prices_clipped_27h", "price_floor_gbp_per_mwh", "runtime_s",
    ]
    daily_view = daily.reindex(columns=[column for column in daily_columns if column in daily])
    figures = outcome_figure_html(daily)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>IMRP {year} model sample report</title>
<style>
body{{margin:0;background:#f3f6f8;color:#1f2933;font-family:"Segoe UI",Arial,sans-serif;line-height:1.55}}main{{max-width:1180px;margin:auto;background:white;padding:38px 52px 70px}}h1,h2{{color:#102a43}}h1{{margin-bottom:4px}}h2{{margin-top:36px;border-bottom:2px solid #d9e2ec;padding-bottom:7px}}.muted{{color:#627d98}}.callout{{background:#e7f5ef;border-left:5px solid #18794e;padding:15px 18px;margin:22px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}.metric{{border:1px solid #d9e2ec;padding:13px}}.metric strong{{font-size:1.35rem;display:block}}.table-wrap{{overflow-x:auto;margin:16px 0 28px}}table.data{{border-collapse:collapse;width:100%;font-size:.86rem}}table.data th,table.data td{{border-bottom:1px solid #d9e2ec;padding:7px 9px;text-align:right;white-space:nowrap}}table.data th:first-child,table.data td:first-child{{text-align:left}}table.data th{{background:#edf2f7}}code{{background:#edf2f7;padding:2px 4px}}@media(max-width:700px){{main{{padding:24px 16px}}}}
figure{{margin:24px 0}}figure img{{display:block;width:100%;height:auto}}figcaption{{color:#52606d;font-size:.92rem;margin-top:8px}}  </style></head><body><main>
<h1>IMRP {year} nominal and optimisation sample report</h1><p class="muted">Generated {generated} | Mode: {"report only" if report_only else "nominal + optimisation"}</p>
<div class="callout"><strong>Interpretation rule:</strong> this trial reports the median, interquartile range, and 5th&ndash;95th percentile across evenly spaced sampled days. It is a pipeline test, not the final full-year estimate. The same wrapper can later be run with <code>--sample-days all</code>.</div>
<div class="metrics"><div class="metric"><strong>{len(manifest)}</strong>sampled days</div><div class="metric"><strong>{complete_cost}</strong>nominal + optimisation runs complete</div><div class="metric"><strong>{len(errors)}</strong>failed cases in latest run</div></div>
<h2>Method</h2><p>Dates are selected at evenly spaced positions across the {year} standard-day sequence. The 23- and 25-period daylight-saving dates are excluded from the fixed 24-hour model. Every run uses 24 observed IMRP prices and the following settlement day's first three observed prices for the extension horizon.</p><p>The wrapper calls only the existing Scenario 1 nominal calculation and Scenario 2 optimisation workflow, including its component cost accounting.</p>{'<p><strong>Recovery price treatment:</strong> after each 27-hour horizon was assembled, negative prices were clipped to GBP 0/MWh. Existing completed checkpoints were resumed, so this treatment applies only to the recovered cases identified in the daily-results and audit tables.</p>' if floor_negative_prices else ''}
  <h2>Outcome figures</h2>{figures}
  <h2>Distribution summary</h2>{dataframe_html(summary_view)}
<h2>Daily results</h2>{dataframe_html(daily_view)}
<h2>Sample manifest</h2>{dataframe_html(manifest)}
<h2>Excluded fixed-horizon dates</h2>{dataframe_html(excluded)}
<h2>Reproducibility</h2><p>Execution is checkpointed under <code>static/data/imrp_annual_sample_outputs/&lt;case_id&gt;/</code>. Re-running resumes completed days unless <code>--force</code> is supplied. Aggregate CSV and JSON files sit beside this report.</p>
</main></body></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def output_paths(report_root: Path, requested_sample: int | None, selected_count: int) -> dict[str, Path]:
    label = f"sample_{selected_count}" if requested_sample is not None else f"all_{selected_count}"
    return {
        "manifest": report_root / f"{label}_manifest.csv",
        "daily": report_root / f"{label}_daily_results.csv",
        "summary": report_root / f"{label}_distribution_summary.csv",
        "metadata": report_root / f"{label}_run_metadata.json",
        "html": report_root / f"{label}_report.html",
        "audit": report_root / f"{label}_date_audit.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--sample-days", type=parse_sample_days, default=50, help="positive integer or 'all' (default: 50)")
    parser.add_argument("--report-only", action="store_true", help="rebuild aggregate outputs without invoking the model")
    parser.add_argument(
        "--floor-negative-prices",
        action="store_true",
        help="clip negative prices to zero after each 27-hour day-plus-extension horizon is assembled",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--force", action="store_true", help="rerun existing per-day checkpoints")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed day")
    parser.add_argument("--dry-run", action="store_true", help="write and print the manifest without invoking a solver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    report_root = args.report_root.resolve()
    horizons, audit = load_observed_horizons(input_path, args.year, args.floor_negative_prices)
    selected_dates = select_evenly_spaced_dates(list(horizons), args.sample_days)
    clipped_counts = audit.set_index("date")["negative_prices_clipped_27h"].fillna(0).astype(int).to_dict()
    cases = build_cases(selected_dates, clipped_counts)
    manifest = build_manifest(selected_dates, horizons)
    manifest = manifest.merge(
        audit[["date", "negative_prices_in_27h_original", "negative_prices_clipped_27h"]],
        on="date",
        how="left",
    )
    paths = output_paths(report_root, args.sample_days, len(selected_dates))
    report_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths["manifest"], index=False)
    audit.to_csv(paths["audit"], index=False)

    print(f"IMRP year: {args.year}")
    print(f"Standard 24-period days available: {len(horizons)}")
    print(f"Selected dates: {len(selected_dates)}")
    for row in manifest.itertuples(index=False):
        gap = "start" if pd.isna(row.gap_from_previous_days) else f"+{int(row.gap_from_previous_days)}d"
        print(f"  {row.sample_order:>2}. {row.date}  {gap:>6}  mean={row.price_mean_gbp_per_mwh:7.2f} GBP/MWh")
    print(f"Manifest: {paths['manifest']}")

    errors: dict[str, str] = {}
    started = datetime.now(timezone.utc)
    if args.dry_run:
        print("Dry run complete; no solver calls were made.")
        return

    if not args.report_only:
        if not solver_available():
            raise SystemExit("SCIP solver not found. Install it or add it to PATH before running model stages.")
        output_root.mkdir(parents=True, exist_ok=True)
        sweep = configure_existing_pipeline(horizons, output_root)
        errors = execute_cases(sweep, cases, args.force, args.fail_fast)

    daily, summary = aggregate_results(cases, output_root)
    daily.to_csv(paths["daily"], index=False)
    summary.to_csv(paths["summary"], index=False)
    write_html_report(
        paths["html"], args.year, manifest, audit, daily, summary,
        args.report_only, args.floor_negative_prices, errors,
    )
    metadata = {
        "year": args.year,
        "requested_sample_days": "all" if args.sample_days is None else args.sample_days,
        "available_standard_days": len(horizons),
        "selected_dates": selected_dates,
        "workflow": "scenario_1_nominal_then_scenario_2_optimisation",
        "report_only": args.report_only,
        "negative_price_floor_gbp_per_mwh": 0.0 if args.floor_negative_prices else None,
        "price_floor_scope": "per_selected_27_hour_horizon" if args.floor_negative_prices else None,
        "cases_with_clipped_prices": int(sum(case["negative_prices_clipped_27h"] > 0 for case in cases)),
        "force": args.force,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nAggregate outputs:")
    for name in ("daily", "summary", "html", "metadata"):
        print(f"  {name}: {paths[name]}")
    if errors:
        print(f"Completed with {len(errors)} failed cases; see {paths['metadata']}")
    else:
        print("Completed without recorded case failures.")


if __name__ == "__main__":
    main()
