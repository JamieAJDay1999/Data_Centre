"""Generate the final annual result package and paper figures.

This script reads only completed rolling-year and representative-week outputs.
It does not launch optimisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "static/data/rolling_year_outputs"
SAMPLED = ROOT / "static/data/monthly_week_sensitivity/2025"
LOAD_PROFILE = ROOT / "static/data/inputs/load_profiles.csv"
REPORT = ROOT / "reports/final_annual_results"
IMAGES = ROOT / "paper/images"
GENERATED = ROOT / "paper/generated"

BASELINE = "2025_baseline_reformulated"
CENTRAL = "2025_optimised_reformulated"
SCENARIOS = (
    ("Baseline", BASELINE, "baseline", 1.0),
    ("Flexible workload 0.5x", "2025_flex_0p5", "flex", 0.5),
    ("Flexible workload 1.0x", CENTRAL, "flex", 1.0),
    ("Flexible workload 1.5x", "2025_flex_1p5", "flex", 1.5),
    ("UPS capacity 0.5x", "2025_ups_0p5", "ups", 0.5),
    ("UPS capacity 1.5x", "2025_ups_1p5", "ups", 1.5),
    ("TES capacity 0.5x", "2025_tes_0p5", "tes", 0.5),
    ("TES capacity 1.5x", "2025_tes_1p5", "tes", 1.5),
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _realised_flexible_share(multiplier: float) -> float:
    load = pd.read_csv(LOAD_PROFILE)
    total = load["inflexible_load"] + load["flexible_load"]
    flexible = (load["flexible_load"] * multiplier).clip(upper=total)
    return 100.0 * float(flexible.sum() / total.sum())


def build_endpoint_table() -> pd.DataFrame:
    baseline_summary = _json(ANNUAL / BASELINE / "annual_summary.json")
    baseline_cost = float(baseline_summary["settlement_cost_gbp"])
    central_summary = _json(ANNUAL / CENTRAL / "annual_summary.json")
    central_saving = 100.0 * (
        baseline_cost - float(central_summary["settlement_cost_gbp"])
    ) / baseline_cost

    rows: list[dict] = []
    for label, scenario, parameter, multiplier in SCENARIOS:
        summary = _json(ANNUAL / scenario / "annual_summary.json")
        checkpoint = _json(
            ANNUAL / scenario / "checkpoints/2025-12-31.json"
        )
        cost = float(summary["settlement_cost_gbp"])
        saving = 100.0 * (baseline_cost - cost) / baseline_cost
        rows.append(
            {
                "label": label,
                "scenario": scenario,
                "parameter": parameter,
                "multiplier": multiplier,
                "annual_cost_gbp": cost,
                "saving_gbp": baseline_cost - cost,
                "saving_percent": saving,
                "change_from_central_pp": saving - central_saving,
                "grid_energy_kwh": float(summary["grid_energy_kwh"]),
                "peak_grid_import_kw": float(summary["peak_grid_import_kw"]),
                "non_optimal_horizons": int(summary["non_optimal_horizons"]),
                "maximum_recorded_gap_percent": 100.0
                * float(summary["maximum_recorded_solver_gap"] or 0.0),
                "final_workload_cpu_h": float(
                    summary["final_outstanding_workload_cpu_h"]
                ),
                "final_ups_energy_kwh": float(
                    checkpoint["closing_state"]["ups_energy_kwh"]
                ),
                "final_tes_energy_kwh": float(
                    checkpoint["closing_state"]["tes_energy_kwh"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT / "annual_endpoints.csv", index=False)
    return result


def select_representative_day() -> tuple[str, pd.DataFrame]:
    annual = pd.read_csv(ANNUAL / CENTRAL / "annual_committed.csv")
    annual["timestamp_utc"] = pd.to_datetime(annual["timestamp_utc"], utc=True)
    grouped = annual.groupby("local_date", sort=False)
    daily = grouped.agg(
        intervals=("timestamp_utc", "size"),
        mean_price=("settlement_price_gbp_per_mwh", "mean"),
        price_std=("settlement_price_gbp_per_mwh", "std"),
        minimum_price=("settlement_price_gbp_per_mwh", "min"),
        maximum_price=("settlement_price_gbp_per_mwh", "max"),
        mean_grid_import=("grid_import_kw", "mean"),
        peak_grid_import=("grid_import_kw", "max"),
        mean_total_cpu=("total_cpu", "mean"),
        settlement_cost=("settlement_cost_gbp", "sum"),
        opening_ups=("state_start_ups_energy_kwh", "first"),
        opening_tes=("state_start_tes_energy_kwh", "first"),
        opening_cold_aisle=("state_start_cold_aisle_temperature_c", "first"),
    ).reset_index()
    negative_share = grouped["settlement_price_gbp_per_mwh"].apply(
        lambda values: float((values < 0).mean())
    )
    daily["negative_price_share"] = daily["local_date"].map(negative_share)

    features = [
        "mean_price",
        "price_std",
        "minimum_price",
        "maximum_price",
        "negative_price_share",
        "mean_grid_import",
        "peak_grid_import",
        "mean_total_cpu",
        "settlement_cost",
        "opening_ups",
        "opening_tes",
        "opening_cold_aisle",
    ]
    candidates = daily[
        (daily["intervals"] == 96)
        & (daily["local_date"] < "2025-12-29")
    ].copy()
    annual_target = daily[features].mean()
    scales = daily[features].std(ddof=0).replace(0, 1.0)
    z = (candidates[features] - annual_target) / scales
    candidates["representative_score"] = np.sqrt((z**2).mean(axis=1))
    candidates = candidates.sort_values(
        ["representative_score", "local_date"], kind="stable"
    ).reset_index(drop=True)
    candidates.to_csv(REPORT / "representative_day_selection.csv", index=False)
    selected = str(candidates.iloc[0]["local_date"])
    (REPORT / "representative_day.txt").write_text(
        selected + "\n", encoding="utf-8"
    )
    return selected, candidates


def _representative_frames(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(ANNUAL / BASELINE / "days" / f"{date}.csv")
    central = pd.read_csv(ANNUAL / CENTRAL / "days" / f"{date}.csv")
    for frame in (baseline, central):
        frame["hour"] = np.arange(len(frame)) / 4.0
    return baseline, central


def plot_representative_day(date: str) -> None:
    baseline, central = _representative_frames(date)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        baseline["hour"],
        baseline["grid_import_kw"],
        color="#6b7280",
        linewidth=2,
        label="Baseline grid import",
    )
    axis.plot(
        central["hour"],
        central["grid_import_kw"],
        color="#0072B2",
        linewidth=2.2,
        label="Optimised grid import",
    )
    axis.set_xlabel("Local time (hour)")
    axis.set_ylabel("Grid import (kW)")
    axis.set_xlim(0, 24)
    price_axis = axis.twinx()
    price_axis.step(
        central["hour"],
        central["settlement_price_gbp_per_mwh"],
        where="post",
        color="#D55E00",
        alpha=0.75,
        linewidth=1.6,
        label="IMRP",
    )
    price_axis.set_ylabel("IMRP (GBP/MWh)")
    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = price_axis.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, loc="upper left")
    axis.set_title(f"Representative operating day: {date}")
    fig.tight_layout()
    fig.savefig(IMAGES / "Figure_4a.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        baseline["hour"],
        baseline["total_cpu"],
        color="#6b7280",
        linewidth=2,
        label="Workload at arrival",
    )
    axis.plot(
        central["hour"],
        central["total_cpu"],
        color="#009E73",
        linewidth=2,
        label="Optimised workload",
    )
    axis.fill_between(
        central["hour"],
        central["inflexible_cpu"],
        central["total_cpu"],
        color="#56B4E9",
        alpha=0.25,
        label="Processed flexible workload",
    )
    axis.set(xlabel="Local time (hour)", ylabel="CPU utilisation", xlim=(0, 24))
    axis.legend(loc="upper left")
    axis.set_title(f"Workload schedule on {date}")
    fig.tight_layout()
    fig.savefig(IMAGES / "Figure_4b.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    x = central["hour"]
    components = [
        ("IT from grid", central["p_grid_it_kw"], "#0072B2"),
        ("CRAC", central["p_chiller_hvac_kw"], "#E69F00"),
        ("TES charging chiller", central["p_chiller_tes_kw"], "#009E73"),
        ("UPS charging", central["p_ups_charge_kw"], "#56B4E9"),
    ]
    bottom = np.zeros(len(central))
    for label, values, colour in components:
        axis.bar(
            x,
            values,
            width=0.25,
            bottom=bottom,
            label=label,
            color=colour,
            edgecolor="none",
        )
        bottom += values.to_numpy()
    axis.bar(
        x,
        -central["p_ups_discharge_kw"],
        width=0.25,
        label="UPS discharge",
        color="#56B4E9",
        hatch="////",
        edgecolor="white",
    )
    axis.plot(
        x,
        baseline["grid_import_kw"],
        color="black",
        linestyle=":",
        linewidth=2,
        label="Baseline grid import",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Local time (hour)", ylabel="Power (kW)", xlim=(0, 24))
    axis.legend(ncol=2, fontsize=9, loc="upper left")
    axis.set_title(f"Optimised component dispatch on {date}")
    fig.tight_layout()
    fig.savefig(IMAGES / "Figure_5.png", dpi=220)
    plt.close(fig)


def plot_sensitivity(endpoints: pd.DataFrame) -> None:
    sampled = pd.read_csv(SAMPLED / "sensitivity_comparison.csv")
    central_sample = float(
        sampled.loc[
            sampled["case"] == "central", "estimated_annual_saving_percent"
        ].iloc[0]
    )
    baseline_cost = float(
        _json(ANNUAL / BASELINE / "annual_summary.json")["settlement_cost_gbp"]
    )
    central_cost = float(
        _json(ANNUAL / CENTRAL / "annual_summary.json")["settlement_cost_gbp"]
    )
    central_annual = 100.0 * (baseline_cost - central_cost) / baseline_cost

    definitions = [
        (
            "flex",
            "Realised flexible-workload share (%)",
            [
                _realised_flexible_share(value)
                for value in (0.5, 0.75, 1.0, 1.25, 1.5)
            ],
            ["flex_min", "flex_075", "central", "flex_125", "flex_max"],
        ),
        (
            "ups",
            "UPS energy capacity (kWh)",
            [300, 450, 600, 750, 900],
            ["ups_min", "ups_075", "central", "ups_125", "ups_max"],
        ),
        (
            "tes",
            "TES energy capacity (kWh-th)",
            [500, 750, 1000, 1250, 1500],
            ["tes_min", "tes_075", "central", "tes_125", "tes_max"],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    annual_scenarios = {
        "flex": ["2025_flex_0p5", CENTRAL, "2025_flex_1p5"],
        "ups": ["2025_ups_0p5", CENTRAL, "2025_ups_1p5"],
        "tes": ["2025_tes_0p5", CENTRAL, "2025_tes_1p5"],
    }
    annual_x = {
        "flex": [
            _realised_flexible_share(value) for value in (0.5, 1.0, 1.5)
        ],
        "ups": [300, 600, 900],
        "tes": [500, 1000, 1500],
    }
    for axis, (parameter, xlabel, x_values, cases) in zip(axes, definitions):
        sample_values = [
            float(
                sampled.loc[
                    sampled["case"] == case,
                    "estimated_annual_saving_percent",
                ].iloc[0]
            )
            - central_sample
            for case in cases
        ]
        axis.plot(
            x_values,
            sample_values,
            "--",
            marker="o",
            color="#777777",
            label="Representative-week response",
        )
        annual_values = []
        for scenario in annual_scenarios[parameter]:
            cost = float(
                _json(ANNUAL / scenario / "annual_summary.json")[
                    "settlement_cost_gbp"
                ]
            )
            annual_values.append(
                100.0 * (baseline_cost - cost) / baseline_cost - central_annual
            )
        axis.plot(
            annual_x[parameter],
            annual_values,
            "-",
            marker="s",
            linewidth=2.2,
            color="#0072B2",
            label="Full-year result",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Change in saving from central case (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.82, bottom=0.2, wspace=0.28)
    fig.savefig(IMAGES / "Figure_9.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(endpoints: pd.DataFrame) -> None:
    shown = endpoints[
        [
            "label",
            "annual_cost_gbp",
            "saving_percent",
            "change_from_central_pp",
            "grid_energy_kwh",
        ]
    ].copy()
    shown["annual_cost_gbp"] = shown["annual_cost_gbp"].map(
        lambda value: f"{value:,.0f}"
    )
    shown["saving_percent"] = shown["saving_percent"].map(
        lambda value: f"{value:.3f}"
    )
    shown["change_from_central_pp"] = shown["change_from_central_pp"].map(
        lambda value: f"{value:+.3f}"
    )
    shown.loc[shown["label"] == "Baseline", "change_from_central_pp"] = "--"
    shown["grid_energy_kwh"] = shown["grid_energy_kwh"].map(
        lambda value: f"{value / 1_000_000:.3f}"
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Full-year operating-cost and sensitivity results.}",
        r"\label{tab:annual-endpoints}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & Cost (GBP) & Saving (\%) & Change (pp) & Energy (GWh) \\",
        r"\midrule",
    ]
    for row in shown.itertuples(index=False):
        label = str(row.label)
        if label.endswith("x"):
            label = label[:-1] + r"$\times$"
        lines.append(
            f"{label} & {row.annual_cost_gbp} & {row.saving_percent} & "
            f"{row.change_from_central_pp} & {row.grid_energy_kwh} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (GENERATED / "annual_endpoint_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    endpoints = build_endpoint_table()
    selected, candidates = select_representative_day()
    plot_representative_day(selected)
    plot_sensitivity(endpoints)
    write_latex_table(endpoints)
    summary = {
        "representative_day": selected,
        "representative_score": float(candidates.iloc[0]["representative_score"]),
        "central_annual_saving_percent": float(
            endpoints.loc[
                endpoints["scenario"] == CENTRAL, "saving_percent"
            ].iloc[0]
        ),
        "endpoint_rows": len(endpoints),
    }
    (REPORT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
