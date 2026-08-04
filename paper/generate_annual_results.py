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

from figure_style import (
    BLUE,
    DARK_GREY,
    GREEN,
    GREY,
    HALF_TEXT_WIDTH,
    LIGHT_GREY,
    ORANGE,
    PINK,
    SKY,
    TEXT_WIDTH,
    hour_axis,
    legend_above,
    save,
    use_paper_style,
)


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


def build_cost_component_table() -> pd.DataFrame:
    """Reconcile annual settlement cost into additive grid-balance terms."""

    component_columns = {
        "IT electrical demand": "p_it_total_kw",
        "CRAC chiller": "p_chiller_hvac_kw",
        "TES charging chiller": "p_chiller_tes_kw",
    }
    scenario_frames: dict[str, pd.DataFrame] = {}
    for label, scenario in (("Baseline", BASELINE), ("Optimised", CENTRAL)):
        frame = pd.read_csv(ANNUAL / scenario / "annual_committed.csv")
        frame["UPS net grid effect"] = (
            frame["p_ups_charge_kw"] - frame["p_ups_discharge_kw"]
        )
        represented = (
            frame["p_it_total_kw"]
            + frame["p_chiller_hvac_kw"]
            + frame["p_chiller_tes_kw"]
            + frame["UPS net grid effect"]
        )
        frame["Auxiliary overhead"] = frame["grid_import_kw"] - represented
        scenario_frames[label] = frame

    rows: list[dict] = []
    ordered = [
        *component_columns,
        "UPS net grid effect",
        "Auxiliary overhead",
    ]
    for component in ordered:
        column = component_columns.get(component, component)
        values: dict[str, float] = {}
        energies: dict[str, float] = {}
        for label, frame in scenario_frames.items():
            price = frame["settlement_price_gbp_per_mwh"]
            values[label] = float((frame[column] * price * 0.25 / 1000).sum())
            energies[label] = float(frame[column].sum() * 0.25)
        rows.append(
            {
                "component": component,
                "baseline_cost_gbp": values["Baseline"],
                "optimised_cost_gbp": values["Optimised"],
                "cost_change_gbp": values["Optimised"] - values["Baseline"],
                "baseline_energy_kwh": energies["Baseline"],
                "optimised_energy_kwh": energies["Optimised"],
            }
        )
    table = pd.DataFrame(rows)
    total = {
        "component": "Total settlement cost",
        "baseline_cost_gbp": float(table["baseline_cost_gbp"].sum()),
        "optimised_cost_gbp": float(table["optimised_cost_gbp"].sum()),
        "cost_change_gbp": float(table["cost_change_gbp"].sum()),
        "baseline_energy_kwh": float(table["baseline_energy_kwh"].sum()),
        "optimised_energy_kwh": float(table["optimised_energy_kwh"].sum()),
    }
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    for label, frame in scenario_frames.items():
        expected = float(frame["settlement_cost_gbp"].sum())
        column = "baseline_cost_gbp" if label == "Baseline" else "optimised_cost_gbp"
        if abs(float(table.iloc[-1][column]) - expected) > 1e-6:
            raise AssertionError(f"{label} component cost does not reconcile")
    table.to_csv(REPORT / "annual_cost_components.csv", index=False)
    return table


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
    use_paper_style()
    _plot_day_import_and_price(baseline, central)
    _plot_day_workload(baseline, central)
    _plot_day_dispatch(baseline, central)


def _plot_day_import_and_price(
    baseline: pd.DataFrame, central: pd.DataFrame
) -> None:
    fig, axis = plt.subplots(
        figsize=(HALF_TEXT_WIDTH, 2.55), layout="constrained"
    )
    price_axis = axis.twinx()
    price_axis.set_zorder(axis.get_zorder() - 1)
    axis.patch.set_visible(False)

    price = central["settlement_price_gbp_per_mwh"]
    price_axis.fill_between(
        central["hour"],
        price,
        step="post",
        color=ORANGE,
        alpha=0.09,
        linewidth=0,
    )
    price_axis.step(
        central["hour"],
        price,
        where="post",
        color=ORANGE,
        linewidth=0.9,
        label="IMRP price",
    )
    price_axis.set_ylabel("IMRP (GBP/MWh)", color="#A8730B")
    price_axis.tick_params(axis="y", colors="#A8730B")
    price_axis.spines["right"].set_visible(True)
    price_axis.spines["right"].set_color("#A8730B")
    price_axis.grid(False)
    price_axis.set_ylim(0, float(price.max()) * 1.30)

    axis.plot(
        baseline["hour"],
        baseline["grid_import_kw"],
        color=GREY,
        linewidth=1.0,
        linestyle=(0, (4, 1.6)),
        label="Benchmark import",
    )
    axis.plot(
        central["hour"],
        central["grid_import_kw"],
        color=BLUE,
        linewidth=1.3,
        label="Optimised import",
    )
    axis.set_ylabel("Grid import (kW)")
    axis.set_xlabel("Local time (h)")
    axis.set_ylim(0, 1500)
    hour_axis(axis)

    handles, labels = axis.get_legend_handles_labels()
    price_handles, price_labels = price_axis.get_legend_handles_labels()
    legend_above(axis, 3, handles + price_handles, labels + price_labels)
    save(fig, IMAGES / "Figure_4a.png")


def _plot_day_workload(baseline: pd.DataFrame, central: pd.DataFrame) -> None:
    fig, axis = plt.subplots(
        figsize=(HALF_TEXT_WIDTH, 2.55), layout="constrained"
    )
    hour = central["hour"]
    axis.fill_between(
        hour,
        0,
        central["inflexible_cpu"],
        step="post",
        color=LIGHT_GREY,
        label="Inflexible",
    )
    axis.fill_between(
        hour,
        central["inflexible_cpu"],
        central["total_cpu"],
        step="post",
        color=SKY,
        alpha=0.65,
        label="Flexible, as executed",
    )
    axis.step(
        baseline["hour"],
        baseline["total_cpu"],
        where="post",
        color=DARK_GREY,
        linewidth=1.0,
        linestyle=(0, (4, 1.6)),
        label="Total at arrival",
    )
    axis.step(
        hour,
        central["total_cpu"],
        where="post",
        color=GREEN,
        linewidth=1.2,
        label="Total as executed",
    )
    axis.set_ylabel("Aggregate CPU utilisation")
    axis.set_xlabel("Local time (h)")
    axis.set_ylim(0, 0.95)
    hour_axis(axis)
    legend_above(axis, 2)
    save(fig, IMAGES / "Figure_4b.png")


def _plot_day_dispatch(baseline: pd.DataFrame, central: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 2.85), layout="constrained")
    hour = central["hour"].to_numpy()
    stack = [
        ("Auxiliary", np.full(len(central), 53.095), LIGHT_GREY),
        ("IT load", central["p_it_total_kw"].to_numpy(), BLUE),
        ("CRAC chiller", central["p_chiller_hvac_kw"].to_numpy(), ORANGE),
        ("TES charging", central["p_chiller_tes_kw"].to_numpy(), GREEN),
        ("UPS charging", central["p_ups_charge_kw"].to_numpy(), PINK),
    ]
    bottom = np.zeros(len(central))
    for label, values, colour in stack:
        axis.bar(
            hour,
            values,
            width=0.25,
            bottom=bottom,
            align="edge",
            label=label,
            color=colour,
            linewidth=0,
        )
        bottom += values
    discharge = central["p_ups_discharge_kw"].to_numpy()
    axis.bar(
        hour,
        -discharge,
        width=0.25,
        align="edge",
        label="UPS discharge",
        color="#EBC3DD",
        linewidth=0,
    )

    # The stack is an exact decomposition of optimised grid import; assert it
    # so the figure cannot silently drift from the settlement accounting.
    closure = np.abs(bottom - discharge - central["grid_import_kw"].to_numpy())
    if closure.max() > 1e-6:
        raise AssertionError("Dispatch stack does not close on grid import")

    axis.plot(
        baseline["hour"],
        baseline["grid_import_kw"],
        color=DARK_GREY,
        linestyle=(0, (3.5, 1.6)),
        linewidth=1.0,
        label="Benchmark import",
    )
    axis.axhline(0, color=DARK_GREY, linewidth=0.6)
    axis.set_ylabel("Power (kW)")
    axis.set_xlabel("Local time (h)")
    hour_axis(axis, step=2.0)
    axis.set_ylim(-1.25 * float(discharge.max()), 1450)
    legend_above(axis, 7)
    save(fig, IMAGES / "Figure_5.png")


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
            "Flexible-workload share (%)",
            "(a) Flexible workload",
            [
                _realised_flexible_share(value)
                for value in (0.5, 0.75, 1.0, 1.25, 1.5)
            ],
            ["flex_min", "flex_075", "central", "flex_125", "flex_max"],
        ),
        (
            "ups",
            "UPS energy capacity (kWh)",
            "(b) UPS capacity",
            [300, 450, 600, 750, 900],
            ["ups_min", "ups_075", "central", "ups_125", "ups_max"],
        ),
        (
            "tes",
            "TES energy capacity (kWh-th)",
            "(c) TES capacity",
            [500, 750, 1000, 1250, 1500],
            ["tes_min", "tes_075", "central", "tes_125", "tes_max"],
        ),
    ]
    use_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH, 2.25), layout="constrained"
    )
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
    for axis, (parameter, xlabel, title, x_values, cases) in zip(
        axes, definitions
    ):
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
            linestyle=(0, (3.5, 1.6)),
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
            color=GREY,
            label="Representative-week response shape",
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
            linestyle="-",
            marker="s",
            markersize=3.4,
            linewidth=1.4,
            color=BLUE,
            label="Full-year result",
        )
        axis.axhline(0, color=DARK_GREY, linewidth=0.6)
        axis.scatter(
            [annual_x[parameter][1]],
            [0.0],
            s=16,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.9,
            zorder=5,
        )
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.margins(x=0.08)
    axes[0].set_ylabel("Change in saving from\ncentral case (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    save(fig, IMAGES / "Figure_9.png")


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


def write_cost_component_latex(table: pd.DataFrame) -> None:
    shown = table.copy()
    for column in ("baseline_cost_gbp", "optimised_cost_gbp", "cost_change_gbp"):
        shown[column] = shown[column].map(lambda value: f"{value:,.0f}")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Annual signed settlement-cost composition.}",
        r"\label{tab:annual-components}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Component & Baseline & Optimised & Change \\",
        r" & \multicolumn{3}{c}{(GBP)} \\",
        r"\midrule",
    ]
    for row in shown.itertuples(index=False):
        if row.component == "Total settlement cost":
            lines.append(r"\midrule")
        lines.append(
            f"{row.component} & {row.baseline_cost_gbp} & "
            f"{row.optimised_cost_gbp} & {row.cost_change_gbp} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (GENERATED / "annual_cost_component_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    endpoints = build_endpoint_table()
    components = build_cost_component_table()
    selected, candidates = select_representative_day()
    plot_representative_day(selected)
    plot_sensitivity(endpoints)
    write_latex_table(endpoints)
    write_cost_component_latex(components)
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
