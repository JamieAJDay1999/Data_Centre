"""Refresh Scenario 3 on the annual model's representative operating day."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.model import solve_horizon
from rolling_optimisation.timeline import (
    add_optimisation_prices,
    apply_flexible_workload_multiplier,
    build_annual_timeline,
    local_day_core_indices,
)
from rolling_optimisation.types import (
    FlexibilityRequest,
    HorizonResult,
    OperationalState,
    WorkloadCohort,
)


ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "static/data/imrp_actuals.csv"
LOAD = ROOT / "static/data/inputs/load_profiles.csv"
SHIFT = ROOT / "static/data/inputs/shiftability_profile.csv"
ANNUAL = ROOT / "static/data/rolling_year_outputs"
SOURCE = "2025_optimised_reformulated"
REPRESENTATIVE_DAY_FILE = ROOT / "reports/final_annual_results/representative_day.txt"
OUTPUT = ROOT / "static/data/representative_day_flexibility"
REPORT = ROOT / "reports/representative_day_flexibility"
IMAGES = ROOT / "paper/images"
START_STEPS = tuple(range(0, 96, 12))
MAGNITUDES_KW = (-500, -400, -300, -200, -100, 25, 50, 75)
MAX_DURATION_STEPS = 48


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(
    date: str, time_limit_s: int
) -> tuple[
    RollingConfig,
    pd.DataFrame,
    int,
    OperationalState,
    list[WorkloadCohort],
    HorizonResult,
]:
    source = ANNUAL / SOURCE
    metadata = _json(source / "run_metadata.json")
    checkpoint = _json(source / "checkpoints" / f"{date}.json")
    config = RollingConfig(**metadata["config"])
    config = replace(
        config,
        scenario_id=f"representative_flex_{date}",
        solver_time_limit_s=time_limit_s,
        fail_on_gap_exceeded=False,
    )
    timeline = build_annual_timeline(
        PRICE, LOAD, SHIFT, 2025, config.lookahead_steps, "actual"
    )
    timeline = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    indices = dict(local_day_core_indices(timeline, 2025))[date]
    start = int(indices[0])
    core_steps = len(indices)
    horizon = timeline.iloc[
        start : start + core_steps + config.lookahead_steps
    ].copy()
    opening_state = OperationalState.from_dict(checkpoint["opening_state"])
    opening_workload = [
        WorkloadCohort.from_dict(row) for row in checkpoint["opening_workload"]
    ]
    baseline = solve_horizon(
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
    )
    stored = pd.read_csv(source / "days" / f"{date}.csv")
    maximum_difference = max(
        float((stored[column] - baseline.committed[column]).abs().max())
        for column in stored.columns
        if column in baseline.committed
        and pd.api.types.is_numeric_dtype(stored[column])
        and pd.api.types.is_numeric_dtype(baseline.committed[column])
    )
    if maximum_difference > 1e-7:
        raise RuntimeError(
            f"Representative baseline did not reproduce annual day: {maximum_difference}"
        )
    return (
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
        baseline,
    )


def _solve_request(
    config: RollingConfig,
    horizon: pd.DataFrame,
    core_steps: int,
    opening_state: OperationalState,
    opening_workload: list[WorkloadCohort],
    baseline: HorizonResult,
    start_step: int,
    duration_steps: int,
    magnitude_kw: float,
) -> HorizonResult | None:
    request = FlexibilityRequest(
        baseline_grid_import_kw=tuple(
            baseline.committed["grid_import_kw"].astype(float)
        ),
        start_step=start_step,
        duration_steps=duration_steps,
        delta_kw=magnitude_kw,
        recovery_state=baseline.terminal_state,
    )
    try:
        return solve_horizon(
            config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
            flexibility_request=request,
        )
    except RuntimeError as error:
        if "did not return a feasible accepted solution" in str(error):
            return None
        raise


def _maximum_duration(
    config: RollingConfig,
    horizon: pd.DataFrame,
    core_steps: int,
    opening_state: OperationalState,
    opening_workload: list[WorkloadCohort],
    baseline: HorizonResult,
    start_step: int,
    magnitude_kw: float,
) -> tuple[int, HorizonResult | None, int, float]:
    low = 1
    high = min(MAX_DURATION_STEPS, core_steps - start_step)
    best_steps = 0
    best_result: HorizonResult | None = None
    solves = 0
    runtime = 0.0
    while low <= high:
        duration = (low + high) // 2
        result = _solve_request(
            config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
            baseline,
            start_step,
            duration,
            magnitude_kw,
        )
        solves += 1
        if result is None:
            high = duration - 1
            continue
        runtime += float(result.solver["runtime_s"])
        best_steps = duration
        best_result = result
        low = duration + 1
    if best_steps and (
        best_result is None
        or len(best_result.committed) < start_step + best_steps
    ):
        raise AssertionError("Missing best flexibility dispatch")
    return best_steps, best_result, solves, runtime


def _plot_heatmap(results: pd.DataFrame, date: str) -> None:
    heat = results.pivot(
        index="magnitude_kw", columns="start_hour", values="duration_hours"
    ).sort_index(ascending=False)
    fig, axis = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="mako_r",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Maximum duration (hours)"},
        ax=axis,
    )
    axis.set_xlabel("Event start time (local hour)")
    axis.set_ylabel("Requested grid-power deviation (kW)")
    axis.set_title(f"Representative-day flexibility envelope: {date}")
    fig.tight_layout()
    fig.savefig(IMAGES / "Figure_6.png", dpi=220)
    plt.close(fig)


def _plot_components(
    detailed: dict[tuple[int, float], HorizonResult],
    baseline: HorizonResult,
    magnitudes: tuple[float, float],
    filename: str,
) -> None:
    starts = (12, 60)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    colours = {
        "IT/grid": "#0072B2",
        "CRAC": "#E69F00",
        "TES chiller": "#009E73",
        "UPS charging": "#56B4E9",
    }
    for row, magnitude in enumerate(magnitudes):
        for column, start in enumerate(starts):
            axis = axes[row, column]
            result = detailed.get((start, magnitude))
            if result is None:
                axis.text(0.5, 0.5, "Infeasible", ha="center", va="center")
                continue
            duration = int(
                round(
                    float(
                        result.solver.get("flexibility_duration_steps", 0)
                    )
                )
            )
            if not duration:
                duration = min(
                    len(result.committed) - start,
                    MAX_DURATION_STEPS,
                )
            stop = start + duration
            current = result.committed.iloc[start:stop]
            reference = baseline.committed.iloc[start:stop]
            components = {
                "IT/grid": current["p_grid_it_kw"].to_numpy()
                - reference["p_grid_it_kw"].to_numpy(),
                "CRAC": current["p_chiller_hvac_kw"].to_numpy()
                - reference["p_chiller_hvac_kw"].to_numpy(),
                "TES chiller": current["p_chiller_tes_kw"].to_numpy()
                - reference["p_chiller_tes_kw"].to_numpy(),
                "UPS charging": current["p_ups_charge_kw"].to_numpy()
                - reference["p_ups_charge_kw"].to_numpy(),
            }
            x = np.arange(len(current)) / 4.0
            positive = np.zeros(len(current))
            negative = np.zeros(len(current))
            for label, values in components.items():
                pos = np.clip(values, 0, None)
                neg = np.clip(values, None, 0)
                axis.bar(
                    x,
                    pos,
                    width=0.24,
                    bottom=positive,
                    color=colours[label],
                    label=label,
                )
                axis.bar(
                    x,
                    neg,
                    width=0.24,
                    bottom=negative,
                    color=colours[label],
                )
                positive += pos
                negative += neg
            axis.axhline(magnitude, color="black", linestyle="--", linewidth=1.5)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_title(
                f"{start / 4:02.0f}:00, {magnitude:+.0f} kW"
            )
            axis.set_xlabel("Hours from event start")
            if column == 0:
                axis.set_ylabel("Component change (kW)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.87, hspace=0.35, wspace=0.15)
    fig.savefig(IMAGES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = args.date or REPRESENTATIVE_DAY_FILE.read_text(
        encoding="utf-8"
    ).strip()
    (
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
        baseline,
    ) = _prepare(date, args.time_limit)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    baseline.planned.to_csv(OUTPUT / f"{date}_baseline_planned.csv", index=False)

    cells = (
        ((0, -100.0), (60, 25.0))
        if args.benchmark
        else tuple(
            (start, float(magnitude))
            for start in START_STEPS
            for magnitude in MAGNITUDES_KW
        )
    )
    rows: list[dict] = []
    detailed: dict[tuple[int, float], HorizonResult] = {}
    for position, (start, magnitude) in enumerate(cells, start=1):
        duration, result, solves, runtime = _maximum_duration(
            config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
            baseline,
            start,
            magnitude,
        )
        if result is not None:
            result.solver["flexibility_duration_steps"] = duration
            if start in {12, 60} and magnitude in {-200, -100, 25, 75}:
                detailed[(start, magnitude)] = result
                result.planned.to_csv(
                    OUTPUT
                    / f"{date}_start_{start:02d}_magnitude_{magnitude:+.0f}.csv",
                    index=False,
                )
        row = {
            "date": date,
            "start_step": start,
            "start_hour": start / 4.0,
            "magnitude_kw": magnitude,
            "duration_steps": duration,
            "duration_hours": duration / 4.0,
            "solver_calls": solves,
            "solver_runtime_s": runtime,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(REPORT / "flexibility_results.csv", index=False)
        print(
            f"[{position}/{len(cells)}] start={start / 4:.2f}h, "
            f"delta={magnitude:+.0f} kW -> {duration / 4:.2f}h "
            f"({solves} solves, {runtime:.2f}s)"
        )

    results = pd.DataFrame(rows)
    summary = {
        "date": date,
        "benchmark": bool(args.benchmark),
        "cells": len(results),
        "solver_calls": int(results["solver_calls"].sum()),
        "solver_runtime_s": float(results["solver_runtime_s"].sum()),
        "recovery_rule": (
            "UPS/TES no worse than baseline horizon terminal energy; "
            "thermal terminal states within 0.05 C of baseline"
        ),
    }
    (REPORT / ("benchmark.json" if args.benchmark else "summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not args.benchmark:
        _plot_heatmap(results, date)
        _plot_components(detailed, baseline, (-100, -200), "Figure_7.png")
        _plot_components(detailed, baseline, (75, 25), "Figure_8.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
