"""Refresh Scenario 3 on the annual model's representative operating day."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.plot_flexibility_figures import (
    DOWNWARD_PANELS,
    UPWARD_PANELS,
    component_deltas as _component_deltas,
    plot_components,
    plot_duration_grid,
)
from rolling_optimisation.model import new_workload_cohorts, solve_horizon
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
SOURCE = "2025_optimised_cohort_trace"
REPRESENTATIVE_DAY_FILE = ROOT / "reports/final_annual_results/representative_day.txt"
OUTPUT = ROOT / "static/data/representative_day_flexibility"
REPORT = ROOT / "reports/representative_day_flexibility"
IMAGES = ROOT / "paper/images"
# A uniform hourly grid is dense enough to expose intraday structure while
# remaining legible as a printed heatmap.  It also exceeds the 240-cell grid
# used by the superseded independent-day study.
START_STEPS = tuple(range(0, 96, 4))
MAGNITUDES_KW = (
    -500,
    -450,
    -400,
    -350,
    -300,
    -250,
    -200,
    -150,
    -100,
    25,
    50,
    75,
)
MAX_DURATION_STEPS = 48
METHOD_SCHEMA_VERSION = 2
FIXED_RECOVERY_STEPS = 12
DETAIL_CELLS = {
    (24, -100.0),
    (60, -100.0),
    (24, -200.0),
    (60, -200.0),
    (0, 25.0),
    (24, 25.0),
    (0, 75.0),
    (24, 75.0),
}
# These eight cells feed the component-decomposition panels and therefore get
# a longer boundary-certification attempt when the normal screening solve times
# out.  Other dense-grid cells retain the certified duration as a lower bound.
EXTENDED_BOUNDARY_CELLS = DETAIL_CELLS


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_id(start_step: int, magnitude_kw: float) -> str:
    sign = "pos" if magnitude_kw >= 0 else "neg"
    return f"start_{start_step:02d}_magnitude_{sign}_{abs(magnitude_kw):.0f}"


def _state_at_boundary(frame: pd.DataFrame, boundary_step: int) -> OperationalState:
    if boundary_step < 0 or boundary_step > len(frame):
        raise IndexError(f"Boundary {boundary_step} is outside the baseline horizon")
    if boundary_step == len(frame):
        row = frame.iloc[-1]
        prefix = "state_end"
    else:
        row = frame.iloc[boundary_step]
        prefix = "state_start"
    return OperationalState(
        ups_energy_kwh=float(row[f"{prefix}_ups_energy_kwh"]),
        tes_energy_kwh=float(row[f"{prefix}_tes_energy_kwh"]),
        it_temperature_c=float(row[f"{prefix}_it_temperature_c"]),
        rack_temperature_c=float(row[f"{prefix}_rack_temperature_c"]),
        cold_aisle_temperature_c=float(row[f"{prefix}_cold_aisle_temperature_c"]),
        hot_aisle_temperature_c=float(row[f"{prefix}_hot_aisle_temperature_c"]),
    )


def _workload_at_boundary(
    config: RollingConfig,
    horizon: pd.DataFrame,
    opening_workload: list[WorkloadCohort],
    workload_trace: pd.DataFrame,
    boundary_step: int,
) -> list[WorkloadCohort]:
    """Return the exact annual cohort ledger before arrivals at the boundary."""

    if boundary_step < 0 or boundary_step > len(horizon):
        raise IndexError(f"Boundary {boundary_step} is outside the baseline horizon")
    if boundary_step == len(horizon):
        boundary = pd.Timestamp(horizon.iloc[-1]["timestamp_utc"]) + pd.Timedelta(
            seconds=config.dt_seconds
        )
    else:
        boundary = pd.Timestamp(horizon.iloc[boundary_step]["timestamp_utc"])

    candidates = opening_workload + new_workload_cohorts(horizon, config)
    eligible = [cohort for cohort in candidates if cohort.arrival < boundary]
    if workload_trace.empty:
        executed: dict[str, float] = {}
    else:
        before = workload_trace[workload_trace["timestamp_utc"] < boundary]
        executed = (
            before.groupby("cohort_id")["executed_cpu_hours"].sum().to_dict()
        )

    result: list[WorkloadCohort] = []
    for cohort in eligible:
        remaining = cohort.remaining_cpu_hours - float(executed.get(cohort.cohort_id, 0.0))
        if remaining < -config.workload_tolerance_cpu_h:
            raise RuntimeError(
                f"Annual trace over-executes {cohort.cohort_id} by {-remaining} CPU-h"
            )
        if remaining > config.workload_tolerance_cpu_h:
            result.append(
                WorkloadCohort(
                    cohort_id=cohort.cohort_id,
                    arrival_utc=cohort.arrival_utc,
                    latest_start_utc=cohort.latest_start_utc,
                    remaining_cpu_hours=max(0.0, remaining),
                    tranche=cohort.tranche,
                )
            )
    return result


def _validate_flexibility_result(
    result: HorizonResult,
    baseline: HorizonResult,
    start_step: int,
    duration_steps: int,
    magnitude_kw: float,
    tolerance_kw: float = 0.1,
    recovery_temperature_tolerance_c: float = 0.05,
) -> dict[str, float | bool | str | None]:
    stop_step = start_step + duration_steps
    recovery_boundary = stop_step + FIXED_RECOVERY_STEPS
    current = result.committed.iloc[:duration_steps]
    reference = baseline.committed.iloc[start_step:stop_step]
    realised = (
        current["grid_import_kw"].to_numpy()
        - reference["grid_import_kw"].to_numpy()
    )
    target_error = float(np.max(np.abs(realised - magnitude_kw)))
    recovery = _state_at_boundary(baseline.planned, recovery_boundary)
    response = result.terminal_state
    energy_shortfall = max(
        0.0,
        recovery.ups_energy_kwh - response.ups_energy_kwh,
        recovery.tes_energy_kwh - response.tes_energy_kwh,
    )
    temperature_error = max(
        abs(response.it_temperature_c - recovery.it_temperature_c),
        abs(response.rack_temperature_c - recovery.rack_temperature_c),
        abs(response.cold_aisle_temperature_c - recovery.cold_aisle_temperature_c),
        abs(response.hot_aisle_temperature_c - recovery.hot_aisle_temperature_c),
    )
    components = _component_deltas(current, reference)
    reconstructed = np.sum(np.vstack(tuple(components.values())), axis=0)
    closure_error = float(np.max(np.abs(reconstructed - realised)))
    workload_excess = float(
        result.audits.get("recovery_workload_max_excess_cpu_h", float("inf"))
    )
    valid = (
        target_error <= tolerance_kw + 1e-5
        and energy_shortfall <= 1e-5
        and temperature_error <= recovery_temperature_tolerance_c + 1e-5
        and closure_error <= 1e-5
        and workload_excess <= 1e-7 + 1e-9
        and abs(result.audits["objective_reconciliation_gbp"]) <= 1e-6
        and result.audits["core_workload_unserved_after_lookahead_cpu_h"] <= 1e-7
    )
    if not valid:
        raise RuntimeError(
            "Scenario 3 validation failed: "
            f"target={target_error}, energy={energy_shortfall}, "
            f"temperature={temperature_error}, workload={workload_excess}, "
            f"closure={closure_error}"
        )
    return {
        "validated": True,
        "maximum_target_error_kw": target_error,
        "maximum_recovery_energy_shortfall_kwh": energy_shortfall,
        "maximum_recovery_temperature_error_c": temperature_error,
        "maximum_recovery_workload_excess_cpu_h": workload_excess,
        "maximum_component_closure_error_kw": closure_error,
        "recovery_steps": FIXED_RECOVERY_STEPS,
        "solver_quality": str(result.solver["solution_quality"]),
        "solver_relative_gap": result.solver["relative_gap"],
        "solver_meets_accepted_gap": bool(result.solver["meets_accepted_gap"]),
    }


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
    if config.lookahead_steps != FIXED_RECOVERY_STEPS:
        raise ValueError(
            "Representative-day recovery requires exactly "
            f"{FIXED_RECOVERY_STEPS} look-ahead intervals"
        )
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
    stored = pd.read_csv(source / "annual_committed.csv")
    stored["timestamp_utc"] = pd.to_datetime(stored["timestamp_utc"], utc=True)
    required_timestamps = pd.to_datetime(horizon["timestamp_utc"], utc=True)
    baseline_planned = (
        stored.set_index("timestamp_utc").loc[required_timestamps].reset_index()
    )
    if len(baseline_planned) != core_steps + config.lookahead_steps:
        raise RuntimeError("Stored annual trajectory does not cover the full horizon")
    workload_trace = pd.read_csv(source / "annual_workload_trace.csv")
    workload_trace["timestamp_utc"] = pd.to_datetime(
        workload_trace["timestamp_utc"], utc=True
    )
    workload_trace = workload_trace[
        workload_trace["timestamp_utc"].isin(required_timestamps)
    ].copy()
    traced_rate = workload_trace.groupby("timestamp_utc")[
        "executed_cpu_rate"
    ].sum()
    maximum_trace_residual = max(
        abs(
            float(traced_rate.get(timestamp, 0.0))
            - float(baseline_planned.iloc[position]["flexible_cpu_processed"])
        )
        for position, timestamp in enumerate(required_timestamps)
    )
    if maximum_trace_residual > config.workload_tolerance_cpu_h:
        raise RuntimeError(
            "Representative-day workload trace does not match the annual CPU path: "
            f"{maximum_trace_residual}"
        )

    def state_from_row(row: pd.Series, prefix: str) -> OperationalState:
        return OperationalState(
            ups_energy_kwh=float(row[f"{prefix}_ups_energy_kwh"]),
            tes_energy_kwh=float(row[f"{prefix}_tes_energy_kwh"]),
            it_temperature_c=float(row[f"{prefix}_it_temperature_c"]),
            rack_temperature_c=float(row[f"{prefix}_rack_temperature_c"]),
            cold_aisle_temperature_c=float(
                row[f"{prefix}_cold_aisle_temperature_c"]
            ),
            hot_aisle_temperature_c=float(
                row[f"{prefix}_hot_aisle_temperature_c"]
            ),
        )

    stored_opening = state_from_row(baseline_planned.iloc[0], "state_start")
    opening_difference = max(
        abs(a - b)
        for a, b in zip(
            stored_opening.to_dict().values(), opening_state.to_dict().values()
        )
    )
    if opening_difference > config.state_tolerance:
        raise RuntimeError(
            f"Stored annual opening state mismatch: {opening_difference}"
        )
    next_state = state_from_row(
        baseline_planned.iloc[core_steps - 1], "state_end"
    )
    terminal_state = state_from_row(baseline_planned.iloc[-1], "state_end")
    baseline = HorizonResult(
        committed=baseline_planned.iloc[:core_steps].copy(),
        planned=baseline_planned.copy(),
        next_state=next_state,
        terminal_state=terminal_state,
        next_workload=[
            WorkloadCohort.from_dict(row) for row in checkpoint["closing_workload"]
        ],
        solver=checkpoint["solver"],
        audits=checkpoint["audits"],
        workload_trace=workload_trace,
        terminal_workload=_workload_at_boundary(
            config,
            horizon,
            opening_workload,
            workload_trace,
            len(horizon),
        ),
    )
    return (
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
        baseline,
    )


def _solve_attempt(
    config: RollingConfig,
    horizon: pd.DataFrame,
    core_steps: int,
    opening_state: OperationalState,
    opening_workload: list[WorkloadCohort],
    baseline: HorizonResult,
    start_step: int,
    duration_steps: int,
    magnitude_kw: float,
) -> tuple[HorizonResult | None, str]:
    del opening_state
    stop_step = start_step + duration_steps
    recovery_boundary = stop_step + FIXED_RECOVERY_STEPS
    if stop_step > core_steps or recovery_boundary > len(horizon):
        raise ValueError("Event plus fixed recovery lies outside the baseline horizon")
    event_horizon = horizon.iloc[start_step:recovery_boundary].copy()
    event_reference = baseline.planned.iloc[start_step:recovery_boundary].copy()
    event_initial_state = _state_at_boundary(baseline.planned, start_step)
    recovery_state = _state_at_boundary(baseline.planned, recovery_boundary)
    event_initial_workload = _workload_at_boundary(
        config,
        horizon,
        opening_workload,
        baseline.workload_trace,
        start_step,
    )
    recovery_workload = _workload_at_boundary(
        config,
        horizon,
        opening_workload,
        baseline.workload_trace,
        recovery_boundary,
    )
    request = FlexibilityRequest(
        baseline_grid_import_kw=tuple(
            event_reference["grid_import_kw"].astype(float)
        ),
        start_step=0,
        duration_steps=duration_steps,
        delta_kw=magnitude_kw,
        recovery_state=recovery_state,
        recovery_workload=tuple(recovery_workload),
    )
    try:
        result = solve_horizon(
            config,
            event_horizon,
            duration_steps,
            event_initial_state,
            event_initial_workload,
            flexibility_request=request,
        )
        result.planned.attrs["baseline_start_step"] = start_step
        result.planned.attrs["recovery_boundary_step"] = recovery_boundary
        return result, "feasible_incumbent"
    except RuntimeError as error:
        if "did not return a feasible accepted solution" in str(error):
            termination = str(error).rsplit(":", maxsplit=1)[-1].strip()
            quality = (
                "proven_infeasible"
                if termination == "infeasible"
                else f"{termination}_no_incumbent"
            )
            print(
                "  rejected flexibility trial "
                f"(start={start_step}, duration={duration_steps}, "
                f"delta={magnitude_kw:+.0f} kW): {error}"
            )
            return None, quality
        raise


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
    """Compatibility wrapper used by the single-cell verification command."""

    result, _ = _solve_attempt(
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
        baseline,
        start_step,
        duration_steps,
        magnitude_kw,
    )
    return result


def _maximum_duration(
    config: RollingConfig,
    horizon: pd.DataFrame,
    core_steps: int,
    opening_state: OperationalState,
    opening_workload: list[WorkloadCohort],
    baseline: HorizonResult,
    start_step: int,
    magnitude_kw: float,
) -> tuple[
    int,
    HorizonResult | None,
    list[dict[str, float | int | bool | str | None]],
    dict[str, float | bool | str | None],
]:
    # A small energy-cost objective is materially better at guiding HiGHS to a
    # feasible incumbent than a constant feasibility objective for the hard
    # recovery-boundary cases.  Every accepted incumbent is still audited
    # against the requested power deviation and terminal recovery constraints.
    maximum_candidate = min(MAX_DURATION_STEPS, core_steps - start_step)
    best_steps = 0
    best_result: HorizonResult | None = None
    trials: list[dict[str, float | int | bool | str | None]] = []

    def attempt(
        duration: int,
        purpose: str,
        trial_config: RollingConfig = config,
    ) -> tuple[HorizonResult | None, str]:
        started = time.perf_counter()
        result, quality = _solve_attempt(
            trial_config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
            baseline,
            start_step,
            duration,
            magnitude_kw,
        )
        wall_time = time.perf_counter() - started
        trials.append(
            {
                "duration_steps": duration,
                "feasible_incumbent": result is not None,
                "wall_time_s": wall_time,
                "solver_runtime_s": (
                    None if result is None else float(result.solver["runtime_s"])
                ),
                "solver_quality": (
                    quality
                    if result is None
                    else str(result.solver["solution_quality"])
                ),
                "solver_relative_gap": (
                    None if result is None else result.solver["relative_gap"]
                ),
                "time_limit_s": trial_config.solver_time_limit_s,
                "purpose": purpose,
            }
        )
        return result, quality

    # Test the complete admissible duration first, then use a logarithmic search.
    # This avoids solving every adjacent hour for long, feasible events.  A
    # timeout without an incumbent still narrows only the certified lower bound;
    # it is never labelled as a proof of infeasibility.
    upper_result, _ = attempt(maximum_candidate, "maximum_duration_screening")
    if upper_result is not None:
        best_steps = maximum_candidate
        best_result = upper_result
        boundary_verified = True
    else:
        low = 1
        high = maximum_candidate - 1
        while low <= high:
            duration = (low + high) // 2
            result, _ = attempt(duration, "duration_binary_search")
            if result is None:
                high = duration - 1
            else:
                best_steps = duration
                best_result = result
                low = duration + 1

        boundary = best_steps + 1
        prior_boundary_attempt = next(
            (
                row
                for row in reversed(trials)
                if row["duration_steps"] == boundary
            ),
            None,
        )
        if (
            prior_boundary_attempt is not None
            and prior_boundary_attempt["solver_quality"] == "proven_infeasible"
        ):
            boundary_verified = True
        else:
            boundary_result: HorizonResult | None = None
            boundary_quality = (
                "unresolved"
                if prior_boundary_attempt is None
                else str(prior_boundary_attempt["solver_quality"])
            )
            if prior_boundary_attempt is None:
                boundary_result, boundary_quality = attempt(
                    boundary,
                    "boundary_feasibility",
                )
            extended_boundary = (start_step, magnitude_kw) in EXTENDED_BOUNDARY_CELLS
            if (
                boundary_result is None
                and boundary_quality != "proven_infeasible"
                and extended_boundary
            ):
                boundary_result, boundary_quality = attempt(
                    boundary,
                    "extended_boundary_feasibility",
                    replace(config, solver_time_limit_s=300),
                )
            if boundary_result is not None:
                best_steps = boundary
                best_result = boundary_result
                boundary_verified = best_steps == maximum_candidate
            else:
                boundary_verified = boundary_quality == "proven_infeasible"
    if best_steps and (
        best_result is None
        or len(best_result.committed) < best_steps
    ):
        raise AssertionError("Missing best flexibility dispatch")
    validation: dict[str, float | bool | str | None] = {
        "validated": best_steps == 0,
        "boundary_verified": boundary_verified,
        "duration_is_lower_bound": not boundary_verified,
    }
    if best_result is not None:
        validation.update(
            _validate_flexibility_result(
                best_result,
                baseline,
                start_step,
                best_steps,
                magnitude_kw,
            )
        )
    return best_steps, best_result, trials, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore compatible per-cell checkpoints and recompute every cell.",
    )
    parser.add_argument(
        "--cell-start-step",
        type=int,
        default=None,
        help="Solve one cell in a fresh process (15-minute timestep index).",
    )
    parser.add_argument(
        "--cell-magnitude-kw",
        type=float,
        default=None,
        help="Magnitude paired with --cell-start-step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.cell_start_step is None) != (args.cell_magnitude_kw is None):
        raise ValueError(
            "--cell-start-step and --cell-magnitude-kw must be supplied together"
        )
    single_cell = args.cell_start_step is not None
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
    output_root = OUTPUT / "benchmark" if args.benchmark else OUTPUT
    report_root = REPORT / "benchmark" if args.benchmark else REPORT
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    if not single_cell:
        baseline.planned.to_csv(
            output_root / f"{date}_baseline_planned.csv", index=False
        )

    if single_cell:
        cells = ((int(args.cell_start_step), float(args.cell_magnitude_kw)),)
    elif args.benchmark:
        cells = ((0, -100.0), (60, 25.0))
    else:
        cells = tuple(
            (start, float(magnitude))
            for start in START_STEPS
            for magnitude in MAGNITUDES_KW
        )
    rows: list[dict] = []
    detailed: dict[tuple[int, float], pd.DataFrame] = {}
    started_all = time.perf_counter()
    for position, (start, magnitude) in enumerate(cells, start=1):
        cell_id = _cell_id(start, magnitude)
        cell_json = report_root / f"{cell_id}.json"
        cell_csv = output_root / f"{date}_{cell_id}.csv"
        if cell_json.exists() and not args.restart:
            payload = _json(cell_json)
            saved_result = payload.get("result", {})
            saved_limit = int(payload.get("time_limit_s", -1))
            saved_exact = bool(saved_result.get("boundary_verified", False))
            compatible = (
                int(payload.get("method_schema_version", -1))
                == METHOD_SCHEMA_VERSION
                and payload.get("date") == date
                and int(payload.get("start_step", -1)) == start
                and float(payload.get("magnitude_kw", float("nan"))) == magnitude
                and (
                    saved_limit == args.time_limit
                    or (saved_exact and 0 < saved_limit <= args.time_limit)
                )
            )
            if compatible:
                row = payload["result"]
                row.setdefault(
                    "duration_is_lower_bound",
                    not bool(row.get("boundary_verified", False)),
                )
                rows.append(row)
                if cell_csv.exists() and int(row["duration_steps"]) > 0:
                    planned = pd.read_csv(cell_csv)
                    planned.attrs["flexibility_duration_steps"] = int(
                        row["duration_steps"]
                    )
                    planned.attrs["baseline_start_step"] = start
                    detailed[(start, magnitude)] = planned
                print(
                    f"[{position}/{len(cells)}] resumed start={start / 4:.2f}h, "
                    f"delta={magnitude:+.0f} kW -> {row['duration_hours']:.2f}h"
                    f"{'+' if row['duration_is_lower_bound'] else ''}"
                )
                continue
        duration, result, trials, validation = _maximum_duration(
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
            result.planned.attrs["flexibility_duration_steps"] = duration
            result.planned.to_csv(cell_csv, index=False)
            detailed[(start, magnitude)] = result.planned
        solver_gaps = [
            float(row["solver_relative_gap"])
            for row in trials
            if row["solver_relative_gap"] is not None
        ]
        runtime = sum(float(row["wall_time_s"]) for row in trials)
        row = {
            "date": date,
            "start_step": start,
            "start_hour": start / 4.0,
            "magnitude_kw": magnitude,
            "duration_steps": duration,
            "duration_hours": duration / 4.0,
            "solver_calls": len(trials),
            "solver_runtime_s": runtime,
            "maximum_solver_relative_gap": max(solver_gaps, default=None),
            **validation,
        }
        rows.append(row)
        cell_payload = {
            "method_schema_version": METHOD_SCHEMA_VERSION,
            "date": date,
            "start_step": start,
            "magnitude_kw": magnitude,
            "time_limit_s": args.time_limit,
            "result": row,
            "trials": sorted(trials, key=lambda item: int(item["duration_steps"])),
            "solver": None if result is None else result.solver,
            "audits": None if result is None else result.audits,
            "terminal_state": (
                None if result is None else result.terminal_state.to_dict()
            ),
        }
        cell_json.write_text(json.dumps(cell_payload, indent=2), encoding="utf-8")
        if not single_cell:
            pd.DataFrame(rows).sort_values(
                ["start_step", "magnitude_kw"]
            ).to_csv(report_root / "flexibility_results.csv", index=False)
        print(
            f"[{position}/{len(cells)}] start={start / 4:.2f}h, "
            f"delta={magnitude:+.0f} kW -> {duration / 4:.2f}h "
            f"{'lower bound, ' if row['duration_is_lower_bound'] else ''}"
            f"({len(trials)} solves, {runtime:.2f}s)"
        )

    if single_cell:
        print(json.dumps(rows[0], indent=2))
        return

    results = pd.DataFrame(rows).sort_values(["start_step", "magnitude_kw"])
    results.to_csv(report_root / "flexibility_results.csv", index=False)
    summary = {
        "date": date,
        "benchmark": bool(args.benchmark),
        "cells": len(results),
        "solver_calls": int(results["solver_calls"].sum()),
        "solver_runtime_s": float(results["solver_runtime_s"].sum()),
        "wall_time_s": time.perf_counter() - started_all,
        "time_limit_s_per_solve": args.time_limit,
        "all_cells_validated": bool(results["validated"].all()),
        "all_boundaries_verified": bool(results["boundary_verified"].all()),
        "exact_duration_cells": int(results["boundary_verified"].sum()),
        "lower_bound_duration_cells": int(
            results["duration_is_lower_bound"].sum()
        ),
        "maximum_target_error_kw": float(
            results["maximum_target_error_kw"].fillna(0).max()
        ),
        "maximum_recovery_temperature_error_c": float(
            results["maximum_recovery_temperature_error_c"].fillna(0).max()
        ),
        "maximum_recovery_workload_excess_cpu_h": float(
            results["maximum_recovery_workload_excess_cpu_h"].fillna(0).max()
        ),
        "maximum_component_closure_error_kw": float(
            results["maximum_component_closure_error_kw"].fillna(0).max()
        ),
        "cells_with_solver_gap_above_one_percent": int(
            (results["maximum_solver_relative_gap"].fillna(0) > 0.01).sum()
        ),
        "recovery_rule": (
            "exact annual event-boundary cohort ledger; fixed three-hour "
            "post-event recovery; UPS/TES no worse than the annual baseline; "
            "thermal states within 0.05 C; per-cohort remaining work no greater "
            "than the annual baseline at the same boundary"
        ),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not args.benchmark:
        plot_duration_grid(results)
        plot_components(
            detailed, baseline.committed, *UPWARD_PANELS, "Figure_7.png"
        )
        plot_components(
            detailed, baseline.committed, *DOWNWARD_PANELS, "Figure_8.png"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
