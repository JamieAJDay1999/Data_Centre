"""Persist an extended-time verification of one flexibility-grid boundary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rolling_optimisation.run_representative_day_flexibility import (
    OUTPUT,
    REPORT,
    _cell_id,
    _prepare,
    _solve_request,
    _validate_flexibility_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--magnitude", type=float, required=True)
    parser.add_argument("--duration-steps", type=int, required=True)
    parser.add_argument("--time-limit", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, horizon, core, state, workload, baseline = _prepare(
        args.date, args.time_limit
    )
    trials: list[dict] = []
    solved = None
    for duration in (args.duration_steps, args.duration_steps + 1):
        started = time.perf_counter()
        result = _solve_request(
            config,
            horizon,
            core,
            state,
            workload,
            baseline,
            args.start_step,
            duration,
            args.magnitude,
        )
        elapsed = time.perf_counter() - started
        trials.append(
            {
                "duration_steps": duration,
                "feasible_incumbent": result is not None,
                "wall_time_s": elapsed,
                "solver_runtime_s": (
                    None if result is None else float(result.solver["runtime_s"])
                ),
                "solver_quality": (
                    "proven_infeasible"
                    if result is None
                    else str(result.solver["solution_quality"])
                ),
                "solver_relative_gap": (
                    None if result is None else result.solver["relative_gap"]
                ),
                "time_limit_s": args.time_limit,
            }
        )
        if duration == args.duration_steps:
            if result is None:
                raise RuntimeError("Claimed feasible duration did not solve")
            solved = result
        elif result is not None:
            raise RuntimeError("Claimed infeasible boundary returned a solution")
    assert solved is not None
    validation = _validate_flexibility_result(
        solved,
        baseline,
        args.start_step,
        args.duration_steps,
        args.magnitude,
    )
    cell_id = _cell_id(args.start_step, args.magnitude)
    report_path = REPORT / f"{cell_id}.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    all_trials = [*payload.get("trials", []), *trials]
    gaps = [
        float(row["solver_relative_gap"])
        for row in all_trials
        if row.get("solver_relative_gap") is not None
    ]
    result_row = {
        **payload["result"],
        "duration_steps": args.duration_steps,
        "duration_hours": args.duration_steps / 4.0,
        "solver_calls": len(all_trials),
        "solver_runtime_s": sum(float(row["wall_time_s"]) for row in all_trials),
        "maximum_solver_relative_gap": max(gaps, default=None),
        **validation,
        "boundary_verified": True,
    }
    solved.planned.attrs["flexibility_duration_steps"] = args.duration_steps
    solved.planned.to_csv(OUTPUT / f"{args.date}_{cell_id}.csv", index=False)
    payload.update(
        {
            "result": result_row,
            "trials": sorted(all_trials, key=lambda row: int(row["duration_steps"])),
            "solver": solved.solver,
            "audits": solved.audits,
            "terminal_state": solved.terminal_state.to_dict(),
            "extended_boundary_verification": {
                "time_limit_s": args.time_limit,
                "feasible_duration_steps": args.duration_steps,
                "first_infeasible_duration_steps": args.duration_steps + 1,
            },
        }
    )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    results_path = REPORT / "flexibility_results.csv"
    results = pd.read_csv(results_path)
    mask = (results["start_step"] == args.start_step) & (
        results["magnitude_kw"] == args.magnitude
    )
    for key, value in result_row.items():
        if key in results.columns:
            results.loc[mask, key] = value
    results.to_csv(results_path, index=False)

    summary_path = REPORT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "solver_calls": int(results["solver_calls"].sum()),
            "solver_runtime_s": float(results["solver_runtime_s"].sum()),
            "maximum_time_limit_s_per_solve": args.time_limit,
            "extended_boundary_checks": 1,
            "all_cells_validated": bool(results["validated"].all()),
            "all_boundaries_verified": bool(results["boundary_verified"].all()),
            "maximum_target_error_kw": float(
                results["maximum_target_error_kw"].fillna(0).max()
            ),
            "maximum_recovery_temperature_error_c": float(
                results["maximum_recovery_temperature_error_c"].fillna(0).max()
            ),
            "maximum_component_closure_error_kw": float(
                results["maximum_component_closure_error_kw"].fillna(0).max()
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(payload["extended_boundary_verification"], indent=2))


if __name__ == "__main__":
    main()
