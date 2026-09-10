"""Solve one blocked annual horizon with a longer limit and splice it safely.

The physical/model configuration and annual fingerprint remain unchanged. Only
the solver's computational time allowance is increased. The replacement is
accepted only when its reported optimality gap meets the annual run's recorded
acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.model import solve_horizon
from rolling_optimisation.runner import _atomic_csv, _atomic_json, _sha256
from rolling_optimisation.timeline import (
    add_optimisation_prices,
    apply_flexible_workload_multiplier,
    build_annual_timeline,
    combined_input_hash,
    local_day_core_indices,
)
from rolling_optimisation.types import OperationalState, WorkloadCohort


ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "static/data/imrp_actuals.csv"
LOAD = ROOT / "static/data/inputs/load_profiles.csv"
SHIFT = ROOT / "static/data/inputs/shiftability_profile.csv"
ANNUAL = ROOT / "static/data/rolling_year_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--time-limit", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ANNUAL / args.scenario
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    date = pd.Timestamp(args.date)
    predecessor_date = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    predecessor_path = run_dir / "checkpoints" / f"{predecessor_date}.json"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoints" / f"{args.date}.json"
    day_path = run_dir / "days" / f"{args.date}.csv"
    trace_path = run_dir / "workload_traces" / f"{args.date}.csv"
    if checkpoint_path.exists() or day_path.exists() or trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing {args.date} checkpoint")

    input_hash = combined_input_hash([PRICE, LOAD, SHIFT])
    code_hash = combined_input_hash(sorted((ROOT / "rolling_optimisation").glob("*.py")))
    if input_hash != metadata["input_hash"] or code_hash != metadata["code_hash"]:
        raise RuntimeError("Current inputs/code do not match the annual metadata")

    config_values = dict(metadata["config"])
    config_values["solver_name"] = "appsi_highs"
    config_values["solver_time_limit_s"] = args.time_limit
    config_values["fail_on_gap_exceeded"] = False
    config = RollingConfig(**config_values)
    timeline = build_annual_timeline(
        PRICE,
        LOAD,
        SHIFT,
        int(metadata["year"]),
        config.lookahead_steps,
        metadata["tail_price_mode"],
    )
    timeline = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    indices = dict(local_day_core_indices(timeline, int(metadata["year"])))[args.date]
    start = int(indices[0])
    core_steps = len(indices)
    horizon = timeline.iloc[start : start + core_steps + config.lookahead_steps].copy()
    opening_state = OperationalState.from_dict(predecessor["closing_state"])
    opening_workload = [
        WorkloadCohort.from_dict(row) for row in predecessor["closing_workload"]
    ]
    result = solve_horizon(
        config,
        horizon,
        core_steps,
        opening_state,
        opening_workload,
    )
    gap = result.solver.get("relative_gap")
    accepted_gap = float(metadata["config"]["maximum_accepted_gap"])
    if gap is None:
        raise RuntimeError("Long solve did not report an optimality gap")

    result.solver["solver_time_limit_s_used"] = args.time_limit
    _atomic_csv(day_path, result.committed)
    _atomic_csv(trace_path, result.workload_trace)
    payload = {
        "schema_version": 3,
        "fingerprint": metadata["fingerprint"],
        "date": args.date,
        "core_steps": core_steps,
        "predecessor_checkpoint_hash": _sha256(predecessor_path),
        "opening_state": opening_state.to_dict(),
        "closing_state": result.next_state.to_dict(),
        "opening_workload": [row.to_dict() for row in opening_workload],
        "closing_workload": [row.to_dict() for row in result.next_workload],
        "solver": result.solver,
        "audits": result.audits,
        "committed_csv_sha256": _sha256(day_path),
        "workload_trace_csv_sha256": _sha256(trace_path),
    }
    _atomic_json(checkpoint_path, payload)
    overrides = list(metadata.get("solver_time_limit_overrides", []))
    overrides.append(
        {
            "date": args.date,
            "time_limit_s": args.time_limit,
            "relative_gap": float(gap),
            "meets_accepted_gap": float(gap) <= accepted_gap,
            "reason": "Blocked horizon did not meet the recorded acceptance gap within 300 s",
        }
    )
    metadata["solver_time_limit_overrides"] = overrides
    _atomic_json(metadata_path, metadata)
    print(
        f"{args.scenario} {args.date}: spliced feasible gap {float(gap):.6%} "
        f"after {result.solver['runtime_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
