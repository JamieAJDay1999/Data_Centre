"""Benchmark the revised rolling MILP against checkpoints from the prior run.

The prior checkpoints provide identical opening physical states and workload
backlogs, so this is a formulation benchmark rather than a new rolling chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.model import solve_horizon
from rolling_optimisation.timeline import build_annual_timeline, local_day_core_indices
from rolling_optimisation.types import OperationalState, WorkloadCohort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATES = [
    "2025-04-05",
    "2025-05-25",
    "2025-06-22",
    "2025-06-23",
    "2025-08-05",
    "2025-09-07",
    "2025-10-04",
    "2025-10-05",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument(
        "--prior-run",
        type=Path,
        default=ROOT / "static/data/rolling_year_outputs/2025_optimised",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/rolling_formulation_benchmark.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = build_annual_timeline(
        ROOT / "static/data/imrp_actuals.csv",
        ROOT / "static/data/inputs/load_profiles.csv",
        ROOT / "static/data/inputs/shiftability_profile.csv",
        2025,
        12,
        "actual",
    )
    day_indices = dict(local_day_core_indices(timeline, 2025))
    rows: list[dict] = []
    config = RollingConfig(
        scenario_id="formulation_benchmark",
        solver_name="appsi_highs",
        solver_time_limit_s=args.time_limit,
    )

    for date in args.dates:
        checkpoint_path = args.prior_run / "checkpoints" / f"{date}.json"
        day_path = args.prior_run / "days" / f"{date}.csv"
        if not checkpoint_path.exists() or not day_path.exists():
            raise FileNotFoundError(
                f"Prior checkpoint pair is unavailable for {date}: {args.prior_run}"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        prior_day = pd.read_csv(day_path)
        indices = day_indices[date]
        start = int(indices[0])
        core_steps = len(indices)
        horizon = timeline.iloc[
            start : start + core_steps + config.lookahead_steps
        ].copy()
        result = solve_horizon(
            config,
            horizon,
            core_steps,
            OperationalState.from_dict(checkpoint["opening_state"]),
            [
                WorkloadCohort.from_dict(item)
                for item in checkpoint["opening_workload"]
            ],
        )
        rows.append(
            {
                "date": date,
                "negative_quarter_hours": int(
                    (horizon["settlement_price_gbp_per_mwh"] < 0).sum()
                ),
                "prior_runtime_s": checkpoint["solver"]["runtime_s"],
                "prior_termination": checkpoint["solver"]["termination_condition"],
                "prior_gap": checkpoint["solver"].get("relative_gap"),
                "revised_runtime_s": result.solver["runtime_s"],
                "revised_termination": result.solver["termination_condition"],
                "revised_gap": result.solver["relative_gap"],
                "revised_solution_quality": result.solver["solution_quality"],
                "prior_committed_cost_gbp": float(
                    prior_day["settlement_cost_gbp"].sum()
                ),
                "revised_committed_cost_gbp": result.audits[
                    "committed_settlement_cost_gbp"
                ],
                "revised_binary_variables": result.solver["binary_variables"],
                "maximum_it_power_approximation_error_kw": result.audits[
                    "maximum_it_power_approximation_error_kw"
                ],
            }
        )
        print(
            f"{date}: {checkpoint['solver']['runtime_s']:.2f}s -> "
            f"{result.solver['runtime_s']:.2f}s "
            f"(gap {result.solver['relative_gap']:.4%})"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
