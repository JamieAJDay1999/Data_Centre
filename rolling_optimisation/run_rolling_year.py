"""Run one continuous year as linked daily rolling-horizon optimisations.

Each local trading day is solved separately with a three-hour look-ahead. Only
that day's intervals are committed. Physical state and outstanding flexible
workload are passed into the following day.

Examples:
    python -m rolling_optimisation.run_rolling_year --validate-only
    python -m rolling_optimisation.run_rolling_year --start-date 2025-01-01 --end-date 2025-01-02
    python -m rolling_optimisation.run_rolling_year --mode baseline
    python -m rolling_optimisation.run_rolling_year --mode optimised
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.runner import run_rolling_scenario
from rolling_optimisation.timeline import build_annual_timeline, local_day_core_indices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE = ROOT / "static" / "data" / "imrp_actuals.csv"
DEFAULT_LOAD = ROOT / "static" / "data" / "inputs" / "load_profiles.csv"
DEFAULT_SHIFTABILITY = (
    ROOT / "static" / "data" / "inputs" / "shiftability_profile.csv"
)
DEFAULT_OUTPUT = ROOT / "static" / "data" / "rolling_year_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--mode", choices=["optimised", "baseline"], default="optimised")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--start-date", default=None, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="inclusive YYYY-MM-DD")
    parser.add_argument("--price-input", type=Path, default=DEFAULT_PRICE)
    parser.add_argument("--load-input", type=Path, default=DEFAULT_LOAD)
    parser.add_argument("--shiftability-input", type=Path, default=DEFAULT_SHIFTABILITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--solver", default="auto", help="auto, scip, or appsi_highs")
    parser.add_argument("--solver-time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument(
        "--maximum-accepted-gap",
        type=float,
        default=0.01,
        help="quality threshold reported separately from the requested solver gap",
    )
    parser.add_argument(
        "--fail-on-gap-exceeded",
        action="store_true",
        help="stop before checkpointing a horizon outside maximum-accepted-gap",
    )
    parser.add_argument(
        "--it-power-segments",
        type=int,
        default=4,
        help="segment count for the IT power curve",
    )
    parser.add_argument(
        "--it-power-representation",
        choices=["CUSTOM", "DLOG", "LOG", "INC", "CC"],
        default="DLOG",
    )
    parser.add_argument(
        "--it-power-breakpoint-exponent",
        type=float,
        default=1.5,
        help="greater than one concentrates breakpoints at low CPU utilisation",
    )
    parser.add_argument(
        "--price-treatment",
        choices=["signed", "floor_zero", "shift_year_min"],
        default="signed",
        help="signed is the central case; alternatives are sensitivity cases",
    )
    parser.add_argument("--grid-import-limit-kw", type=float, default=None)
    parser.add_argument("--ups-reserve-kwh", type=float, default=None)
    parser.add_argument(
        "--ups-throughput-cost-gbp-per-kwh", type=float, default=0.0
    )
    parser.add_argument(
        "--tes-throughput-cost-gbp-per-kwh-th", type=float, default=0.0
    )
    parser.add_argument(
        "--terminal-ups-value-gbp-per-kwh", type=float, default=0.0
    )
    parser.add_argument(
        "--terminal-tes-value-gbp-per-kwh-th", type=float, default=0.0
    )
    parser.add_argument(
        "--tail-price-mode",
        choices=["actual", "repeat_last"],
        default="actual",
        help="price assumption for the final three-hour look-ahead",
    )
    parser.add_argument("--tee", action="store_true", help="show solver output")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and summarise the timeline without solving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_id = args.scenario_id or f"{args.year}_{args.mode}"
    config = RollingConfig(
        scenario_id=scenario_id,
        mode=args.mode,
        solver_name=args.solver,
        solver_time_limit_s=args.solver_time_limit,
        mip_gap=args.mip_gap,
        maximum_accepted_gap=args.maximum_accepted_gap,
        fail_on_gap_exceeded=args.fail_on_gap_exceeded,
        it_power_segments=args.it_power_segments,
        it_power_representation=args.it_power_representation,
        it_power_breakpoint_exponent=args.it_power_breakpoint_exponent,
        price_treatment=args.price_treatment,
        grid_import_limit_kw=args.grid_import_limit_kw,
        ups_reserve_kwh=args.ups_reserve_kwh,
        ups_throughput_cost_gbp_per_kwh=args.ups_throughput_cost_gbp_per_kwh,
        tes_throughput_cost_gbp_per_kwh_th=args.tes_throughput_cost_gbp_per_kwh_th,
        terminal_ups_value_gbp_per_kwh=args.terminal_ups_value_gbp_per_kwh,
        terminal_tes_value_gbp_per_kwh_th=args.terminal_tes_value_gbp_per_kwh_th,
    )
    price_path = args.price_input.resolve()
    load_path = args.load_input.resolve()
    shiftability_path = args.shiftability_input.resolve()

    if args.validate_only:
        timeline = build_annual_timeline(
            price_path,
            load_path,
            shiftability_path,
            args.year,
            config.lookahead_steps,
            args.tail_price_mode,
        )
        target = timeline[timeline["is_target_year"]]
        days = local_day_core_indices(timeline, args.year)
        counts = target.groupby("local_date").size()
        print(f"Year: {args.year}")
        print(f"Committed quarter-hours: {len(target):,}")
        print(f"Local dates: {len(days)}")
        print(
            "Local-day interval counts: "
            + ", ".join(f"{int(count)} × {int((counts == count).sum())} days" for count in sorted(counts.unique()))
        )
        print(
            f"UTC range: {target['timestamp_utc'].iloc[0]} to "
            f"{target['timestamp_utc'].iloc[-1]}"
        )
        print("Timeline validation passed; no solver calls were made.")
        return

    summary = run_rolling_scenario(
        root=ROOT,
        config=config,
        year=args.year,
        price_path=price_path,
        load_profile_path=load_path,
        shiftability_path=shiftability_path,
        output_root=args.output_root.resolve(),
        start_date=args.start_date,
        end_date=args.end_date,
        tail_price_mode=args.tail_price_mode,
        tee=args.tee,
    )
    print("\nRolling run complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
