"""Regenerate annual and flexibility results after correcting tranche timing.

Annual scenarios are independent and run concurrently.  The central and
baseline cases are submitted first.  Once every annual case completes, the
paper result tables select the new representative day and the independent
flexibility cells run concurrently through the existing sweep driver.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "static/data/rolling_year_outputs"
LOGS = ROOT / "reports/corrected_tranche_rerun/logs"

SCENARIOS = (
    ("2025_optimised_cohort_trace", "optimised", []),
    ("2025_baseline_reformulated", "baseline", []),
    ("2025_flex_0p5", "optimised", ["--flexible-workload-multiplier", "0.5"]),
    ("2025_flex_0p75", "optimised", ["--flexible-workload-multiplier", "0.75"]),
    ("2025_flex_1p25", "optimised", ["--flexible-workload-multiplier", "1.25"]),
    ("2025_flex_1p5", "optimised", ["--flexible-workload-multiplier", "1.5"]),
    ("2025_ups_0p5", "optimised", ["--ups-capacity-multiplier", "0.5"]),
    ("2025_ups_0p75", "optimised", ["--ups-capacity-multiplier", "0.75"]),
    ("2025_ups_1p25", "optimised", ["--ups-capacity-multiplier", "1.25"]),
    ("2025_ups_1p5", "optimised", ["--ups-capacity-multiplier", "1.5"]),
    ("2025_tes_0p5", "optimised", ["--tes-capacity-multiplier", "0.5"]),
    ("2025_tes_0p75", "optimised", ["--tes-capacity-multiplier", "0.75"]),
    ("2025_tes_1p25", "optimised", ["--tes-capacity-multiplier", "1.25"]),
    ("2025_tes_1p5", "optimised", ["--tes-capacity-multiplier", "1.5"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-workers", type=int, default=8)
    parser.add_argument("--flexibility-workers", type=int, default=8)
    parser.add_argument("--solver-time-limit", type=int, default=300)
    parser.add_argument("--flexibility-time-limit", type=int, default=60)
    parser.add_argument("--skip-flexibility", action="store_true")
    return parser.parse_args()


def run_scenario(item: tuple[str, str, list[str]], solver_time_limit: int) -> dict:
    scenario, mode, extra = item
    command = [
        sys.executable,
        "-m",
        "rolling_optimisation.run_rolling_year",
        "--year",
        "2025",
        "--mode",
        mode,
        "--scenario-id",
        scenario,
        "--output-root",
        str(OUTPUT),
        "--solver",
        "appsi_highs",
        "--solver-time-limit",
        str(solver_time_limit),
        "--mip-gap",
        "0.001",
        "--maximum-accepted-gap",
        "0.01",
        "--fail-on-gap-exceeded",
        *extra,
    ]
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{scenario}.log"
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{scenario} failed; see {log_path}")
    summary_path = OUTPUT / scenario / "annual_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"{scenario} completed without {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "scenario": scenario,
        "wall_time_s": elapsed,
        "settlement_cost_gbp": summary["settlement_cost_gbp"],
        "maximum_recorded_solver_gap": summary["maximum_recorded_solver_gap"],
        "gap_exceeded_horizons": summary["gap_exceeded_horizons"],
    }


def checked_run(command: list[str], label: str, log_name: str) -> None:
    log_path = LOGS / log_name
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {log_path}")


def main() -> None:
    args = parse_args()
    if args.annual_workers < 1 or args.flexibility_workers < 1:
        raise ValueError("Worker counts must be positive")
    LOGS.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    records: list[dict] = []
    print(f"Starting {len(SCENARIOS)} annual scenarios with {args.annual_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.annual_workers) as executor:
        futures = {
            executor.submit(run_scenario, scenario, args.solver_time_limit): scenario[0]
            for scenario in SCENARIOS
        }
        for position, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            result = future.result()
            records.append(result)
            print(
                f"ANNUAL [{position}/{len(SCENARIOS)}] {name} completed in "
                f"{result['wall_time_s'] / 60:.1f} min",
                flush=True,
            )

    checked_run(
        [sys.executable, "paper/generate_annual_results.py"],
        "annual result generation",
        "generate_annual_results.log",
    )
    print("Annual tables, figures and representative-day selection regenerated", flush=True)

    if not args.skip_flexibility:
        checked_run(
            [
                sys.executable,
                "-m",
                "rolling_optimisation.run_parallel_flexibility_sweep",
                "--workers",
                str(args.flexibility_workers),
                "--time-limit",
                str(args.flexibility_time_limit),
            ],
            "parallel flexibility sweep",
            "flexibility_sweep.log",
        )
        print("Representative-day flexibility sweep regenerated", flush=True)

    payload = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": time.perf_counter() - run_started,
        "annual_workers": args.annual_workers,
        "flexibility_workers": args.flexibility_workers,
        "annual_scenarios": sorted(records, key=lambda row: row["scenario"]),
        "flexibility_completed": not args.skip_flexibility,
    }
    summary_path = ROOT / "reports/corrected_tranche_rerun/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Corrected rerun complete in {payload['wall_time_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
