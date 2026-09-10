"""Solve May 25 and then complete the annual simulation for a given scenario."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "2025_ups_1p25": ["--ups-capacity-multiplier", "1.25"],
    "2025_ups_1p5": ["--ups-capacity-multiplier", "1.5"],
    "2025_tes_1p25": ["--tes-capacity-multiplier", "1.25"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--time-limit", type=int, default=900)
    args = parser.parse_args()

    sc = args.scenario
    print(f"[{sc}] Step 1: Solving 2025-05-25 with time limit {args.time_limit}s...", flush=True)
    cmd1 = [
        sys.executable,
        "-m", "tasks.solve_and_splice_horizon",
        "--scenario", sc,
        "--date", "2025-05-25",
        "--time-limit", str(args.time_limit),
    ]
    subprocess.run(cmd1, cwd=ROOT, check=True)
    print(f"[{sc}] Step 1 complete! Spliced 2025-05-25.", flush=True)

    print(f"[{sc}] Step 2: Resuming through 2025-12-31...", flush=True)
    extra = CONFIGS[sc]
    cmd2 = [
        sys.executable,
        "-u",
        "-m", "rolling_optimisation.run_rolling_year",
        "--year", "2025",
        "--mode", "optimised",
        "--scenario-id", sc,
        "--output-root", "static/data/rolling_year_outputs",
        "--solver", "appsi_highs",
        "--solver-time-limit", "300",
        "--mip-gap", "0.001",
        "--maximum-accepted-gap", "0.01",
        "--fail-on-gap-exceeded",
        *extra,
    ]
    subprocess.run(cmd2, cwd=ROOT, check=True)
    print(f"[{sc}] Step 2 complete! Annual run finished.", flush=True)

if __name__ == "__main__":
    main()
