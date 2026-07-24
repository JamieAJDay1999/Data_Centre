"""Run the five outstanding monthly-week endpoint cases sequentially.

Completed cases are skipped. Partially completed cases resume through the
underlying daily checkpoints.

Example:
    python -m rolling_optimisation.run_remaining_monthly_week_cases --year 2025
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .run_monthly_week_sensitivity import DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ("central", "ups_min", "ups_max", "tes_min", "tes_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=DEFAULT_CASES,
        default=list(DEFAULT_CASES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    year_root = output_root / str(args.year)
    records: list[dict] = []
    batch_started = time.perf_counter()

    for position, case in enumerate(args.cases, start=1):
        summary_path = year_root / case / "case_summary.json"
        if summary_path.exists():
            print(f"[{position}/{len(args.cases)}] {case}: already complete; skipped")
            records.append(
                {
                    "case": case,
                    "status": "skipped_complete",
                    "wall_runtime_s": 0.0,
                    "summary_path": str(summary_path),
                }
            )
            continue

        command = [
            sys.executable,
            "-m",
            "rolling_optimisation.run_monthly_week_sensitivity",
            "--year",
            str(args.year),
            "--case",
            case,
            "--output-root",
            str(output_root),
        ]
        print(
            f"\n[{position}/{len(args.cases)}] {case}: starting "
            f"with {Path(sys.executable).name}"
        )
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT)
        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            records.append(
                {
                    "case": case,
                    "status": "failed",
                    "return_code": completed.returncode,
                    "wall_runtime_s": runtime,
                    "summary_path": str(summary_path),
                }
            )
            _write_batch_summary(year_root, records, batch_started)
            raise SystemExit(completed.returncode)
        if not summary_path.exists():
            raise RuntimeError(f"{case} exited successfully without {summary_path}")
        records.append(
            {
                "case": case,
                "status": "completed",
                "return_code": completed.returncode,
                "wall_runtime_s": runtime,
                "summary_path": str(summary_path),
            }
        )
        print(f"[{position}/{len(args.cases)}] {case}: completed in {runtime:.1f}s")

    _write_batch_summary(year_root, records, batch_started)
    comparison = subprocess.run(
        [
            sys.executable,
            "-m",
            "rolling_optimisation.summarise_monthly_week_sensitivity",
            "--year",
            str(args.year),
            "--output-root",
            str(output_root),
            "--require-all",
        ],
        cwd=ROOT,
    )
    if comparison.returncode != 0:
        raise SystemExit(comparison.returncode)


def _write_batch_summary(
    year_root: Path,
    records: list[dict],
    batch_started: float,
) -> None:
    year_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_runtime_s": time.perf_counter() - batch_started,
        "cases": records,
    }
    (year_root / "remaining_cases_batch_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
