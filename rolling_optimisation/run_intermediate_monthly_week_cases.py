"""Run the six 0.75x and 1.25x monthly-week sensitivity cases.

Completed cases are skipped and partial cases resume from their existing daily
checkpoints. The complete comparison is regenerated after all six finish.

Example:
    python -m rolling_optimisation.run_intermediate_monthly_week_cases --year 2025
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
INTERMEDIATE_CASES = (
    "ups_075",
    "ups_125",
    "tes_075",
    "tes_125",
    "flex_075",
    "flex_125",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    year_root = output_root / str(args.year)
    records: list[dict] = []
    batch_started = time.perf_counter()

    for position, case in enumerate(INTERMEDIATE_CASES, start=1):
        summary_path = year_root / case / "case_summary.json"
        if summary_path.exists():
            print(
                f"[{position}/{len(INTERMEDIATE_CASES)}] "
                f"{case}: already complete; skipped"
            )
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
        print(f"\n[{position}/{len(INTERMEDIATE_CASES)}] {case}: starting")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT)
        runtime = time.perf_counter() - started
        record = {
            "case": case,
            "status": "completed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "wall_runtime_s": runtime,
            "summary_path": str(summary_path),
        }
        records.append(record)
        if completed.returncode != 0:
            _write_batch_summary(year_root, records, batch_started)
            raise SystemExit(completed.returncode)
        if not summary_path.exists():
            raise RuntimeError(f"{case} exited successfully without {summary_path}")
        print(
            f"[{position}/{len(INTERMEDIATE_CASES)}] "
            f"{case}: completed in {runtime:.1f}s"
        )

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
    (year_root / "intermediate_cases_batch_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
