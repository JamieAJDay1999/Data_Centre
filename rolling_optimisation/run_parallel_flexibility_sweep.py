"""Solve independent representative-day flexibility cells concurrently.

Each worker invokes the existing single-cell command, which writes only its own
JSON/CSV pair.  The ordinary sweep command is run once after all workers finish
to validate compatibility, aggregate the table, and regenerate the figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rolling_optimisation.run_representative_day_flexibility import (
    MAGNITUDES_KW,
    METHOD_SCHEMA_VERSION,
    REPORT,
    START_STEPS,
    _cell_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def _compatible(start: int, magnitude: float, time_limit: int) -> bool:
    path = REPORT / f"{_cell_id(start, magnitude)}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        saved_limit = int(payload.get("time_limit_s", -1))
        exact = bool(result.get("boundary_verified", False))
        return bool(
            int(payload.get("method_schema_version", -1))
            == METHOD_SCHEMA_VERSION
            and int(payload.get("start_step", -1)) == start
            and float(payload.get("magnitude_kw", float("nan"))) == magnitude
            and (
                saved_limit == time_limit
                or (exact and 0 < saved_limit <= time_limit)
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _solve_cell(start: int, magnitude: float, time_limit: int) -> str:
    command = [
        sys.executable,
        "-m",
        "rolling_optimisation.run_representative_day_flexibility",
        "--time-limit",
        str(time_limit),
        "--cell-start-step",
        str(start),
        "--cell-magnitude-kw",
        str(magnitude),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cell start={start}, magnitude={magnitude:+g} failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "completed"


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    cells = [
        (start, float(magnitude))
        for start in START_STEPS
        for magnitude in MAGNITUDES_KW
    ]
    pending = (
        cells
        if args.restart
        else [
            cell
            for cell in cells
            if not _compatible(cell[0], cell[1], args.time_limit)
        ]
    )
    print(
        f"Fixed-recovery cells: {len(cells) - len(pending)} compatible, "
        f"{len(pending)} pending; workers={args.workers}"
    )
    started = time.perf_counter()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_solve_cell, start, magnitude, args.time_limit): (
                start,
                magnitude,
            )
            for start, magnitude in pending
        }
        for position, future in enumerate(as_completed(futures), start=1):
            start, magnitude = futures[future]
            try:
                future.result()
                print(
                    f"[{position}/{len(pending)}] start={start / 4:.2f}h, "
                    f"delta={magnitude:+.0f} kW"
                )
            except Exception as error:  # noqa: BLE001 - aggregate all worker failures
                failures.append(str(error))
                print(f"[{position}/{len(pending)}] FAILED: {error}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} flexibility workers failed; first failure:\n"
            + failures[0]
        )

    aggregate = subprocess.run(
        [
            sys.executable,
            "-m",
            "rolling_optimisation.run_representative_day_flexibility",
            "--time-limit",
            str(args.time_limit),
        ],
        check=False,
    )
    if aggregate.returncode != 0:
        raise RuntimeError("Final flexibility aggregation failed")
    elapsed = time.perf_counter() - started
    summary_path = REPORT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "parallel_workers": args.workers,
            "parallel_pending_cells": len(pending),
            "parallel_wall_time_s": elapsed,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Parallel sweep completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
