"""Rerun one annual horizon from its exact recorded opening state.

This is a solver-quality benchmark. It does not alter or splice the source
annual chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.model import solve_horizon
from rolling_optimisation.runner import combined_input_hash
from rolling_optimisation.timeline import (
    add_optimisation_prices,
    apply_flexible_workload_multiplier,
    build_annual_timeline,
    local_day_core_indices,
)
from rolling_optimisation.types import OperationalState, WorkloadCohort


ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "static/data/imrp_actuals.csv"
LOAD = ROOT / "static/data/inputs/load_profiles.csv"
SHIFTABILITY = ROOT / "static/data/inputs/shiftability_profile.csv"
ANNUAL_OUTPUTS = ROOT / "static/data/rolling_year_outputs"
DEFAULT_OUTPUT = ROOT / "reports/targeted_horizon_reruns/results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _maximum_state_difference(
    left: dict[str, float], right: dict[str, float]
) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scenario", required=True)
    parser.add_argument("--rerun-id", required=True)
    parser.add_argument("--date", default="2025-05-25")
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ANNUAL_OUTPUTS / args.source_scenario
    metadata_path = source_dir / "run_metadata.json"
    checkpoint_path = source_dir / "checkpoints" / f"{args.date}.json"
    source_day_path = source_dir / "days" / f"{args.date}.csv"
    if not metadata_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing source metadata/checkpoint for {args.source_scenario}"
        )
    if not source_day_path.exists():
        raise FileNotFoundError(f"Missing source committed day: {source_day_path}")

    output_dir = args.output_root.resolve() / args.rerun_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if metadata.get("finished_utc") is None:
        raise RuntimeError(f"Source annual scenario is not complete: {source_dir}")
    if checkpoint.get("date") != args.date:
        raise RuntimeError("Checkpoint date does not match requested date")

    input_paths = [PRICE, LOAD, SHIFTABILITY]
    current_input_hash = combined_input_hash(input_paths)
    code_paths = sorted((ROOT / "rolling_optimisation").glob("*.py"))
    current_code_hash = combined_input_hash(code_paths)
    if current_input_hash != metadata["input_hash"]:
        raise RuntimeError("Current model inputs do not match the source annual run")
    if current_code_hash != metadata["code_hash"]:
        raise RuntimeError("Current rolling model code does not match the source annual run")

    source_config = dict(metadata["config"])
    source_config["scenario_id"] = args.rerun_id
    source_config["solver_time_limit_s"] = args.time_limit
    source_config["fail_on_gap_exceeded"] = False
    config = RollingConfig(**source_config)

    timeline = build_annual_timeline(
        PRICE,
        LOAD,
        SHIFTABILITY,
        int(metadata["year"]),
        config.lookahead_steps,
        metadata["tail_price_mode"],
    )
    timeline = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    day_indices = dict(local_day_core_indices(timeline, int(metadata["year"])))
    indices = day_indices[args.date]
    start = int(indices[0])
    core_steps = len(indices)
    horizon = timeline.iloc[
        start : start + core_steps + config.lookahead_steps
    ].copy()
    if len(horizon) != core_steps + config.lookahead_steps:
        raise RuntimeError("Target horizon does not contain the full look-ahead")

    predecessor_date = (
        pd.Timestamp(args.date) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    predecessor_path = source_dir / "checkpoints" / f"{predecessor_date}.json"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if _sha256(predecessor_path) != checkpoint["predecessor_checkpoint_hash"]:
        raise RuntimeError("Target checkpoint predecessor hash is invalid")
    if predecessor["closing_state"] != checkpoint["opening_state"]:
        raise RuntimeError("Opening physical state does not match predecessor closing state")
    if predecessor["closing_workload"] != checkpoint["opening_workload"]:
        raise RuntimeError("Opening workload does not match predecessor closing workload")

    preflight = {
        "source_scenario": args.source_scenario,
        "rerun_id": args.rerun_id,
        "date": args.date,
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_predecessor_checkpoint_sha256": _sha256(predecessor_path),
        "input_hash": current_input_hash,
        "code_hash": current_code_hash,
        "core_steps": core_steps,
        "lookahead_steps": config.lookahead_steps,
        "opening_workload_cohorts": len(checkpoint["opening_workload"]),
        "config": config.to_dict(),
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return

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

    output_dir.mkdir(parents=True)
    result.committed.to_csv(output_dir / "committed.csv", index=False)
    source_day = pd.read_csv(source_day_path)
    payload = {
        **preflight,
        "source_solver": checkpoint["solver"],
        "rerun_solver": result.solver,
        "rerun_audits": result.audits,
        "source_committed_settlement_cost_gbp": float(
            source_day["settlement_cost_gbp"].sum()
        ),
        "rerun_committed_settlement_cost_gbp": result.audits[
            "committed_settlement_cost_gbp"
        ],
        "committed_settlement_cost_change_gbp": (
            result.audits["committed_settlement_cost_gbp"]
            - float(source_day["settlement_cost_gbp"].sum())
        ),
        "source_closing_state": checkpoint["closing_state"],
        "rerun_closing_state": result.next_state.to_dict(),
        "maximum_closing_state_difference": _maximum_state_difference(
            checkpoint["closing_state"], result.next_state.to_dict()
        ),
        "source_closing_workload_cpu_h": sum(
            float(item["remaining_cpu_hours"])
            for item in checkpoint["closing_workload"]
        ),
        "rerun_closing_workload_cpu_h": sum(
            cohort.remaining_cpu_hours for cohort in result.next_workload
        ),
    }
    _atomic_json(output_dir / "result.json", payload)
    print(
        f"{args.rerun_id}: {result.solver['solution_quality']}, "
        f"gap={result.solver['relative_gap']:.6%}, "
        f"runtime={result.solver['runtime_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
