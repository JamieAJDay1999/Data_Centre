"""Rerun the final local day of every completed annual scenario.

The rerun derives its opening state and outstanding workload directly from the
30 December checkpoint, and requires the 12 look-ahead intervals to be the
first three hours of 1 January in the following year. Source annual outputs are
read-only; rerun artefacts and comparisons are written under ``reports``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
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
DEFAULT_OUTPUT = ROOT / "reports/annual_year_end_rerun"
NUMERIC_MATCH_TOLERANCE = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        frame.to_csv(temporary_name, index=False, float_format="%.15g")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _state_difference(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def _maximum_numeric_frame_difference(
    source: pd.DataFrame, rerun: pd.DataFrame
) -> float:
    if list(source.columns) != list(rerun.columns) or len(source) != len(rerun):
        return float("inf")
    differences: list[float] = []
    for column in source.columns:
        if not (
            pd.api.types.is_numeric_dtype(source[column])
            and pd.api.types.is_numeric_dtype(rerun[column])
        ):
            continue
        difference = (source[column] - rerun[column]).abs().max()
        differences.append(float(difference) if pd.notna(difference) else 0.0)
    return max(differences, default=0.0)


def _workload_total(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["remaining_cpu_hours"]) for row in rows)


def _scenario_names(year: int, requested: list[str]) -> list[str]:
    if requested:
        return requested
    names: list[str] = []
    for directory in sorted(ANNUAL_OUTPUTS.iterdir()):
        metadata_path = directory / "run_metadata.json"
        if not directory.is_dir() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("year", -1)) == year
            and metadata.get("finished_utc") is not None
            and metadata.get("start_date") == f"{year}-01-01"
            and metadata.get("end_date") == f"{year}-12-31"
        ):
            names.append(directory.name)
    return names


def _tail_audit(
    horizon: pd.DataFrame, core_steps: int, year: int
) -> dict[str, Any]:
    tail = horizon.iloc[core_steps:].copy()
    expected_date = f"{year + 1}-01-01"
    expected_periods = [1] * 4 + [2] * 4 + [3] * 4
    source_dates = tail["source_date"].astype(str).tolist()
    source_periods = tail["source_period"].astype(int).tolist()
    if source_dates != [expected_date] * 12:
        raise RuntimeError(
            f"Year-end look-ahead does not use only {expected_date}: {source_dates}"
        )
    if source_periods != expected_periods:
        raise RuntimeError(
            f"Year-end look-ahead periods are not 1, 2, 3: {source_periods}"
        )
    hourly = (
        tail.groupby(["source_date", "source_period"], sort=False)[
            "settlement_price_gbp_per_mwh"
        ]
        .first()
        .reset_index()
    )
    return {
        "source_date": expected_date,
        "source_periods": hourly["source_period"].astype(int).tolist(),
        "prices_gbp_per_mwh": hourly[
            "settlement_price_gbp_per_mwh"
        ].astype(float).tolist(),
        "quarter_hour_intervals": len(tail),
    }


def rerun_scenario(
    scenario: str,
    year: int,
    output_root: Path,
    input_hash: str,
    code_hash: str,
) -> dict[str, Any]:
    source_dir = ANNUAL_OUTPUTS / scenario
    metadata_path = source_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing annual metadata for {scenario}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("finished_utc") is None:
        raise RuntimeError(f"Annual scenario is incomplete: {scenario}")
    if metadata.get("tail_price_mode") != "actual":
        raise RuntimeError(f"{scenario} did not declare actual tail prices")
    if metadata.get("input_hash") != input_hash:
        raise RuntimeError(f"{scenario} used a different input snapshot")

    predecessor_date = f"{year}-12-30"
    final_date = f"{year}-12-31"
    predecessor_path = source_dir / "checkpoints" / f"{predecessor_date}.json"
    source_checkpoint_path = source_dir / "checkpoints" / f"{final_date}.json"
    source_day_path = source_dir / "days" / f"{final_date}.csv"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    source_checkpoint = json.loads(
        source_checkpoint_path.read_text(encoding="utf-8")
    )
    if _sha256(predecessor_path) != source_checkpoint[
        "predecessor_checkpoint_hash"
    ]:
        raise RuntimeError(f"{scenario} has an invalid 30-to-31 December chain")
    if predecessor["closing_state"] != source_checkpoint["opening_state"]:
        raise RuntimeError(f"{scenario} physical state was not carried into 31 December")
    if predecessor["closing_workload"] != source_checkpoint["opening_workload"]:
        raise RuntimeError(f"{scenario} workload was not carried into 31 December")

    config_values = dict(metadata["config"])
    config_values["fail_on_gap_exceeded"] = False
    config = RollingConfig(**config_values)
    timeline = build_annual_timeline(
        PRICE,
        LOAD,
        SHIFTABILITY,
        year,
        config.lookahead_steps,
        "actual",
    )
    timeline = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    indices = dict(local_day_core_indices(timeline, year))[final_date]
    start = int(indices[0])
    core_steps = len(indices)
    horizon = timeline.iloc[
        start : start + core_steps + config.lookahead_steps
    ].copy()
    if len(horizon) != core_steps + config.lookahead_steps:
        raise RuntimeError(f"{scenario} final horizon lacks the full look-ahead")
    tail = _tail_audit(horizon, core_steps, year)

    result = solve_horizon(
        config,
        horizon,
        core_steps,
        OperationalState.from_dict(predecessor["closing_state"]),
        [
            WorkloadCohort.from_dict(row)
            for row in predecessor["closing_workload"]
        ],
    )
    source_day = pd.read_csv(source_day_path)
    maximum_numeric_difference = _maximum_numeric_frame_difference(
        source_day, result.committed
    )
    source_cost = float(source_day["settlement_cost_gbp"].sum())
    cost_change = result.audits["committed_settlement_cost_gbp"] - source_cost
    closing_state_difference = _state_difference(
        source_checkpoint["closing_state"], result.next_state.to_dict()
    )
    source_closing_workload = _workload_total(
        source_checkpoint["closing_workload"]
    )
    rerun_closing_workload = sum(
        row.remaining_cpu_hours for row in result.next_workload
    )
    code_match = metadata.get("code_hash") == code_hash
    exact_reproduction = (
        maximum_numeric_difference <= NUMERIC_MATCH_TOLERANCE
        and abs(cost_change) <= NUMERIC_MATCH_TOLERANCE
        and closing_state_difference <= NUMERIC_MATCH_TOLERANCE
        and abs(rerun_closing_workload - source_closing_workload)
        <= NUMERIC_MATCH_TOLERANCE
    )
    payload = {
        "scenario": scenario,
        "year": year,
        "final_date": final_date,
        "rerun_utc": datetime.now(timezone.utc).isoformat(),
        "opening_source": predecessor_date,
        "opening_state": predecessor["closing_state"],
        "opening_workload_cpu_h": _workload_total(
            predecessor["closing_workload"]
        ),
        "tail": tail,
        "input_hash": input_hash,
        "source_code_hash": metadata.get("code_hash"),
        "rerun_code_hash": code_hash,
        "source_code_hash_matches_rerun": code_match,
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "predecessor_checkpoint_sha256": _sha256(predecessor_path),
        "source_solver": source_checkpoint["solver"],
        "rerun_solver": result.solver,
        "source_committed_settlement_cost_gbp": source_cost,
        "rerun_committed_settlement_cost_gbp": result.audits[
            "committed_settlement_cost_gbp"
        ],
        "committed_settlement_cost_change_gbp": cost_change,
        "maximum_committed_numeric_difference": maximum_numeric_difference,
        "maximum_closing_state_difference": closing_state_difference,
        "source_closing_workload_cpu_h": source_closing_workload,
        "rerun_closing_workload_cpu_h": rerun_closing_workload,
        "core_workload_unserved_after_lookahead_cpu_h": result.audits[
            "core_workload_unserved_after_lookahead_cpu_h"
        ],
        "exact_reproduction_at_tolerance": exact_reproduction,
        "numeric_match_tolerance": NUMERIC_MATCH_TOLERANCE,
    }
    scenario_output = output_root / scenario
    _atomic_csv(scenario_output / f"{final_date}.csv", result.committed)
    _atomic_json(scenario_output / "audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="scenario to rerun; repeat the option, or omit it for all complete annual runs",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    input_hash = combined_input_hash([PRICE, LOAD, SHIFTABILITY])
    code_hash = combined_input_hash(
        sorted((ROOT / "rolling_optimisation").glob("*.py"))
    )
    scenarios = _scenario_names(args.year, args.scenario)
    if not scenarios:
        raise RuntimeError("No completed annual scenarios were selected")

    records: list[dict[str, Any]] = []
    for position, scenario in enumerate(scenarios, start=1):
        payload = rerun_scenario(
            scenario,
            args.year,
            output_root,
            input_hash,
            code_hash,
        )
        records.append(payload)
        print(
            f"[{position}/{len(scenarios)}] {scenario}: "
            f"{payload['rerun_solver']['solution_quality']}, "
            f"cost change={payload['committed_settlement_cost_change_gbp']:.12g} GBP, "
            f"exact reproduction={payload['exact_reproduction_at_tolerance']}"
        )

    comparison_columns = [
        "scenario",
        "source_code_hash_matches_rerun",
        "source_committed_settlement_cost_gbp",
        "rerun_committed_settlement_cost_gbp",
        "committed_settlement_cost_change_gbp",
        "maximum_committed_numeric_difference",
        "maximum_closing_state_difference",
        "source_closing_workload_cpu_h",
        "rerun_closing_workload_cpu_h",
        "core_workload_unserved_after_lookahead_cpu_h",
        "exact_reproduction_at_tolerance",
    ]
    comparison = pd.DataFrame(records)[comparison_columns]
    _atomic_csv(output_root / "comparison.csv", comparison)
    _atomic_json(
        output_root / "summary.json",
        {
            "year": args.year,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_count": len(records),
            "input_hash": input_hash,
            "rerun_code_hash": code_hash,
            "tail": records[0]["tail"],
            "exact_reproduction_count": sum(
                bool(row["exact_reproduction_at_tolerance"]) for row in records
            ),
            "records": records,
        },
    )
    print(f"Comparison written to {output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
