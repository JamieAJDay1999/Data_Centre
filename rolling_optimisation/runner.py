from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RollingConfig, default_initial_state
from .model import solve_horizon
from .timeline import (
    add_optimisation_prices,
    build_annual_timeline,
    combined_input_hash,
    local_day_core_indices,
)
from .types import OperationalState, WorkloadCohort


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        frame.to_csv(temp_name, index=False, float_format="%.15g")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _boundary_residual(previous: pd.Series, current: pd.Series) -> float:
    keys = [
        "ups_energy_kwh",
        "tes_energy_kwh",
        "it_temperature_c",
        "rack_temperature_c",
        "cold_aisle_temperature_c",
        "hot_aisle_temperature_c",
    ]
    return max(
        abs(
            float(previous[f"state_end_{key}"])
            - float(current[f"state_start_{key}"])
        )
        for key in keys
    )


def _write_annual_outputs(
    run_dir: Path,
    frames: list[pd.DataFrame],
    checkpoint_rows: list[dict[str, Any]],
    config: RollingConfig,
    expected_steps: int,
    final_workload: list[WorkloadCohort],
) -> dict[str, Any]:
    committed = pd.concat(frames, ignore_index=True)
    if len(committed) != expected_steps:
        raise RuntimeError(
            f"Stitched output has {len(committed)} intervals; expected {expected_steps}"
        )
    timestamps = pd.to_datetime(committed["timestamp_utc"], utc=True)
    if timestamps.duplicated().any():
        raise RuntimeError("Duplicate committed UTC intervals")
    if len(timestamps) > 1 and not (
        timestamps.diff().dropna() == pd.Timedelta(seconds=config.dt_seconds)
    ).all():
        raise RuntimeError("Gap in committed UTC intervals")

    boundary_residual = 0.0
    for position in range(1, len(committed)):
        if committed.at[position, "local_date"] != committed.at[position - 1, "local_date"]:
            boundary_residual = max(
                boundary_residual,
                _boundary_residual(committed.iloc[position - 1], committed.iloc[position]),
            )

    annual_path = run_dir / "annual_committed.csv"
    _atomic_csv(annual_path, committed)

    daily = (
        committed.groupby("local_date", sort=False)
        .agg(
            intervals=("timestamp_utc", "size"),
            settlement_cost_gbp=("settlement_cost_gbp", "sum"),
            grid_energy_kwh=("grid_import_kw", lambda values: values.sum() * config.dt_hours),
            peak_grid_import_kw=("grid_import_kw", "max"),
            minimum_price_gbp_per_mwh=("settlement_price_gbp_per_mwh", "min"),
            maximum_price_gbp_per_mwh=("settlement_price_gbp_per_mwh", "max"),
        )
        .reset_index()
    )
    _atomic_csv(run_dir / "daily_summary.csv", daily)

    total_cost = float(committed["settlement_cost_gbp"].sum())
    direct_cost = float(
        (
            committed["grid_import_kw"]
            * committed["settlement_price_gbp_per_mwh"]
            / 1000.0
            * config.dt_hours
        ).sum()
    )
    summary = {
        "scenario_id": config.scenario_id,
        "mode": config.mode,
        "committed_intervals": len(committed),
        "local_dates": int(committed["local_date"].nunique()),
        "settlement_cost_gbp": total_cost,
        "grid_energy_kwh": float(committed["grid_import_kw"].sum() * config.dt_hours),
        "peak_grid_import_kw": float(committed["grid_import_kw"].max()),
        "cost_reconciliation_gbp": total_cost - direct_cost,
        "maximum_daily_boundary_residual": boundary_residual,
        "maximum_initial_state_residual": max(
            row["audits"]["initial_state_max_residual"] for row in checkpoint_rows
        ),
        "maximum_workload_conservation_residual_cpu_h": max(
            abs(row["audits"]["workload_conservation_residual_cpu_h"])
            for row in checkpoint_rows
        ),
        "maximum_ups_overlap_kw": max(
            row["audits"]["max_ups_charge_discharge_overlap_kw"]
            for row in checkpoint_rows
        ),
        "maximum_tes_overlap_kw": max(
            row["audits"]["max_tes_charge_discharge_overlap_kw"]
            for row in checkpoint_rows
        ),
        "non_optimal_horizons": sum(
            row["solver"]["termination_condition"] != "optimal"
            for row in checkpoint_rows
        ),
        "gap_exceeded_horizons": sum(
            not row["solver"].get("meets_accepted_gap", False)
            for row in checkpoint_rows
        ),
        "gap_exceeded_dates": [
            row["date"]
            for row in checkpoint_rows
            if not row["solver"].get("meets_accepted_gap", False)
        ],
        "maximum_recorded_solver_gap": max(
            (
                row["solver"]["relative_gap"]
                for row in checkpoint_rows
                if row["solver"].get("relative_gap") is not None
            ),
            default=None,
        ),
        "total_solver_runtime_s": sum(
            float(row["solver"]["runtime_s"]) for row in checkpoint_rows
        ),
        "final_outstanding_workload_cpu_h": sum(
            cohort.remaining_cpu_hours for cohort in final_workload
        ),
        "final_workload_unserved_after_planned_lookahead_cpu_h": float(
            checkpoint_rows[-1]["audits"][
                "core_workload_unserved_after_lookahead_cpu_h"
            ]
        ),
    }
    _atomic_json(run_dir / "annual_summary.json", summary)
    return summary


def run_rolling_scenario(
    *,
    root: Path,
    config: RollingConfig,
    year: int,
    price_path: Path,
    load_profile_path: Path,
    shiftability_path: Path,
    output_root: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    tail_price_mode: str = "actual",
    tee: bool = False,
) -> dict[str, Any]:
    """Run linked local-day horizons sequentially, resuming validated checkpoints."""

    input_paths = [price_path, load_profile_path, shiftability_path]
    input_hash = combined_input_hash(input_paths)
    code_paths = sorted((root / "rolling_optimisation").glob("*.py")) + [
        root / "run_rolling_year.py"
    ]
    code_hash = combined_input_hash(code_paths)
    fingerprint = config.fingerprint(input_hash, code_hash)
    timeline = build_annual_timeline(
        price_path,
        load_profile_path,
        shiftability_path,
        year,
        config.lookahead_steps,
        tail_price_mode,
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    days = local_day_core_indices(timeline, year)
    if start_date:
        days = [item for item in days if item[0] >= start_date]
    if end_date:
        days = [item for item in days if item[0] <= end_date]
    if not days:
        raise ValueError("No local dates selected")

    run_dir = output_root / config.scenario_id
    day_dir = run_dir / "days"
    checkpoint_dir = run_dir / "checkpoints"
    day_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "run_metadata.json"
    requested = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "input_hash": input_hash,
        "code_hash": code_hash,
        "config": config.to_dict(),
        "year": year,
        "start_date": days[0][0],
        "end_date": days[-1][0],
        "tail_price_mode": tail_price_mode,
        "git": _git_metadata(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {
            key: existing.get(key)
            for key in (
                "schema_version",
                "fingerprint",
                "input_hash",
                "code_hash",
                "config",
                "year",
                "start_date",
                "end_date",
                "tail_price_mode",
            )
        }
        expected = {key: requested[key] for key in comparable}
        if comparable != expected:
            raise RuntimeError(
                f"{run_dir} contains checkpoints for a different run configuration"
            )
    else:
        _atomic_json(metadata_path, requested)

    state = default_initial_state(config)
    workload: list[WorkloadCohort] = []
    previous_checkpoint_hash = "ROOT"
    frames: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for position, (date, indices) in enumerate(days, start=1):
        csv_path = day_dir / f"{date}.csv"
        checkpoint_path = checkpoint_dir / f"{date}.json"
        if csv_path.exists() != checkpoint_path.exists():
            raise RuntimeError(f"Incomplete checkpoint pair for {date}")

        if checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Fingerprint mismatch in checkpoint {checkpoint_path}")
            if payload.get("predecessor_checkpoint_hash") != previous_checkpoint_hash:
                raise RuntimeError(f"Broken predecessor chain at {checkpoint_path}")
            if payload.get("committed_csv_sha256") != _sha256(csv_path):
                raise RuntimeError(f"Committed CSV hash mismatch for {date}")
            if OperationalState.from_dict(payload["opening_state"]) != state:
                raise RuntimeError(f"Opening physical state mismatch at {date}")
            if [WorkloadCohort.from_dict(row) for row in payload["opening_workload"]] != workload:
                raise RuntimeError(f"Opening workload state mismatch at {date}")
            frame = pd.read_csv(csv_path)
            state = OperationalState.from_dict(payload["closing_state"])
            workload = [
                WorkloadCohort.from_dict(row) for row in payload["closing_workload"]
            ]
            previous_checkpoint_hash = _sha256(checkpoint_path)
            frames.append(frame)
            checkpoint_rows.append(payload)
            print(f"[{position}/{len(days)}] {date}: resumed")
            continue

        start = int(indices[0])
        core_steps = len(indices)
        horizon = timeline.iloc[
            start : start + core_steps + config.lookahead_steps
        ].copy()
        if len(horizon) != core_steps + config.lookahead_steps:
            raise RuntimeError(f"Insufficient look-ahead for {date}")
        opening_state = state
        opening_workload = list(workload)
        result = solve_horizon(
            config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
            tee=tee,
        )
        _atomic_csv(csv_path, result.committed)
        payload = {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "date": date,
            "core_steps": core_steps,
            "predecessor_checkpoint_hash": previous_checkpoint_hash,
            "opening_state": opening_state.to_dict(),
            "closing_state": result.next_state.to_dict(),
            "opening_workload": [row.to_dict() for row in opening_workload],
            "closing_workload": [row.to_dict() for row in result.next_workload],
            "solver": result.solver,
            "audits": result.audits,
            "committed_csv_sha256": _sha256(csv_path),
        }
        _atomic_json(checkpoint_path, payload)
        state = result.next_state
        workload = result.next_workload
        previous_checkpoint_hash = _sha256(checkpoint_path)
        frames.append(result.committed)
        checkpoint_rows.append(payload)
        quality = result.solver["solution_quality"]
        print(
            f"[{position}/{len(days)}] {date}: {quality} "
            f"({core_steps} intervals, {result.solver['runtime_s']:.2f}s, "
            f"{len(workload)} carried cohorts)"
        )

    expected_steps = sum(len(indices) for _, indices in days)
    summary = _write_annual_outputs(
        run_dir,
        frames,
        checkpoint_rows,
        config,
        expected_steps,
        workload,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["summary"] = summary
    _atomic_json(metadata_path, metadata)
    return summary
