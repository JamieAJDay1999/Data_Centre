"""Run one independently selectable monthly-week sensitivity case.

One complete Monday-Sunday week is selected for every calendar month. Candidate
weeks are ranked by their standardised distance from the month's mean, standard
deviation, minimum, maximum, and negative-price share. Each selected week starts
from the case's default physical state; no warm-up days are solved.

Examples:
    python -m rolling_optimisation.run_monthly_week_sensitivity --select-only
    python -m rolling_optimisation.run_monthly_week_sensitivity --case baseline
    python -m rolling_optimisation.run_monthly_week_sensitivity --case central
    python -m rolling_optimisation.run_monthly_week_sensitivity --case ups_min
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from calendar import month_name
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rolling_optimisation.config import RollingConfig
from rolling_optimisation.runner import run_rolling_scenario
from rolling_optimisation.timeline import build_annual_timeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE = ROOT / "static" / "data" / "imrp_actuals.csv"
DEFAULT_LOAD = ROOT / "static" / "data" / "inputs" / "load_profiles.csv"
DEFAULT_SHIFTABILITY = (
    ROOT / "static" / "data" / "inputs" / "shiftability_profile.csv"
)
DEFAULT_OUTPUT = ROOT / "static" / "data" / "monthly_week_sensitivity"
SELECTION_METHOD = "monthly_standardised_price_distance_v3"
NEGATIVE_SHARE_TOLERANCE = 0.0025
MEAN_PRICE_TOLERANCE_GBP_PER_MWH = 0.5
PRICE_FEATURES = (
    "mean_price_gbp_per_mwh",
    "std_price_gbp_per_mwh",
    "minimum_price_gbp_per_mwh",
    "maximum_price_gbp_per_mwh",
    "negative_price_share",
)

CASE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "mode": "baseline",
        "description": "No workload or storage flexibility",
    },
    "central": {
        "mode": "optimised",
        "description": "Central optimised parameterisation",
    },
    "ups_min": {
        "mode": "optimised",
        "ups_capacity_multiplier": 0.5,
        "description": "UPS energy capacity at 0.5x",
    },
    "ups_075": {
        "mode": "optimised",
        "ups_capacity_multiplier": 0.75,
        "description": "UPS energy capacity at 0.75x",
    },
    "ups_125": {
        "mode": "optimised",
        "ups_capacity_multiplier": 1.25,
        "description": "UPS energy capacity at 1.25x",
    },
    "ups_max": {
        "mode": "optimised",
        "ups_capacity_multiplier": 1.5,
        "description": "UPS energy capacity at 1.5x",
    },
    "tes_min": {
        "mode": "optimised",
        "tes_capacity_multiplier": 0.5,
        "description": "TES energy capacity at 0.5x",
    },
    "tes_075": {
        "mode": "optimised",
        "tes_capacity_multiplier": 0.75,
        "description": "TES energy capacity at 0.75x",
    },
    "tes_125": {
        "mode": "optimised",
        "tes_capacity_multiplier": 1.25,
        "description": "TES energy capacity at 1.25x",
    },
    "tes_max": {
        "mode": "optimised",
        "tes_capacity_multiplier": 1.5,
        "description": "TES energy capacity at 1.5x",
    },
    "flex_min": {
        "mode": "optimised",
        "flexible_workload_multiplier": 0.5,
        "description": "Flexible workload share at 0.5x",
    },
    "flex_075": {
        "mode": "optimised",
        "flexible_workload_multiplier": 0.75,
        "description": "Flexible workload share at 0.75x",
    },
    "flex_125": {
        "mode": "optimised",
        "flexible_workload_multiplier": 1.25,
        "description": "Flexible workload share at 1.25x",
    },
    "flex_max": {
        "mode": "optimised",
        "flexible_workload_multiplier": 1.5,
        "description": "Flexible workload share at 1.5x",
    },
}


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


def _price_features(frame: pd.DataFrame) -> dict[str, float]:
    prices = frame["settlement_price_gbp_per_mwh"].astype(float)
    return {
        "mean_price_gbp_per_mwh": float(prices.mean()),
        "std_price_gbp_per_mwh": float(prices.std(ddof=0)),
        "minimum_price_gbp_per_mwh": float(prices.min()),
        "maximum_price_gbp_per_mwh": float(prices.max()),
        "negative_price_share": float((prices < 0).mean()),
    }


def select_representative_weeks(
    timeline: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected weeks and all auditable candidate scores."""

    target = timeline[timeline["is_target_year"]].copy()
    target["local_day"] = pd.to_datetime(target["local_date"])
    target["month"] = target["local_day"].dt.month

    candidate_rows: list[dict[str, Any]] = []
    monthly_features: dict[int, dict[str, float]] = {}
    monthly_intervals: dict[int, int] = {}
    for month in range(1, 13):
        month_frame = target[target["month"] == month]
        monthly_features[month] = _price_features(month_frame)
        monthly_intervals[month] = len(month_frame)
        first = pd.Timestamp(year=year, month=month, day=1)
        last = first + pd.offsets.MonthEnd(0)
        for start in pd.date_range(first, last, freq="D"):
            if start.weekday() != 0 or start + pd.Timedelta(days=6) > last:
                continue
            end = start + pd.Timedelta(days=6)
            week = target[
                (target["local_day"] >= start) & (target["local_day"] <= end)
            ]
            if week["local_date"].nunique() != 7:
                continue
            candidate_rows.append(
                {
                    "selection_method": SELECTION_METHOD,
                    "year": year,
                    "month": month,
                    "month_name": month_name[month],
                    "week_start": start.strftime("%Y-%m-%d"),
                    "week_end": end.strftime("%Y-%m-%d"),
                    "week_intervals": len(week),
                    **_price_features(week),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty or set(candidates["month"]) != set(range(1, 13)):
        raise RuntimeError("Could not construct complete Monday-Sunday candidates")

    feature_scales = candidates[list(PRICE_FEATURES)].std(ddof=0)
    feature_scales = feature_scales.mask(feature_scales <= 1e-12, 1.0)
    scores: list[float] = []
    for row in candidates.itertuples(index=False):
        month_target = monthly_features[int(row.month)]
        squared = [
            (
                (
                    float(getattr(row, feature))
                    - float(month_target[feature])
                )
                / float(feature_scales[feature])
            )
            ** 2
            for feature in PRICE_FEATURES
        ]
        scores.append(float(np.sqrt(np.mean(squared))))
    candidates["selection_score"] = scores

    # Selecting each month independently systematically favours zero-negative
    # weeks when negative prices occur in short clusters. Jointly select the
    # lowest-scoring combination that also represents annual negative exposure.
    total_intervals = sum(monthly_intervals.values())
    annual_negative_share = float(
        (target["settlement_price_gbp_per_mwh"] < 0).mean()
    )
    annual_mean_price = float(target["settlement_price_gbp_per_mwh"].mean())
    target_negative_intervals = annual_negative_share * total_intervals
    states: dict[
        tuple[int, int], tuple[float, list[int], float, float]
    ] = {(0, 0): (0.0, [], 0.0, 0.0)}
    for month in range(1, 13):
        next_states: dict[
            tuple[int, int], tuple[float, list[int], float, float]
        ] = {}
        for index, candidate in candidates[candidates["month"] == month].iterrows():
            negative_contribution = (
                float(candidate["negative_price_share"])
                * monthly_intervals[month]
            )
            mean_contribution = (
                float(candidate["mean_price_gbp_per_mwh"])
                * monthly_intervals[month]
                / total_intervals
            )
            for score, path, accumulated_negative, accumulated_mean in states.values():
                new_negative = accumulated_negative + negative_contribution
                new_mean = accumulated_mean + mean_contribution
                key = (int(round(new_negative)), int(round(new_mean * 10)))
                option = (
                    score + float(candidate["selection_score"]),
                    path + [int(index)],
                    new_negative,
                    new_mean,
                )
                incumbent = next_states.get(key)
                if incumbent is None or option[0] < incumbent[0]:
                    next_states[key] = option
        states = next_states

    acceptable = [
        state
        for state in states.values()
        if abs(state[2] - target_negative_intervals) / total_intervals
        <= NEGATIVE_SHARE_TOLERANCE
        and abs(state[3] - annual_mean_price)
        <= MEAN_PRICE_TOLERANCE_GBP_PER_MWH
    ]
    if acceptable:
        (
            total_score,
            selected_indices,
            selected_negative_intervals,
            selected_mean_price,
        ) = min(
            acceptable,
            key=lambda state: (
                state[0],
                abs(state[2] - target_negative_intervals),
                abs(state[3] - annual_mean_price),
            ),
        )
    else:
        (
            total_score,
            selected_indices,
            selected_negative_intervals,
            selected_mean_price,
        ) = min(
            states.values(),
            key=lambda state: (
                abs(state[2] - target_negative_intervals)
                / (NEGATIVE_SHARE_TOLERANCE * total_intervals)
                + abs(state[3] - annual_mean_price)
                / MEAN_PRICE_TOLERANCE_GBP_PER_MWH,
                state[0],
            ),
        )

    selected = candidates.loc[selected_indices].sort_values("month").reset_index(drop=True)
    selected["month_intervals"] = selected["month"].map(monthly_intervals)
    selected["annualisation_weight"] = (
        selected["month_intervals"] / selected["week_intervals"]
    )
    for feature in PRICE_FEATURES:
        selected[f"monthly_{feature}"] = selected["month"].map(
            {month: values[feature] for month, values in monthly_features.items()}
        )
    selected["selection_total_score"] = total_score
    selected["annual_negative_price_share"] = annual_negative_share
    selected["selected_weighted_negative_price_share"] = (
        selected_negative_intervals / total_intervals
    )
    selected["negative_share_tolerance"] = NEGATIVE_SHARE_TOLERANCE
    selected["annual_mean_price_gbp_per_mwh"] = annual_mean_price
    selected["selected_weighted_mean_price_gbp_per_mwh"] = selected_mean_price
    selected["mean_price_tolerance_gbp_per_mwh"] = (
        MEAN_PRICE_TOLERANCE_GBP_PER_MWH
    )
    return selected, candidates


def _selection_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        ["selection_method", "year", "month", "week_start", "week_end"]
    ].reset_index(drop=True)


def _write_or_validate_selection(
    year_root: Path,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
) -> Path:
    selected_path = year_root / "selected_weeks.csv"
    if selected_path.exists():
        existing = pd.read_csv(selected_path)
        if not _selection_key(existing).equals(_selection_key(selected)):
            raise RuntimeError(
                f"{selected_path} contains a different week selection. "
                "Use a different output root to preserve completed cases."
            )
    else:
        _atomic_csv(selected_path, selected)
    _atomic_csv(year_root / "week_selection_candidates.csv", candidates)
    return selected_path


def _case_config(args: argparse.Namespace, case: str) -> RollingConfig:
    definition = CASE_DEFINITIONS[case]
    return RollingConfig(
        scenario_id="placeholder",
        mode=str(definition["mode"]),
        solver_name=args.solver,
        solver_time_limit_s=args.solver_time_limit,
        mip_gap=args.mip_gap,
        maximum_accepted_gap=args.maximum_accepted_gap,
        fail_on_gap_exceeded=args.fail_on_gap_exceeded,
        it_power_segments=args.it_power_segments,
        it_power_representation=args.it_power_representation,
        it_power_breakpoint_exponent=args.it_power_breakpoint_exponent,
        ups_capacity_multiplier=float(
            definition.get("ups_capacity_multiplier", 1.0)
        ),
        tes_capacity_multiplier=float(
            definition.get("tes_capacity_multiplier", 1.0)
        ),
        flexible_workload_multiplier=float(
            definition.get("flexible_workload_multiplier", 1.0)
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--case", choices=sorted(CASE_DEFINITIONS))
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="select and record the twelve weeks without solving",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected weeks and case configuration without solving",
    )
    parser.add_argument("--price-input", type=Path, default=DEFAULT_PRICE)
    parser.add_argument("--load-input", type=Path, default=DEFAULT_LOAD)
    parser.add_argument("--shiftability-input", type=Path, default=DEFAULT_SHIFTABILITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--solver-time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--maximum-accepted-gap", type=float, default=0.01)
    parser.add_argument("--fail-on-gap-exceeded", action="store_true")
    parser.add_argument("--it-power-segments", type=int, default=4)
    parser.add_argument(
        "--it-power-representation",
        choices=["CUSTOM", "DLOG", "LOG", "INC", "CC"],
        default="DLOG",
    )
    parser.add_argument("--it-power-breakpoint-exponent", type=float, default=1.5)
    parser.add_argument("--tail-price-mode", choices=["actual", "repeat_last"], default="actual")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()
    if not args.select_only and args.case is None:
        parser.error("--case is required unless --select-only is used")
    return args


def main() -> None:
    args = parse_args()
    price_path = args.price_input.resolve()
    load_path = args.load_input.resolve()
    shiftability_path = args.shiftability_input.resolve()
    output_root = args.output_root.resolve()
    year_root = output_root / str(args.year)

    timeline = build_annual_timeline(
        price_path,
        load_path,
        shiftability_path,
        args.year,
        lookahead_steps=12,
        tail_price_mode=args.tail_price_mode,
    )
    selected, candidates = select_representative_weeks(timeline, args.year)
    selected_path = _write_or_validate_selection(year_root, selected, candidates)
    print(f"Selected weeks recorded in {selected_path}")
    print(
        selected[
            ["month_name", "week_start", "week_end", "selection_score"]
        ].to_string(index=False)
    )
    print(
        "\nNegative-price share: "
        f"selected annualised="
        f"{selected['selected_weighted_negative_price_share'].iloc[0]:.3%}, "
        f"full year={selected['annual_negative_price_share'].iloc[0]:.3%}"
    )
    print(
        "Mean settlement price: "
        f"selected annualised="
        f"GBP {selected['selected_weighted_mean_price_gbp_per_mwh'].iloc[0]:.2f}/MWh, "
        f"full year=GBP {selected['annual_mean_price_gbp_per_mwh'].iloc[0]:.2f}/MWh"
    )
    if args.select_only:
        return

    case = str(args.case)
    template = _case_config(args, case)
    print(f"\nCase: {case} - {CASE_DEFINITIONS[case]['description']}")
    print(
        "Multipliers: "
        f"UPS={template.ups_capacity_multiplier:g}, "
        f"TES={template.tes_capacity_multiplier:g}, "
        f"flexible workload={template.flexible_workload_multiplier:g}"
    )
    print("Warm-up days: 0 (each monthly week starts from its default state)")
    if args.dry_run:
        return

    case_root = year_root / case
    monthly_rows: list[dict[str, Any]] = []
    for position, week in enumerate(selected.itertuples(index=False), start=1):
        scenario_id = f"{int(week.month):02d}_{week.week_start}"
        config_values = template.to_dict()
        config_values["scenario_id"] = scenario_id
        config = RollingConfig(**config_values)
        print(
            f"\n[{position}/12] {week.month_name}: "
            f"{week.week_start} to {week.week_end}"
        )
        summary = run_rolling_scenario(
            root=ROOT,
            config=config,
            year=args.year,
            price_path=price_path,
            load_profile_path=load_path,
            shiftability_path=shiftability_path,
            output_root=case_root / "blocks",
            start_date=week.week_start,
            end_date=week.week_end,
            tail_price_mode=args.tail_price_mode,
            tee=args.tee,
        )
        weight = float(week.annualisation_weight)
        monthly_rows.append(
            {
                "case": case,
                "month": int(week.month),
                "month_name": week.month_name,
                "week_start": week.week_start,
                "week_end": week.week_end,
                "week_intervals": int(week.week_intervals),
                "month_intervals": int(week.month_intervals),
                "annualisation_weight": weight,
                "sample_settlement_cost_gbp": summary["settlement_cost_gbp"],
                "weighted_settlement_cost_gbp": (
                    summary["settlement_cost_gbp"] * weight
                ),
                "sample_grid_energy_kwh": summary["grid_energy_kwh"],
                "weighted_grid_energy_kwh": summary["grid_energy_kwh"] * weight,
                "sample_peak_grid_import_kw": summary["peak_grid_import_kw"],
                "non_optimal_horizons": summary["non_optimal_horizons"],
                "gap_exceeded_horizons": summary["gap_exceeded_horizons"],
                "maximum_recorded_solver_gap": summary[
                    "maximum_recorded_solver_gap"
                ],
                "solver_runtime_s": summary["total_solver_runtime_s"],
            }
        )

    monthly = pd.DataFrame(monthly_rows)
    _atomic_csv(case_root / "monthly_results.csv", monthly)
    recorded_gaps = monthly["maximum_recorded_solver_gap"].dropna()
    case_summary = {
        "case": case,
        "description": CASE_DEFINITIONS[case]["description"],
        "year": args.year,
        "selection_method": SELECTION_METHOD,
        "selected_weeks_file": str(selected_path),
        "selected_weeks_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "warm_up_days": 0,
        "scored_days": int(len(selected) * 7),
        "simulated_days": int(len(selected) * 7),
        "config": template.to_dict(),
        "estimated_annual_settlement_cost_gbp": float(
            monthly["weighted_settlement_cost_gbp"].sum()
        ),
        "estimated_annual_grid_energy_kwh": float(
            monthly["weighted_grid_energy_kwh"].sum()
        ),
        "sample_peak_grid_import_kw": float(
            monthly["sample_peak_grid_import_kw"].max()
        ),
        "non_optimal_horizons": int(monthly["non_optimal_horizons"].sum()),
        "gap_exceeded_horizons": int(monthly["gap_exceeded_horizons"].sum()),
        "maximum_recorded_solver_gap": (
            float(recorded_gaps.max()) if not recorded_gaps.empty else None
        ),
        "total_solver_runtime_s": float(monthly["solver_runtime_s"].sum()),
    }
    _atomic_json(case_root / "case_summary.json", case_summary)
    print("\nMonthly-week sensitivity case complete")
    for key, value in case_summary.items():
        if key not in {"config"}:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
