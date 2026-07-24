"""Compare completed monthly-week sensitivity cases against the sampled baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_monthly_week_sensitivity import CASE_DEFINITIONS, DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parent
DEFAULT_FULL_YEAR_OUTPUT = ROOT / "static" / "data" / "rolling_year_outputs"
CASE_ORDER = [
    "central",
    "ups_min",
    "ups_075",
    "ups_125",
    "ups_max",
    "tes_min",
    "tes_075",
    "tes_125",
    "tes_max",
    "flex_min",
    "flex_075",
    "flex_125",
    "flex_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--full-year-output-root",
        type=Path,
        default=DEFAULT_FULL_YEAR_OUTPUT,
    )
    parser.add_argument(
        "--full-year-baseline-scenario",
        default="2025_baseline_reformulated",
    )
    parser.add_argument(
        "--full-year-optimised-scenario",
        default="2025_optimised_reformulated",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail unless baseline and all seven optimised cases are complete",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    year_root = args.output_root.resolve() / str(args.year)
    baseline_path = year_root / "baseline" / "case_summary.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Run the baseline case first; missing {baseline_path}"
        )
    baseline = _load_summary(baseline_path)
    baseline_cost = float(baseline["estimated_annual_settlement_cost_gbp"])
    baseline_energy = float(baseline["estimated_annual_grid_energy_kwh"])

    missing: list[str] = []
    rows: list[dict] = []
    for case in CASE_ORDER:
        path = year_root / case / "case_summary.json"
        if not path.exists():
            missing.append(case)
            continue
        summary = _load_summary(path)
        if summary["selected_weeks_sha256"] != baseline["selected_weeks_sha256"]:
            raise RuntimeError(f"{case} used a different representative-week selection")
        cost = float(summary["estimated_annual_settlement_cost_gbp"])
        energy = float(summary["estimated_annual_grid_energy_kwh"])
        saving = baseline_cost - cost
        rows.append(
            {
                "case": case,
                "description": CASE_DEFINITIONS[case]["description"],
                "ups_capacity_multiplier": summary["config"][
                    "ups_capacity_multiplier"
                ],
                "tes_capacity_multiplier": summary["config"][
                    "tes_capacity_multiplier"
                ],
                "flexible_workload_multiplier": summary["config"][
                    "flexible_workload_multiplier"
                ],
                "estimated_annual_cost_gbp": cost,
                "estimated_annual_saving_gbp": saving,
                "estimated_annual_saving_percent": (
                    100.0 * saving / baseline_cost
                ),
                "estimated_annual_grid_energy_kwh": energy,
                "grid_energy_change_from_baseline_kwh": energy - baseline_energy,
                "sample_peak_grid_import_kw": summary[
                    "sample_peak_grid_import_kw"
                ],
                "non_optimal_horizons": summary["non_optimal_horizons"],
                "gap_exceeded_horizons": summary["gap_exceeded_horizons"],
                "maximum_recorded_solver_gap": summary[
                    "maximum_recorded_solver_gap"
                ],
                "solver_runtime_s": summary["total_solver_runtime_s"],
            }
        )

    if args.require_all and missing:
        raise FileNotFoundError(
            "Missing completed sensitivity cases: " + ", ".join(missing)
        )
    if not rows:
        raise FileNotFoundError("No completed optimised sensitivity cases found")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        year_root / "sensitivity_comparison.csv",
        index=False,
        float_format="%.15g",
    )
    print(
        comparison[
            [
                "case",
                "estimated_annual_cost_gbp",
                "estimated_annual_saving_gbp",
                "estimated_annual_saving_percent",
                "solver_runtime_s",
            ]
        ].to_string(index=False)
    )
    if missing:
        print("\nNot yet completed: " + ", ".join(missing))
    print(f"\nComparison written to {year_root / 'sensitivity_comparison.csv'}")

    validation_rows: list[dict] = []
    full_year_root = args.full_year_output_root.resolve()
    full_baseline_path = (
        full_year_root
        / args.full_year_baseline_scenario
        / "annual_summary.json"
    )
    if full_baseline_path.exists():
        full_baseline = _load_summary(full_baseline_path)
        for metric, sampled_value, full_key in (
            (
                "baseline_settlement_cost_gbp",
                baseline_cost,
                "settlement_cost_gbp",
            ),
            (
                "baseline_grid_energy_kwh",
                baseline_energy,
                "grid_energy_kwh",
            ),
        ):
            full_value = float(full_baseline[full_key])
            validation_rows.append(
                {
                    "metric": metric,
                    "sampled_value": sampled_value,
                    "full_year_value": full_value,
                    "absolute_error": sampled_value - full_value,
                    "relative_error_percent": (
                        100.0 * (sampled_value - full_value) / full_value
                    ),
                }
            )

        central_rows = comparison[comparison["case"] == "central"]
        full_optimised_path = (
            full_year_root
            / args.full_year_optimised_scenario
            / "annual_summary.json"
        )
        if not central_rows.empty and full_optimised_path.exists():
            full_optimised = _load_summary(full_optimised_path)
            sampled_central_cost = float(
                central_rows.iloc[0]["estimated_annual_cost_gbp"]
            )
            sampled_central_energy = float(
                central_rows.iloc[0]["estimated_annual_grid_energy_kwh"]
            )
            sampled_saving_percent = float(
                central_rows.iloc[0]["estimated_annual_saving_percent"]
            )
            full_central_cost = float(full_optimised["settlement_cost_gbp"])
            full_central_energy = float(full_optimised["grid_energy_kwh"])
            full_saving_percent = (
                100.0
                * (
                    float(full_baseline["settlement_cost_gbp"])
                    - full_central_cost
                )
                / float(full_baseline["settlement_cost_gbp"])
            )
            for metric, sampled_value, full_value in (
                (
                    "central_settlement_cost_gbp",
                    sampled_central_cost,
                    full_central_cost,
                ),
                (
                    "central_grid_energy_kwh",
                    sampled_central_energy,
                    full_central_energy,
                ),
                (
                    "central_saving_percent",
                    sampled_saving_percent,
                    full_saving_percent,
                ),
            ):
                validation_rows.append(
                    {
                        "metric": metric,
                        "sampled_value": sampled_value,
                        "full_year_value": full_value,
                        "absolute_error": sampled_value - full_value,
                        "relative_error_percent": (
                            100.0 * (sampled_value - full_value) / full_value
                        ),
                    }
                )

    if validation_rows:
        validation = pd.DataFrame(validation_rows)
        validation_path = year_root / "sampling_validation.csv"
        validation.to_csv(validation_path, index=False, float_format="%.15g")
        print("\nSampling validation against completed full-year runs")
        print(validation.to_string(index=False))
        print(f"\nValidation written to {validation_path}")


if __name__ == "__main__":
    main()
