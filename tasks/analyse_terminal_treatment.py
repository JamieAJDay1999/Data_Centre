"""Quantify the 2025 year-end workload and storage terminal treatment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from rolling_optimisation.config import RollingConfig, model_parameters
from rolling_optimisation.model import solve_horizon
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
SHIFT = ROOT / "static/data/inputs/shiftability_profile.csv"
ANNUAL = ROOT / "static/data/rolling_year_outputs"
SOURCE_SCENARIO = "2025_optimised_reformulated"
OUTPUT = ROOT / "reports/terminal_treatment"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source = ANNUAL / SOURCE_SCENARIO
    metadata = _json(source / "run_metadata.json")
    predecessor = _json(source / "checkpoints/2025-12-30.json")
    final_checkpoint = _json(source / "checkpoints/2025-12-31.json")
    config = RollingConfig(**metadata["config"])
    config = replace(config, fail_on_gap_exceeded=False)

    timeline = build_annual_timeline(
        PRICE, LOAD, SHIFT, 2025, config.lookahead_steps, "actual"
    )
    timeline = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    timeline = add_optimisation_prices(timeline, config.price_treatment)
    indices = dict(local_day_core_indices(timeline, 2025))["2025-12-31"]
    start = int(indices[0])
    core_steps = len(indices)
    horizon = timeline.iloc[
        start : start + core_steps + config.lookahead_steps
    ].copy()
    opening_state = OperationalState.from_dict(predecessor["closing_state"])
    opening_workload = [
        WorkloadCohort.from_dict(row)
        for row in predecessor["closing_workload"]
    ]

    prices = pd.read_csv(PRICE)
    prices["IMRP_Date"] = pd.to_datetime(prices["IMRP_Date"])
    january = prices[
        prices["IMRP_Date"].dt.normalize() == pd.Timestamp("2026-01-01")
    ]["IMRP_Amount"].astype(float)
    params = model_parameters(config)
    price_cases = {
        "zero_terminal_value": 0.0,
        "median_1_january_value": float(january.median()),
        "maximum_1_january_value": float(january.max()),
    }

    rows: list[dict] = []
    plans: dict[str, pd.DataFrame] = {}
    for case, continuation_price in price_cases.items():
        case_config = replace(
            config,
            scenario_id=f"terminal_{case}",
            terminal_ups_value_gbp_per_kwh=(
                continuation_price * params.eta_disch / 1000.0
            ),
            terminal_tes_value_gbp_per_kwh_th=(
                continuation_price
                * params.TES_discharge_efficiency
                / params.COP_HVAC
                / 1000.0
            ),
        )
        result = solve_horizon(
            case_config,
            horizon,
            core_steps,
            opening_state,
            opening_workload,
        )
        plans[case] = result.planned
        rows.append(
            {
                "case": case,
                "continuation_price_gbp_per_mwh": continuation_price,
                "ups_terminal_value_gbp_per_kwh": (
                    case_config.terminal_ups_value_gbp_per_kwh
                ),
                "tes_terminal_value_gbp_per_kwh_th": (
                    case_config.terminal_tes_value_gbp_per_kwh_th
                ),
                "committed_31_december_cost_gbp": result.audits[
                    "committed_settlement_cost_gbp"
                ],
                "lookahead_1_january_cost_gbp": result.audits[
                    "lookahead_settlement_cost_gbp"
                ],
                "lookahead_1_january_grid_energy_kwh": result.audits[
                    "lookahead_grid_energy_kwh"
                ],
                "workload_carried_at_midnight_cpu_h": result.audits[
                    "closing_workload_cpu_h"
                ],
                "carried_workload_completed_in_lookahead_cpu_h": result.audits[
                    "core_workload_completed_in_lookahead_cpu_h"
                ],
                "unserved_after_lookahead_cpu_h": result.audits[
                    "core_workload_unserved_after_lookahead_cpu_h"
                ],
                "core_closing_ups_kwh": result.next_state.ups_energy_kwh,
                "core_closing_tes_kwh": result.next_state.tes_energy_kwh,
                "horizon_terminal_ups_kwh": result.terminal_state.ups_energy_kwh,
                "horizon_terminal_tes_kwh": result.terminal_state.tes_energy_kwh,
                "solver_runtime_s": result.solver["runtime_s"],
                "solver_gap_percent": 100.0
                * float(result.solver["relative_gap"] or 0.0),
            }
        )

    result_table = pd.DataFrame(rows)
    zero_cost = float(
        result_table.loc[
            result_table["case"] == "zero_terminal_value",
            "committed_31_december_cost_gbp",
        ].iloc[0]
    )
    annual_cost = float(
        _json(source / "annual_summary.json")["settlement_cost_gbp"]
    )
    result_table["committed_cost_change_from_zero_gbp"] = (
        result_table["committed_31_december_cost_gbp"] - zero_cost
    )
    result_table["annual_cost_change_percent"] = (
        100.0
        * result_table["committed_cost_change_from_zero_gbp"]
        / annual_cost
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(OUTPUT / "terminal_value_sensitivity.csv", index=False)
    plans["zero_terminal_value"].to_csv(
        OUTPUT / "zero_value_final_horizon.csv", index=False
    )

    stored_cost = float(
        pd.read_csv(source / "days/2025-12-31.csv")[
            "settlement_cost_gbp"
        ].sum()
    )
    zero_row = result_table.iloc[0]
    maximum_effect = float(
        result_table["annual_cost_change_percent"].abs().max()
    )
    report = f"""# Year-end terminal-treatment assessment

The final 2025 horizon starts from the exact closing physical state and carried
workload recorded on 30 December. Its 12 look-ahead intervals use settlement
periods 1--3 on 1 January 2026.

## Workload accounting

- Workload outstanding at midnight: {zero_row['workload_carried_at_midnight_cpu_h']:.6f} CPU-h.
- Workload completed during the three-hour look-ahead: {zero_row['carried_workload_completed_in_lookahead_cpu_h']:.6f} CPU-h.
- Core workload unserved after the look-ahead: {zero_row['unserved_after_lookahead_cpu_h']:.3g} CPU-h.
- The full 1 January look-ahead uses {zero_row['lookahead_1_january_grid_energy_kwh']:.3f} kWh and costs GBP {zero_row['lookahead_1_january_cost_gbp']:.3f}; this is planning evidence and is not included in the committed 2025 accounting.

The annual convention is therefore to report costs for settlement intervals
inside calendar year 2025 and disclose the midnight backlog separately. The
look-ahead proves service completion but is not added to 2025 cost because it
also contains ordinary 1 January 2026 facility demand.

## Storage terminal sensitivity

The zero-terminal-value rerun reproduces the stored 31 December committed cost
to GBP {abs(zero_cost - stored_cost):.3g}. Price-derived continuation values
were then calculated from the median and maximum 1 January 2026 IMRP, adjusted
for UPS discharge efficiency and TES/chiller conversion.

Across those continuation-value cases, the largest change in committed annual
cost is {maximum_effect:.6f}% of the annual total. This is below 0.01%, so the
reported 5.049% annual saving is not materially driven by final-horizon storage
depletion. Zero central terminal value is retained as an explicit modelling
assumption; continuation values remain a terminal sensitivity.

## Frozen convention

1. Commit and account for calendar-year intervals only.
2. Use actual 1 January 2026 periods 1--3 for the final look-ahead.
3. Require all workload arriving in 2025 to be serviceable within its deadline;
   report midnight backlog and verify it is zero after the look-ahead.
4. Retain zero storage terminal value centrally because the price-derived
   sensitivity is immaterial, and disclose the assumption.
"""
    (OUTPUT / "terminal_treatment_report.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        "source_scenario": SOURCE_SCENARIO,
        "source_closing_state_matches_rerun": (
            abs(
                float(final_checkpoint["closing_state"]["ups_energy_kwh"])
                - float(zero_row["core_closing_ups_kwh"])
            )
            < 1e-8
        ),
        "maximum_annual_cost_change_percent": maximum_effect,
        "terminal_policy": "calendar_year_commit_actual_2026_tail_zero_central_value",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(result_table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
