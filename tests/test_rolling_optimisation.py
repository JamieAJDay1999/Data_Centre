from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

from rolling_optimisation.config import RollingConfig, default_initial_state
from rolling_optimisation.model import new_workload_cohorts, solve_horizon
from rolling_optimisation.timeline import build_annual_timeline, local_day_core_indices
from rolling_optimisation.types import OperationalState, WorkloadCohort


ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "static" / "data" / "imrp_actuals.csv"
LOAD = ROOT / "static" / "data" / "inputs" / "load_profiles.csv"
SHIFT = ROOT / "static" / "data" / "inputs" / "shiftability_profile.csv"


@pytest.fixture(scope="module")
def timeline() -> pd.DataFrame:
    return build_annual_timeline(PRICE, LOAD, SHIFT, 2025)


def test_annual_timeline_covers_dst_and_every_interval(timeline: pd.DataFrame) -> None:
    target = timeline[timeline["is_target_year"]]
    counts = target.groupby("local_date").size()
    assert len(target) == 35_040
    assert counts.value_counts().to_dict() == {96: 363, 92: 1, 100: 1}
    assert counts.loc["2025-03-30"] == 92
    assert counts.loc["2025-10-26"] == 100
    assert len(local_day_core_indices(timeline, 2025)) == 365
    timestamps = target["timestamp_utc"]
    assert not timestamps.duplicated().any()
    assert (timestamps.diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_state_and_workload_round_trip() -> None:
    config = RollingConfig()
    state = default_initial_state(config)
    assert OperationalState.from_dict(state.to_dict()) == state
    cohort = WorkloadCohort(
        cohort_id="test",
        arrival_utc="2025-01-01T23:45:00+00:00",
        latest_start_utc="2025-01-02T02:45:00+00:00",
        remaining_cpu_hours=0.1,
        tranche=4,
    )
    assert WorkloadCohort.from_dict(cohort.to_dict()) == cohort


def test_generated_workload_matches_flexible_arrivals(timeline: pd.DataFrame) -> None:
    config = RollingConfig()
    horizon = timeline.iloc[:4].copy()
    cohorts = new_workload_cohorts(horizon, config)
    expected = float(horizon["flexible_cpu"].sum() * config.dt_hours)
    assert sum(cohort.remaining_cpu_hours for cohort in cohorts) == pytest.approx(expected)
    assert all(cohort.latest_start >= cohort.arrival for cohort in cohorts)


def _available_solver() -> str | None:
    for name in ("scip", "appsi_highs"):
        if pyo.SolverFactory(name).available(exception_flag=False):
            return name
    return None


@pytest.mark.integration
def test_two_linked_signed_price_horizons(timeline: pd.DataFrame) -> None:
    solver = _available_solver()
    if solver is None:
        pytest.skip("No supported MILP solver is available")
    config = RollingConfig(
        scenario_id="test",
        solver_name=solver,
        solver_time_limit_s=30,
        mip_gap=0.01,
    )
    first_core = 4
    first_start = 12
    first = solve_horizon(
        config,
        timeline.iloc[
            first_start : first_start + first_core + config.lookahead_steps
        ],
        first_core,
        default_initial_state(config),
    )
    second_start = first_start + first_core
    second = solve_horizon(
        config,
        timeline.iloc[
            second_start : second_start + first_core + config.lookahead_steps
        ],
        first_core,
        first.next_state,
        first.next_workload,
    )

    assert first.committed["settlement_price_gbp_per_mwh"].min() < 0
    assert first.audits["initial_state_max_residual"] <= config.state_tolerance
    assert second.audits["initial_state_max_residual"] <= config.state_tolerance
    assert abs(first.audits["workload_conservation_residual_cpu_h"]) <= (
        config.workload_tolerance_cpu_h
    )
    assert abs(second.audits["workload_conservation_residual_cpu_h"]) <= (
        config.workload_tolerance_cpu_h
    )
    assert first.audits["max_ups_charge_discharge_overlap_kw"] == pytest.approx(0)
    assert first.audits["max_tes_charge_discharge_overlap_kw"] == pytest.approx(0)
    direct = (
        first.committed["grid_import_kw"]
        * first.committed["settlement_price_gbp_per_mwh"]
        / 1000
        * config.dt_hours
    ).sum()
    assert first.committed["settlement_cost_gbp"].sum() == pytest.approx(direct)
