from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

import rolling_optimisation.run_rolling_year as annual_runner
from rolling_optimisation.config import (
    RollingConfig,
    default_initial_state,
    model_parameters,
)
from rolling_optimisation.model import new_workload_cohorts, solve_horizon
from rolling_optimisation.run_representative_day_flexibility import (
    FIXED_RECOVERY_STEPS,
    _state_at_boundary,
    _workload_at_boundary,
)
from rolling_optimisation.timeline import (
    add_optimisation_prices,
    apply_flexible_workload_multiplier,
    build_annual_timeline,
    local_day_core_indices,
)
from rolling_optimisation.types import (
    FlexibilityRequest,
    OperationalState,
    WorkloadCohort,
)


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


def test_logarithmic_piecewise_configuration_validation() -> None:
    config = RollingConfig()
    assert config.it_power_representation == "DLOG"
    assert config.it_power_segments == 4
    with pytest.raises(ValueError, match="power of two"):
        RollingConfig(it_power_segments=6, it_power_representation="DLOG")


def test_sensitivity_multipliers_preserve_capacity_fractions() -> None:
    config = RollingConfig(
        ups_capacity_multiplier=0.5,
        tes_capacity_multiplier=1.5,
        flexible_workload_multiplier=1.5,
    )
    params = model_parameters(config)
    state = default_initial_state(config)

    assert params.e_nom_kwh == pytest.approx(300)
    assert params.e_min_kwh == pytest.approx(150)
    assert params.e_max_kwh == pytest.approx(300)
    assert state.ups_energy_kwh == pytest.approx(300)
    assert params.TES_capacity_kWh == pytest.approx(1500)
    assert state.tes_energy_kwh == pytest.approx(750)

    with pytest.raises(ValueError, match="must be positive"):
        RollingConfig(ups_capacity_multiplier=0)


def test_annual_runner_parses_sensitivity_multipliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rolling_year.py",
            "--ups-capacity-multiplier",
            "0.5",
            "--tes-capacity-multiplier",
            "1.5",
            "--flexible-workload-multiplier",
            "1.25",
        ],
    )

    args = annual_runner.parse_args()

    assert args.ups_capacity_multiplier == pytest.approx(0.5)
    assert args.tes_capacity_multiplier == pytest.approx(1.5)
    assert args.flexible_workload_multiplier == pytest.approx(1.25)


def test_annual_runner_forwards_sensitivity_multipliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, RollingConfig] = {}

    def fake_run_rolling_scenario(**kwargs: object) -> dict[str, object]:
        captured["config"] = kwargs["config"]  # type: ignore[assignment]
        return {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rolling_year.py",
            "--scenario-id",
            "cli_multiplier_test",
            "--ups-capacity-multiplier",
            "0.5",
            "--tes-capacity-multiplier",
            "1.5",
            "--flexible-workload-multiplier",
            "1.25",
        ],
    )
    monkeypatch.setattr(
        annual_runner, "run_rolling_scenario", fake_run_rolling_scenario
    )

    annual_runner.main()

    config = captured["config"]
    assert config.ups_capacity_multiplier == pytest.approx(0.5)
    assert config.tes_capacity_multiplier == pytest.approx(1.5)
    assert config.flexible_workload_multiplier == pytest.approx(1.25)


def test_flexible_multiplier_preserves_interval_demand(
    timeline: pd.DataFrame,
) -> None:
    original_total = timeline["inflexible_cpu"] + timeline["flexible_cpu"]
    scaled = apply_flexible_workload_multiplier(timeline, 1.5)
    scaled_total = scaled["inflexible_cpu"] + scaled["flexible_cpu"]

    pd.testing.assert_series_equal(scaled_total, original_total)
    assert (scaled["flexible_cpu"] <= scaled_total).all()
    assert (scaled["inflexible_cpu"] >= 0).all()


def test_price_treatments_preserve_settlement_series(timeline: pd.DataFrame) -> None:
    original = timeline["settlement_price_gbp_per_mwh"].copy()
    signed = add_optimisation_prices(timeline, "signed")
    capped = add_optimisation_prices(timeline, "floor_zero")
    shifted = add_optimisation_prices(timeline, "shift_year_min")

    pd.testing.assert_series_equal(
        signed["settlement_price_gbp_per_mwh"], original
    )
    pd.testing.assert_series_equal(
        capped["settlement_price_gbp_per_mwh"], original
    )
    assert capped["optimisation_price_gbp_per_mwh"].min() == 0
    assert shifted.loc[
        shifted["is_target_year"], "optimisation_price_gbp_per_mwh"
    ].min() == pytest.approx(0)


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
    assert first.audits["core_workload_unserved_after_lookahead_cpu_h"] == pytest.approx(
        0
    )
    assert first.solver["found_feasible_incumbent"]
    assert first.solver["binary_variables"] == 64
    assert first.audits["maximum_it_power_approximation_error_kw"] < 6.25
    direct = (
        first.committed["grid_import_kw"]
        * first.committed["settlement_price_gbp_per_mwh"]
        / 1000
        * config.dt_hours
    ).sum()
    assert first.committed["settlement_cost_gbp"].sum() == pytest.approx(direct)
    assert len(first.planned) == first_core + config.lookahead_steps
    traced = first.workload_trace.groupby("timestamp_utc")[
        "executed_cpu_rate"
    ].sum()
    for row in first.committed.itertuples(index=False):
        assert float(traced.get(row.timestamp_utc, 0.0)) == pytest.approx(
            row.flexible_cpu_processed
        )
    assert first.audits["workload_trace_max_interval_residual_cpu"] <= (
        config.workload_tolerance_cpu_h
    )
    assert first.terminal_state == OperationalState.from_dict(
        {
            key.removeprefix("state_end_"): first.planned.iloc[-1][key]
            for key in first.planned.columns
            if key.startswith("state_end_")
        }
    )


@pytest.mark.integration
def test_flexibility_request_tracks_grid_target_and_recovers(
    timeline: pd.DataFrame,
) -> None:
    solver = _available_solver()
    if solver is None:
        pytest.skip("No supported MILP solver is available")
    config = RollingConfig(
        scenario_id="flexibility_request_test",
        solver_name=solver,
        solver_time_limit_s=30,
        mip_gap=0.01,
    )
    core_steps = 8
    start = 24
    horizon = timeline.iloc[
        start : start + core_steps + config.lookahead_steps
    ]
    baseline = solve_horizon(
        config, horizon, core_steps, default_initial_state(config)
    )
    request = FlexibilityRequest(
        baseline_grid_import_kw=tuple(baseline.committed["grid_import_kw"]),
        start_step=2,
        duration_steps=1,
        delta_kw=-10.0,
        baseline_total_cpu=tuple(baseline.committed["total_cpu"]),
        event_initial_state=OperationalState.from_dict(
            {
                key.removeprefix("state_start_"): baseline.committed.iloc[2][key]
                for key in baseline.committed.columns
                if key.startswith("state_start_")
            }
        ),
        recovery_state=baseline.terminal_state,
    )
    response = solve_horizon(
        config,
        horizon,
        core_steps,
        default_initial_state(config),
        flexibility_request=request,
    )

    target = baseline.committed.iloc[2]["grid_import_kw"] - 10.0
    assert response.committed.iloc[2]["grid_import_kw"] == pytest.approx(
        target, abs=request.tolerance_kw
    )
    assert response.committed.iloc[:2]["total_cpu"].to_list() == pytest.approx(
        baseline.committed.iloc[:2]["total_cpu"].to_list()
    )
    for key, value in request.event_initial_state.to_dict().items():
        assert response.committed.iloc[2][f"state_start_{key}"] == pytest.approx(
            value
        )
    assert (
        response.terminal_state.ups_energy_kwh
        >= baseline.terminal_state.ups_energy_kwh - 1e-6
    )
    assert (
        response.terminal_state.tes_energy_kwh
        >= baseline.terminal_state.tes_energy_kwh - 1e-6
    )


@pytest.mark.integration
def test_event_boundary_cohorts_and_fixed_recovery(
    timeline: pd.DataFrame,
) -> None:
    solver = _available_solver()
    if solver is None:
        pytest.skip("No supported MILP solver is available")
    config = RollingConfig(
        scenario_id="fixed_recovery_test",
        solver_name=solver,
        solver_time_limit_s=30,
        mip_gap=0.01,
    )
    day_start = 24
    core_steps = 16
    full_horizon = timeline.iloc[
        day_start : day_start + core_steps + config.lookahead_steps
    ].copy()
    initial_state = default_initial_state(config)
    baseline = solve_horizon(
        config,
        full_horizon,
        core_steps,
        initial_state,
    )
    event_start = 2
    duration = 1
    recovery_boundary = event_start + duration + FIXED_RECOVERY_STEPS
    event_horizon = full_horizon.iloc[event_start:recovery_boundary].copy()
    opening_workload = _workload_at_boundary(
        config,
        full_horizon,
        [],
        baseline.workload_trace.assign(
            timestamp_utc=pd.to_datetime(
                baseline.workload_trace["timestamp_utc"], utc=True
            )
        ),
        event_start,
    )
    recovery_workload = _workload_at_boundary(
        config,
        full_horizon,
        [],
        baseline.workload_trace.assign(
            timestamp_utc=pd.to_datetime(
                baseline.workload_trace["timestamp_utc"], utc=True
            )
        ),
        recovery_boundary,
    )
    reference = baseline.planned.iloc[event_start:recovery_boundary]
    request = FlexibilityRequest(
        baseline_grid_import_kw=tuple(reference["grid_import_kw"]),
        start_step=0,
        duration_steps=duration,
        delta_kw=-10.0,
        recovery_state=_state_at_boundary(baseline.planned, recovery_boundary),
        recovery_workload=tuple(recovery_workload),
    )
    response = solve_horizon(
        config,
        event_horizon,
        duration,
        _state_at_boundary(baseline.planned, event_start),
        opening_workload,
        flexibility_request=request,
    )

    assert len(response.planned) == duration + FIXED_RECOVERY_STEPS
    assert response.committed.iloc[0]["grid_import_kw"] == pytest.approx(
        reference.iloc[0]["grid_import_kw"] - 10.0,
        abs=request.tolerance_kw,
    )
    assert response.audits["recovery_workload_max_excess_cpu_h"] <= (
        request.recovery_workload_tolerance_cpu_h + 1e-9
    )


@pytest.mark.integration
def test_optional_physical_costs_and_limits_reconcile(timeline: pd.DataFrame) -> None:
    solver = _available_solver()
    if solver is None:
        pytest.skip("No supported MILP solver is available")
    config = RollingConfig(
        scenario_id="physical_cost_test",
        solver_name=solver,
        solver_time_limit_s=30,
        mip_gap=0.01,
        ups_reserve_kwh=400,
        grid_import_limit_kw=1700,
        ups_throughput_cost_gbp_per_kwh=0.01,
        tes_throughput_cost_gbp_per_kwh_th=0.002,
        terminal_ups_value_gbp_per_kwh=0.02,
        terminal_tes_value_gbp_per_kwh_th=0.001,
    )
    result = solve_horizon(
        config,
        timeline.iloc[12 : 12 + 4 + config.lookahead_steps],
        4,
        default_initial_state(config),
    )

    assert result.audits["objective_reconciliation_gbp"] == pytest.approx(0)
    assert result.audits["effective_grid_import_limit_kw"] == 1700
    assert (
        result.committed["state_start_ups_energy_kwh"].min()
        >= config.ups_reserve_kwh
    )


@pytest.mark.integration
def test_sensitivity_multipliers_solve_at_combined_extremes(
    timeline: pd.DataFrame,
) -> None:
    solver = _available_solver()
    if solver is None:
        pytest.skip("No supported MILP solver is available")
    config = RollingConfig(
        scenario_id="sensitivity_extremes_test",
        solver_name=solver,
        solver_time_limit_s=30,
        mip_gap=0.01,
        ups_capacity_multiplier=0.5,
        tes_capacity_multiplier=0.5,
        flexible_workload_multiplier=1.5,
    )
    scaled = apply_flexible_workload_multiplier(
        timeline, config.flexible_workload_multiplier
    )
    result = solve_horizon(
        config,
        scaled.iloc[12 : 12 + 4 + config.lookahead_steps],
        4,
        default_initial_state(config),
    )

    assert result.solver["found_feasible_incumbent"]
    assert result.committed["state_start_ups_energy_kwh"].max() <= 300 + 1e-6
    assert result.committed["state_start_tes_energy_kwh"].max() <= 500 + 1e-6
