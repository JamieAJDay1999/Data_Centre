from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from inputs.parameters_optimisation import ModelParameters

from .config import RollingConfig
from .timeline import TRANCHE_DELAYS
from .types import HorizonResult, OperationalState, WorkloadCohort


def _iso(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat()


def new_workload_cohorts(
    horizon: pd.DataFrame,
    config: RollingConfig,
) -> list[WorkloadCohort]:
    cohorts: list[WorkloadCohort] = []
    for row in horizon.itertuples(index=False):
        arrival = pd.Timestamp(row.timestamp_utc)
        for tranche, delay_steps in TRANCHE_DELAYS.items():
            work = (
                float(row.flexible_cpu)
                * float(getattr(row, f"shift_fraction_{tranche}"))
                * config.dt_hours
            )
            if work <= config.workload_tolerance_cpu_h:
                continue
            latest = arrival + pd.Timedelta(seconds=config.dt_seconds * delay_steps)
            cohorts.append(
                WorkloadCohort(
                    cohort_id=f"{arrival.isoformat()}__t{tranche}",
                    arrival_utc=_iso(arrival),
                    latest_start_utc=_iso(latest),
                    remaining_cpu_hours=work,
                    tranche=tranche,
                )
            )
    return cohorts


def _select_solver(config: RollingConfig):
    candidates = (
        ["scip", "appsi_highs"]
        if config.solver_name == "auto"
        else [config.solver_name]
    )
    for name in candidates:
        solver = pyo.SolverFactory(name)
        if solver.available(exception_flag=False):
            if name == "scip":
                solver.options["limits/time"] = config.solver_time_limit_s
                solver.options["limits/gap"] = config.mip_gap
            elif name == "appsi_highs":
                solver.options["time_limit"] = config.solver_time_limit_s
                solver.options["mip_rel_gap"] = config.mip_gap
            return name, solver
    raise RuntimeError(
        f"No requested MILP solver is available. Tried: {', '.join(candidates)}"
    )


def _state_from_model(model, boundary: int) -> OperationalState:
    return OperationalState(
        ups_energy_kwh=pyo.value(model.e_ups_kwh[boundary]),
        tes_energy_kwh=pyo.value(model.e_tes_kwh[boundary]),
        it_temperature_c=pyo.value(model.t_it[boundary]),
        rack_temperature_c=pyo.value(model.t_rack[boundary]),
        cold_aisle_temperature_c=pyo.value(model.t_cold[boundary]),
        hot_aisle_temperature_c=pyo.value(model.t_hot[boundary]),
    )


def _max_state_difference(left: OperationalState, right: OperationalState) -> float:
    return max(abs(left.to_dict()[key] - right.to_dict()[key]) for key in left.to_dict())


def solve_horizon(
    config: RollingConfig,
    horizon: pd.DataFrame,
    core_steps: int,
    initial_state: OperationalState,
    carried_workload: Iterable[WorkloadCohort] = (),
    *,
    tee: bool = False,
) -> HorizonResult:
    """Solve one local-day core plus look-ahead and return only committed actions."""

    horizon = horizon.reset_index(drop=True).copy()
    n_steps = len(horizon)
    if core_steps < 1 or core_steps >= n_steps:
        raise ValueError("core_steps must leave at least one look-ahead interval")
    if n_steps - core_steps < config.lookahead_steps:
        raise ValueError("Horizon does not contain the configured look-ahead")

    params = ModelParameters(dt_seconds=config.dt_seconds)
    carried = list(carried_workload)
    generated = [] if config.mode == "baseline" else new_workload_cohorts(horizon, config)
    cohorts = carried + generated
    if len({cohort.cohort_id for cohort in cohorts}) != len(cohorts):
        raise ValueError("Duplicate workload cohort identifiers in horizon")

    timestamps = [pd.Timestamp(value) for value in horizon["timestamp_utc"]]
    last_interval = timestamps[-1]
    allowed: dict[str, list[int]] = {}
    for cohort in cohorts:
        slots = [
            step
            for step, timestamp in enumerate(timestamps)
            if timestamp >= cohort.arrival and timestamp <= cohort.latest_start
        ]
        if not slots and cohort.remaining_cpu_hours > config.workload_tolerance_cpu_h:
            raise ValueError(f"No execution interval available for {cohort.cohort_id}")
        allowed[cohort.cohort_id] = slots

    m = pyo.ConcreteModel(name=f"Rolling_DC_{config.mode}")
    m.I = pyo.RangeSet(0, n_steps - 1)
    m.B = pyo.RangeSet(0, n_steps)

    # Physical boundary states.
    m.e_ups_kwh = pyo.Var(m.B, bounds=(params.e_min_kwh, params.e_max_kwh))
    m.e_tes_kwh = pyo.Var(
        m.B, bounds=(params.E_TES_min_kWh, params.TES_capacity_kWh)
    )
    m.t_it = pyo.Var(m.B, bounds=(18.0, 60.0))
    m.t_rack = pyo.Var(m.B, bounds=(18.0, 40.0))
    m.t_cold = pyo.Var(
        m.B, bounds=(params.T_cAisle_lower_limit_Celsius, params.T_cAisle_upper_limit_Celsius)
    )
    m.t_hot = pyo.Var(m.B, bounds=(18.0, 40.0))

    m.initial_state = pyo.ConstraintList()
    m.initial_state.add(m.e_ups_kwh[0] == initial_state.ups_energy_kwh)
    m.initial_state.add(m.e_tes_kwh[0] == initial_state.tes_energy_kwh)
    m.initial_state.add(m.t_it[0] == initial_state.it_temperature_c)
    m.initial_state.add(m.t_rack[0] == initial_state.rack_temperature_c)
    m.initial_state.add(m.t_cold[0] == initial_state.cold_aisle_temperature_c)
    m.initial_state.add(m.t_hot[0] == initial_state.hot_aisle_temperature_c)

    # Workload and exact piecewise-linear CPU power.
    m.total_cpu = pyo.Var(m.I, bounds=(0.0, params.max_cpu_usage))
    m.p_it_total_kw = pyo.Var(
        m.I, bounds=(params.idle_power_kw, params.max_power_kw)
    )
    pair_index = [
        (cohort.cohort_id, step)
        for cohort in cohorts
        for step in allowed[cohort.cohort_id]
    ]
    m.X_INDEX = pyo.Set(initialize=pair_index, dimen=2)
    m.work_rate = pyo.Var(m.X_INDEX, bounds=(0.0, params.max_cpu_usage))

    if cohorts:
        m.work_completion = pyo.ConstraintList()
        by_id = {cohort.cohort_id: cohort for cohort in cohorts}
        for cohort_id, cohort in by_id.items():
            expression = sum(
                m.work_rate[cohort_id, step] * config.dt_hours
                for step in allowed[cohort_id]
            )
            if cohort.latest_start <= last_interval:
                m.work_completion.add(expression == cohort.remaining_cpu_hours)
            else:
                m.work_completion.add(expression <= cohort.remaining_cpu_hours)

    m.cpu_balance = pyo.ConstraintList()
    for step in range(n_steps):
        if config.mode == "baseline":
            flexible_rate = float(horizon.at[step, "flexible_cpu"])
        else:
            flexible_rate = sum(
                m.work_rate[cohort.cohort_id, step]
                for cohort in cohorts
                if (cohort.cohort_id, step) in m.X_INDEX
            )
        m.cpu_balance.add(
            m.total_cpu[step]
            == float(horizon.at[step, "inflexible_cpu"]) + flexible_rate
        )

    breakpoints = np.linspace(0.0, 1.0, 11)
    power_factors = breakpoints**1.32
    m.P = pyo.RangeSet(0, len(breakpoints) - 1)
    m.SEG = pyo.RangeSet(0, len(breakpoints) - 2)
    m.weight = pyo.Var(m.I, m.P, within=pyo.NonNegativeReals)
    m.segment = pyo.Var(m.I, m.SEG, within=pyo.Binary)
    m.piecewise = pyo.ConstraintList()
    for step in range(n_steps):
        m.piecewise.add(sum(m.weight[step, point] for point in m.P) == 1)
        m.piecewise.add(sum(m.segment[step, segment] for segment in m.SEG) == 1)
        m.piecewise.add(m.weight[step, 0] <= m.segment[step, 0])
        m.piecewise.add(
            m.weight[step, len(breakpoints) - 1]
            <= m.segment[step, len(breakpoints) - 2]
        )
        for point in range(1, len(breakpoints) - 1):
            m.piecewise.add(
                m.weight[step, point]
                <= m.segment[step, point - 1] + m.segment[step, point]
            )
        m.piecewise.add(
            m.total_cpu[step]
            == sum(
                float(breakpoints[point]) * m.weight[step, point] for point in m.P
            )
        )
        m.piecewise.add(
            m.p_it_total_kw[step]
            == params.idle_power_kw
            + (params.max_power_kw - params.idle_power_kw)
            * sum(
                float(power_factors[point]) * m.weight[step, point] for point in m.P
            )
        )

    # Interval actions.
    storage_bound = (0.0, 0.0) if config.mode == "baseline" else None
    m.p_grid_it_kw = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.p_ups_ch_kw = pyo.Var(
        m.I, within=pyo.NonNegativeReals, bounds=storage_bound
    )
    m.p_ups_disch_kw = pyo.Var(
        m.I, within=pyo.NonNegativeReals, bounds=storage_bound
    )
    m.z_ups_ch = pyo.Var(m.I, within=pyo.Binary)
    m.z_ups_disch = pyo.Var(m.I, within=pyo.Binary)

    tes_bound = (0.0, 0.0) if config.mode == "baseline" else (0.0, params.TES_w_charge_max)
    tes_dis_bound = (
        (0.0, 0.0)
        if config.mode == "baseline"
        else (0.0, params.TES_w_discharge_max)
    )
    m.p_chiller_hvac_w = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.p_chiller_tes_w = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.q_cool_w = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.q_ch_tes_w = pyo.Var(m.I, bounds=tes_bound)
    m.q_dis_tes_w = pyo.Var(m.I, bounds=tes_dis_bound)
    m.z_tes_ch = pyo.Var(m.I, within=pyo.Binary)
    m.z_tes_disch = pyo.Var(m.I, within=pyo.Binary)
    m.t_in = pyo.Var(m.I, bounds=(18.0, 30.0))

    m.physics = pyo.ConstraintList()
    mcp = params.m_dot_air * params.c_p_air
    for step in range(n_steps):
        m.physics.add(
            m.p_it_total_kw[step]
            == m.p_grid_it_kw[step] + m.p_ups_disch_kw[step]
        )
        m.physics.add(m.p_ups_ch_kw[step] <= params.p_max_ch_kw * m.z_ups_ch[step])
        m.physics.add(
            m.p_ups_disch_kw[step] <= params.p_max_disch_kw * m.z_ups_disch[step]
        )
        m.physics.add(m.z_ups_ch[step] + m.z_ups_disch[step] <= 1)
        m.physics.add(
            m.e_ups_kwh[step + 1]
            == m.e_ups_kwh[step]
            + params.eta_ch * m.p_ups_ch_kw[step] * config.dt_hours
            - m.p_ups_disch_kw[step] / params.eta_disch * config.dt_hours
        )

        m.physics.add(
            m.q_ch_tes_w[step] <= params.TES_w_charge_max * m.z_tes_ch[step]
        )
        m.physics.add(
            m.q_dis_tes_w[step]
            <= params.TES_w_discharge_max * m.z_tes_disch[step]
        )
        m.physics.add(m.z_tes_ch[step] + m.z_tes_disch[step] <= 1)
        m.physics.add(m.q_ch_tes_w[step] == params.COP_HVAC * m.p_chiller_tes_w[step])
        m.physics.add(
            m.q_cool_w[step]
            == params.COP_HVAC * m.p_chiller_hvac_w[step]
            + m.q_dis_tes_w[step]
        )
        m.physics.add(
            m.p_chiller_hvac_w[step] + m.p_chiller_tes_w[step]
            <= params.P_chiller_max
        )
        m.physics.add(
            m.e_tes_kwh[step + 1]
            == m.e_tes_kwh[step]
            + (
                m.q_ch_tes_w[step] * params.TES_charge_efficiency
                - m.q_dis_tes_w[step] / params.TES_discharge_efficiency
            )
            * config.dt_hours
            / 1000.0
        )

        # Backward (implicit) Euler is used for the thermal states. The original
        # explicit 15-minute update is unstable for the configured air-flow time
        # constants and only remained feasible because its first states were free.
        # The implicit form is linear and fixes that numerical/physical defect.
        m.physics.add(
            m.t_in[step]
            == m.t_hot[step + 1] - m.q_cool_w[step] / mcp
        )
        m.physics.add(
            m.q_cool_w[step]
            <= (m.t_hot[step + 1] - params.T_cAisle_lower_limit_Celsius) * mcp
        )
        m.physics.add(m.q_cool_w[step] >= m.p_it_total_kw[step] * 1000.0)

        m.physics.add(
            m.t_it[step + 1]
            == m.t_it[step]
            + config.dt_seconds
            * (
                m.p_it_total_kw[step] * 1000.0
                - params.G_conv * (m.t_it[step + 1] - m.t_rack[step + 1])
            )
            / params.C_IT
        )
        m.physics.add(
            m.t_rack[step + 1]
            == m.t_rack[step]
            + config.dt_seconds
            * (
                params.m_dot_air
                * params.kappa
                * params.c_p_air
                * (m.t_cold[step + 1] - m.t_rack[step + 1])
                + params.G_conv
                * (m.t_it[step + 1] - m.t_rack[step + 1])
            )
            / params.C_Rack
        )
        m.physics.add(
            m.t_cold[step + 1]
            == m.t_cold[step]
            + config.dt_seconds
            * (
                params.m_dot_air
                * params.kappa
                * params.c_p_air
                * (m.t_in[step] - m.t_cold[step + 1])
                - params.G_cold
                * (m.t_cold[step + 1] - params.T_out_Celsius)
            )
            / params.C_cAisle
        )
        m.physics.add(
            m.t_hot[step + 1]
            == m.t_hot[step]
            + config.dt_seconds
            * (
                params.m_dot_air
                * params.kappa
                * params.c_p_air
                * (m.t_rack[step + 1] - m.t_hot[step + 1])
            )
            / params.C_hAisle
        )

    prices = (
        horizon["optimisation_price_gbp_per_mwh"]
        if "optimisation_price_gbp_per_mwh" in horizon
        else horizon["settlement_price_gbp_per_mwh"]
    ).to_numpy(dtype=float)

    def grid_import(mod, step):
        return (
            mod.p_grid_it_kw[step]
            + mod.p_chiller_hvac_w[step] / 1000.0
            + mod.p_chiller_tes_w[step] / 1000.0
            + params.nominal_overhead_addition
            + mod.p_ups_ch_kw[step]
        )

    m.grid_import_kw = pyo.Expression(m.I, rule=grid_import)
    m.objective = pyo.Objective(
        expr=sum(
            config.dt_hours
            * m.grid_import_kw[step]
            * float(prices[step])
            / 1000.0
            for step in range(n_steps)
        ),
        sense=pyo.minimize,
    )

    solver_name, solver = _select_solver(config)
    started = time.perf_counter()
    results = solver.solve(m, tee=tee)
    runtime = time.perf_counter() - started
    termination = results.solver.termination_condition
    accepted = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.maxTimeLimit,
    }
    if termination not in accepted:
        raise RuntimeError(
            f"{solver_name} did not return an accepted solution: {termination}"
        )

    solved_initial = _state_from_model(m, 0)
    initial_residual = _max_state_difference(solved_initial, initial_state)
    next_state = _state_from_model(m, core_steps)
    core_boundary = timestamps[core_steps]

    rows: list[dict] = []
    for step in range(core_steps):
        flexible_processed = (
            float(horizon.at[step, "flexible_cpu"])
            if config.mode == "baseline"
            else sum(
                pyo.value(m.work_rate[cohort.cohort_id, step])
                for cohort in cohorts
                if (cohort.cohort_id, step) in m.X_INDEX
            )
        )
        state_start = _state_from_model(m, step)
        state_end = _state_from_model(m, step + 1)
        grid_kw = pyo.value(m.grid_import_kw[step])
        settlement_price = float(horizon.at[step, "settlement_price_gbp_per_mwh"])
        rows.append(
            {
                "timestamp_utc": _iso(timestamps[step]),
                "timestamp_local": _iso(pd.Timestamp(horizon.at[step, "timestamp_local"])),
                "local_date": str(horizon.at[step, "local_date"]),
                "source_period": int(horizon.at[step, "source_period"]),
                "settlement_price_gbp_per_mwh": settlement_price,
                "optimisation_price_gbp_per_mwh": float(prices[step]),
                "inflexible_cpu": float(horizon.at[step, "inflexible_cpu"]),
                "flexible_cpu_processed": flexible_processed,
                "total_cpu": pyo.value(m.total_cpu[step]),
                "p_it_total_kw": pyo.value(m.p_it_total_kw[step]),
                "p_grid_it_kw": pyo.value(m.p_grid_it_kw[step]),
                "p_ups_charge_kw": pyo.value(m.p_ups_ch_kw[step]),
                "p_ups_discharge_kw": pyo.value(m.p_ups_disch_kw[step]),
                "p_chiller_hvac_kw": pyo.value(m.p_chiller_hvac_w[step]) / 1000.0,
                "p_chiller_tes_kw": pyo.value(m.p_chiller_tes_w[step]) / 1000.0,
                "q_tes_charge_kw": pyo.value(m.q_ch_tes_w[step]) / 1000.0,
                "q_tes_discharge_kw": pyo.value(m.q_dis_tes_w[step]) / 1000.0,
                "grid_import_kw": grid_kw,
                "settlement_cost_gbp": grid_kw
                * settlement_price
                / 1000.0
                * config.dt_hours,
                **{f"state_start_{key}": value for key, value in state_start.to_dict().items()},
                **{f"state_end_{key}": value for key, value in state_end.to_dict().items()},
            }
        )
    committed = pd.DataFrame(rows)

    next_workload: list[WorkloadCohort] = []
    core_generated = [cohort for cohort in generated if cohort.arrival < core_boundary]
    opening_work = sum(cohort.remaining_cpu_hours for cohort in carried + core_generated)
    committed_work = 0.0
    for cohort in carried + core_generated:
        processed = sum(
            pyo.value(m.work_rate[cohort.cohort_id, step]) * config.dt_hours
            for step in allowed[cohort.cohort_id]
            if step < core_steps
        )
        committed_work += processed
        remaining = cohort.remaining_cpu_hours - processed
        if remaining > config.workload_tolerance_cpu_h:
            if cohort.latest_start < core_boundary:
                raise RuntimeError(
                    f"Workload deadline crossed with {remaining} CPU-h remaining: "
                    f"{cohort.cohort_id}"
                )
            next_workload.append(
                replace(cohort, remaining_cpu_hours=max(0.0, remaining))
            )
    closing_work = sum(cohort.remaining_cpu_hours for cohort in next_workload)
    workload_residual = opening_work - committed_work - closing_work

    max_ups_overlap = max(
        min(pyo.value(m.p_ups_ch_kw[step]), pyo.value(m.p_ups_disch_kw[step]))
        for step in range(n_steps)
    )
    max_tes_overlap_kw = max(
        min(pyo.value(m.q_ch_tes_w[step]), pyo.value(m.q_dis_tes_w[step]))
        / 1000.0
        for step in range(n_steps)
    )
    objective_recalculated = sum(
        config.dt_hours
        * pyo.value(m.grid_import_kw[step])
        * float(prices[step])
        / 1000.0
        for step in range(n_steps)
    )
    audits = {
        "initial_state_max_residual": initial_residual,
        "workload_conservation_residual_cpu_h": workload_residual,
        "opening_workload_cpu_h": opening_work,
        "committed_workload_cpu_h": committed_work,
        "closing_workload_cpu_h": closing_work,
        "max_ups_charge_discharge_overlap_kw": max_ups_overlap,
        "max_tes_charge_discharge_overlap_kw": max_tes_overlap_kw,
        "objective_reconciliation_gbp": pyo.value(m.objective)
        - objective_recalculated,
        "committed_settlement_cost_gbp": float(committed["settlement_cost_gbp"].sum()),
        "committed_grid_energy_kwh": float(
            committed["grid_import_kw"].sum() * config.dt_hours
        ),
    }
    if initial_residual > config.state_tolerance:
        raise RuntimeError(f"Initial state residual {initial_residual} exceeds tolerance")
    if abs(workload_residual) > config.workload_tolerance_cpu_h:
        raise RuntimeError(
            f"Workload conservation residual {workload_residual} exceeds tolerance"
        )
    if max_ups_overlap > config.flow_tolerance_kw:
        raise RuntimeError("UPS charged and discharged simultaneously")
    if max_tes_overlap_kw > config.flow_tolerance_kw:
        raise RuntimeError("TES charged and discharged simultaneously")

    lower_bound = getattr(results.problem, "lower_bound", None)
    upper_bound = getattr(results.problem, "upper_bound", None)
    relative_gap = None
    if lower_bound is not None and upper_bound is not None:
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            relative_gap = abs(upper_bound - lower_bound) / max(1.0, abs(upper_bound))
    solver_metadata = {
        "name": solver_name,
        "termination_condition": str(termination),
        "status": str(results.solver.status),
        "runtime_s": runtime,
        "objective_gbp": pyo.value(m.objective),
        "lower_bound_gbp": lower_bound,
        "upper_bound_gbp": upper_bound,
        "relative_gap": relative_gap,
    }
    return HorizonResult(
        committed=committed,
        next_state=next_state,
        next_workload=next_workload,
        solver=solver_metadata,
        audits=audits,
    )
