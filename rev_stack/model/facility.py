"""Facility model: Paper 1's whole-facility MILP, scaled to 10 MW.

The physical model is reused verbatim from the repo root (constraints.py);
only the parameters are scaled. Replication scaling (x10 racks) multiplies
every extensive quantity (powers, energies, heat capacities, conductances,
air mass flow) by the scale factor and leaves intensive quantities (COP,
efficiencies, temperatures, CPU fractions) unchanged, so the temperature
dynamics are identical to the validated 1 MW case.
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from . import config  # noqa: F401  (side effect: repo root on sys.path)
from inputs.parameters_optimisation import ModelParameters
from constraints import (add_it_and_job_constraints, add_ups_constraints,
                         add_power_balance_constraints, add_cooling_constraints)

CYCLE_TES_ENERGY = True

# Extensive attributes multiplied by the scale factor.
_EXTENSIVE = [
    # IT & overhead
    "idle_power_kw", "max_power_kw", "nominal_overhead_addition",
    # UPS
    "e_nom_kwh", "e_start_kwh", "p_max_ch_kw", "p_max_disch_kw",
    "p_min_ch_kw", "p_min_disch_kw",
    # Cooling / thermal (replication: capacities, conductances, flows all x N)
    "C_IT", "G_conv", "C_Rack", "C_cAisle", "C_hAisle", "G_cold", "m_dot_air",
    "P_IT_heat_source", "P_chiller_max", "P_HVAC_ramp",
    # TES
    "TES_kwh_cap", "TES_w_discharge_max", "TES_w_charge_max",
    "TES_initial_charge_kWh", "E_TES_min_kWh",
]


def scaled_parameters(scale: float = config.SCALE,
                      soc_floor: float = None) -> ModelParameters:
    """Paper 1 parameters scaled by replication to `scale` x 1 MW."""
    params = ModelParameters()
    for attr in _EXTENSIVE:
        setattr(params, attr, getattr(params, attr) * scale)
    if soc_floor is not None:
        params.soc_min = soc_floor        # UPS resilience-floor sensitivity
    params.e_min_kwh = params.soc_min * params.e_nom_kwh
    params.e_max_kwh = params.soc_max * params.e_nom_kwh
    params.e_start_kwh = params.e_max_kwh
    params.TES_capacity_kWh = params.TES_kwh_cap
    params.scale = scale
    return params


def load_facility_data(params: ModelParameters, da_price_slots: np.ndarray) -> dict:
    """Paper 1's input pipeline (optimisation.load_and_prepare_data) with the
    synthetic ToU tariff replaced by a market day-ahead price array."""
    load_profiles = pd.read_csv(config.FACILITY_INPUT_DIR / "load_profiles.csv",
                                index_col="time_slot")
    shiftability = pd.read_csv(config.FACILITY_INPUT_DIR / "shiftability_profile.csv",
                               index_col="time_slot")
    shiftability.columns = shiftability.columns.astype(int)
    shift_dict = shiftability.stack().to_dict()

    inflex = load_profiles["inflexible_load"].values
    flex = load_profiles["flexible_load"].values
    n_ext, n_day = config.N_SLOTS_EXT, config.N_SLOTS_DAY

    data = {
        "inflexibleLoadProfile_TEXT": np.insert(inflex[:n_ext], 0, 0),
        "flexibleLoadProfile_TEXT": np.insert(flex[:n_ext], 0, 0),
        "shiftabilityProfile": {(t, k): shift_dict.get((t, k), 0)
                                for t in range(1, n_ext + 1)
                                for k in range(1, 5)},
    }
    data["Rt"] = np.insert(flex[:n_day], 0, 0) * params.dt_hours
    flex_ext = np.insert(flex[:n_ext], 0, 0)
    data["Pt_IT_nom_TEXT"] = (params.idle_power_kw
                              + (params.max_power_kw - params.idle_power_kw)
                              * (data["inflexibleLoadProfile_TEXT"] + flex_ext))
    data["electricity_price"] = np.asarray(da_price_slots, dtype=float)
    return data


def build_facility_model(params: ModelParameters, data: dict) -> pyo.ConcreteModel:
    """Variables + the four Paper 1 constraint groups (no objective).

    Mirrors optimisation.build_model so that the facility physics stay
    byte-identical to the published model; the market layer and objective
    are added separately by stack_model.add_market_layer.
    """
    m = pyo.ConcreteModel(name="DC_Revenue_Stacking")
    m.TEXT_SLOTS = pyo.Set(initialize=params.TEXT_SLOTS)
    m.T_SLOTS = pyo.Set(initialize=params.T_SLOTS)
    m.K_TRANCHES = pyo.Set(initialize=params.K_TRANCHES)

    m.total_cpu = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.max_cpu_usage), initialize=0)
    m.p_grid_it_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_it_total_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)

    m.p_ups_ch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_ups_disch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.e_ups_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.e_min_kwh, params.e_max_kwh),
                          initialize=params.e_start_kwh)
    m.z_ch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary, initialize=0)
    m.z_disch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary, initialize=0)

    ut_ks_idx = [(t, k, s) for t in m.T_SLOTS for k in m.K_TRANCHES
                 for s in m.TEXT_SLOTS
                 if s >= t and s <= t + params.tranche_max_delay[k]]
    m.ut_ks_idx = pyo.Set(initialize=ut_ks_idx)
    m.ut_ks = pyo.Var(m.ut_ks_idx, within=pyo.NonNegativeReals, initialize=0)

    m.t_it = pyo.Var(m.TEXT_SLOTS, bounds=(18, 60), initialize=25)
    m.t_rack = pyo.Var(m.TEXT_SLOTS, bounds=(18, 40), initialize=25)
    m.t_cold_aisle = pyo.Var(m.TEXT_SLOTS,
                             bounds=(18, params.T_cAisle_upper_limit_Celsius),
                             initialize=20)
    m.t_hot_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(18, 40), initialize=30)
    m.e_tes_kwh = pyo.Var(m.TEXT_SLOTS,
                          bounds=(params.E_TES_min_kWh, params.TES_capacity_kWh),
                          initialize=params.TES_initial_charge_kWh)
    m.p_chiller_hvac_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_chiller_tes_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.q_cool_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.q_ch_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_charge_max), initialize=0)
    m.q_dis_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_discharge_max), initialize=0)
    m.t_in = pyo.Var(m.TEXT_SLOTS, bounds=(14, 30), initialize=20)

    add_it_and_job_constraints(m, params, data)
    add_ups_constraints(m, params)
    add_power_balance_constraints(m, params)
    add_cooling_constraints(m, params, CYCLE_TES_ENERGY)

    # Total grid draw expression, Eq. (1) of the paper - used everywhere.
    def grid_power(mod, s):
        return (mod.p_grid_it_kw[s]
                + mod.p_chiller_hvac_w[s] / 1000.0
                + mod.p_chiller_tes_w[s] / 1000.0
                + params.nominal_overhead_addition
                + mod.p_ups_ch_kw[s])
    m.P_grid = pyo.Expression(m.TEXT_SLOTS, rule=grid_power)
    return m
