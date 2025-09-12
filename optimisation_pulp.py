# optimisation.py  — PuLP version
import datetime
import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from types import SimpleNamespace

import pulp as pl  # <-- switched to PuLP

from inputs.parameters_optimisation import ModelParameters, generate_tariff
from plotting_and_saving.nom_opt_charts import gen_charts

# --- Path Configuration ------------------------------------------------------
DATA_DIR_INPUTS_1 = pathlib.Path("static/data/inputs")
DATA_DIR_INPUTS_2 = pathlib.Path("static/data/nominal_outputs")
DATA_DIR_OUTPUTS = pathlib.Path("static/data/optimisation_outputs")
IMAGE_DIR = pathlib.Path("static/images/optimisation_outputs")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR_INPUTS_1.mkdir(parents=True, exist_ok=True)
DATA_DIR_INPUTS_2.mkdir(parents=True, exist_ok=True)
DATA_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

# --- Model Configuration -----------------------------------------------------
CYCLE_TES_ENERGY = True

MIP_REL_GAP = 0.005
# Small helper that behaves like your old index sets
class IndexSet(list):
    def first(self):
        return self[0]
    def last(self):
        return self[-1]


def build_model(params: ModelParameters, data: dict, linear: bool = False):
    """
    Builds the PuLP optimization model, defining all variables, constraints, and the objective.
    Returns a SimpleNamespace 'm' that holds:
        - prob: the LpProblem
        - index sets on m.TEXT_SLOTS, m.T_SLOTS, m.K_TRANCHES
        - all decision variables as dicts keyed by indices
        - any helper sets (e.g., ut_ks_idx)
    """
    # Problem
    prob = pl.LpProblem("DC_Cost_Optimization", pl.LpMinimize)

    # --- Index sets (preserve names & order) ---
    TEXT_SLOTS = IndexSet(sorted(list(params.TEXT_SLOTS)))
    T_SLOTS = IndexSet(sorted(list(params.T_SLOTS)))
    K_TRANCHES = IndexSet(sorted(list(params.K_TRANCHES)))

    # --- Variables -----------------------------------------------------------
    # IT & Grid (kW)
    total_cpu = pl.LpVariable.dicts("total_cpu", TEXT_SLOTS, lowBound=0, upBound=params.max_cpu_usage)
    p_grid_it_kw = pl.LpVariable.dicts("p_grid_it_kw", TEXT_SLOTS, lowBound=0)
    p_grid_od_kw = pl.LpVariable.dicts("p_grid_od_kw", TEXT_SLOTS, lowBound=0)
    p_it_total_kw = pl.LpVariable.dicts("p_it_total_kw", TEXT_SLOTS, lowBound=0)

    # UPS
    p_ups_ch_kw = pl.LpVariable.dicts("p_ups_ch_kw", TEXT_SLOTS, lowBound=0)
    p_ups_disch_kw = pl.LpVariable.dicts("p_ups_disch_kw", TEXT_SLOTS, lowBound=0)
    e_ups_kwh = pl.LpVariable.dicts("e_ups_kwh", TEXT_SLOTS, lowBound=params.e_min_kwh, upBound=params.e_max_kwh)
    z_ch = pl.LpVariable.dicts("z_ch", TEXT_SLOTS, lowBound=0, upBound=1, cat="Binary")
    z_disch = pl.LpVariable.dicts("z_disch", TEXT_SLOTS, lowBound=0, upBound=1, cat="Binary")

    # Job scheduling (u_{t,k,s})
    ut_ks_idx = [(t, k, s) for t in T_SLOTS for k in K_TRANCHES
                 for s in TEXT_SLOTS if s >= t and s <= t + params.tranche_max_delay[k]]
    ut_ks = {idx: pl.LpVariable(f"ut_ks_{idx[0]}_{idx[1]}_{idx[2]}", lowBound=0)
             for idx in ut_ks_idx}

    # Cooling variables
    t_it = pl.LpVariable.dicts("t_it", TEXT_SLOTS, lowBound=14, upBound=60)
    t_rack = pl.LpVariable.dicts("t_rack", TEXT_SLOTS, lowBound=14, upBound=40)
    t_cold_aisle = pl.LpVariable.dicts("t_cold_aisle", TEXT_SLOTS, lowBound=18, upBound=params.T_cAisle_upper_limit_Celsius)
    t_hot_aisle = pl.LpVariable.dicts("t_hot_aisle", TEXT_SLOTS, lowBound=14, upBound=40)
    e_tes_kwh = pl.LpVariable.dicts("e_tes_kwh", TEXT_SLOTS, lowBound=params.E_TES_min_kWh, upBound=params.TES_capacity_kWh)

    # Electrical chiller power (W)
    p_chiller_hvac_w = pl.LpVariable.dicts("p_chiller_hvac_w", TEXT_SLOTS, lowBound=0)
    p_chiller_tes_w  = pl.LpVariable.dicts("p_chiller_tes_w", TEXT_SLOTS, lowBound=0)
    # Thermal powers (W)
    q_cool_w    = pl.LpVariable.dicts("q_cool_w", TEXT_SLOTS, lowBound=0)
    q_ch_tes_w  = pl.LpVariable.dicts("q_ch_tes_w", TEXT_SLOTS, lowBound=0, upBound=params.TES_w_charge_max)
    q_dis_tes_w = pl.LpVariable.dicts("q_dis_tes_w", TEXT_SLOTS, lowBound=0, upBound=params.TES_w_discharge_max)
    t_in        = pl.LpVariable.dicts("t_in", TEXT_SLOTS, lowBound=14, upBound=30)

    # --- Constraints ---------------------------------------------------------
    if not linear:
        # Piecewise (x^1.32) approximation via convex-combination with adjacent-segment binaries
        num_pw_points = 11
        pw_x = [i / (num_pw_points - 1) for i in range(num_pw_points)]
        pw_y = [x ** 1.32 for x in pw_x]
        PW_POINTS = list(range(num_pw_points))
        SEGMENTS = list(range(num_pw_points - 1))  # segments between points

        # weights w[s,i] and segment binaries delta[s,seg]
        w = {(s, i): pl.LpVariable(f"w_{s}_{i}", lowBound=0) for s in TEXT_SLOTS for i in PW_POINTS}
        delta = {(s, seg): pl.LpVariable(f"seg_{s}_{seg}", lowBound=0, upBound=1, cat="Binary")
                 for s in TEXT_SLOTS for seg in SEGMENTS}
        cpu_power_factor = pl.LpVariable.dicts("cpu_power_factor", TEXT_SLOTS, lowBound=0)

        add_it_and_job_constraints_pwl(prob, TEXT_SLOTS, T_SLOTS, K_TRANCHES,
                                       ut_ks_idx, ut_ks, total_cpu,
                                       w, delta, PW_POINTS, SEGMENTS, pw_x, pw_y,
                                       params, data, cpu_power_factor, p_it_total_kw)
    else:
        # Use simple linear relationship
        add_it_and_job_constraints_linear(prob, TEXT_SLOTS, T_SLOTS, K_TRANCHES,
                                          ut_ks_idx, ut_ks, total_cpu,
                                          params, data, p_it_total_kw)


    add_ups_constraints(prob, TEXT_SLOTS, e_ups_kwh, p_ups_ch_kw, p_ups_disch_kw, z_ch, z_disch, params)

    add_power_balance_constraints(prob, TEXT_SLOTS, p_it_total_kw, p_grid_it_kw, p_ups_disch_kw,
                                  p_grid_od_kw, params)

    add_cooling_constraints(prob, TEXT_SLOTS, t_it, t_rack, t_cold_aisle, t_hot_aisle, e_tes_kwh,
                            p_chiller_hvac_w, p_chiller_tes_w, q_cool_w, q_ch_tes_w, q_dis_tes_w, t_in,
                            p_it_total_kw, params, CYCLE_TES_ENERGY)

    # --- Objective -----------------------------------------------------------
    # Price input is GBP/MWh; convert power (kW) × hours × price/1000 to GBP
    obj = []
    for s in TEXT_SLOTS:
        step_cost = params.dt_hours * (
            p_grid_it_kw[s] +
            (p_chiller_hvac_w[s] / 1000.0) +
            (p_chiller_tes_w[s] / 1000.0) +
            p_grid_od_kw[s] +
            p_ups_ch_kw[s]
        ) * (data['electricity_price'][s] / 1000.0)
        obj.append(step_cost)
    prob += pl.lpSum(obj)

    # Pack everything into a namespace for downstream functions
    m_dict = {
        "prob": prob, "TEXT_SLOTS": TEXT_SLOTS, "T_SLOTS": T_SLOTS, "K_TRANCHES": K_TRANCHES,
        "ut_ks_idx": ut_ks_idx, "total_cpu": total_cpu, "p_grid_it_kw": p_grid_it_kw,
        "p_grid_od_kw": p_grid_od_kw, "p_it_total_kw": p_it_total_kw, "p_ups_ch_kw": p_ups_ch_kw,
        "p_ups_disch_kw": p_ups_disch_kw, "e_ups_kwh": e_ups_kwh, "z_ch": z_ch, "z_disch": z_disch,
        "ut_ks": ut_ks, "t_it": t_it, "t_rack": t_rack, "t_cold_aisle": t_cold_aisle,
        "t_hot_aisle": t_hot_aisle, "e_tes_kwh": e_tes_kwh, "p_chiller_hvac_w": p_chiller_hvac_w,
        "p_chiller_tes_w": p_chiller_tes_w, "q_cool_w": q_cool_w, "q_ch_tes_w": q_ch_tes_w,
        "q_dis_tes_w": q_dis_tes_w, "t_in": t_in
    }

    if not linear:
        m_dict.update({
            "w": w, "delta": delta, "PW_POINTS": PW_POINTS, "SEGMENTS": SEGMENTS,
            "pw_x": pw_x, "pw_y": pw_y, "cpu_power_factor": cpu_power_factor
        })
    m = SimpleNamespace(**m_dict)
    return m


def add_it_and_job_constraints_pwl(prob, TEXT_SLOTS, T_SLOTS, K_TRANCHES,
                                   ut_ks_idx, ut_ks, total_cpu,
                                   w, delta, PW_POINTS, SEGMENTS, pw_x, pw_y,
                                   params, data, cpu_power_factor, p_it_total_kw):
    # --- Job completion (per tranche at original time t) ---
    for t in T_SLOTS:
        for k in K_TRANCHES:
            expr = []
            for s in TEXT_SLOTS:
                if (t, k, s) in ut_ks_idx:
                    expr.append(ut_ks[(t, k, s)] * params.dt_hours)
            rhs = data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0)
            prob += pl.lpSum(expr) == rhs

    # --- Piecewise linearisation x^1.32 via convex combination + adjacent segment selection ---
    # For each s: sum_i w[s,i] = 1; limit weights to one adjacent segment:
    #   sum_seg delta[s,seg] = 1
    #   w[s,0]   <= delta[s,0]
    #   w[s,n-1] <= delta[s,n-2]
    #   w[s,i]   <= delta[s,i-1] + delta[s,i] for i = 1..n-2
    for s in TEXT_SLOTS:
        n = len(PW_POINTS)               # number of points
        prob += pl.lpSum(w[(s, i)] for i in PW_POINTS) == 1
        prob += pl.lpSum(delta[(s, seg)] for seg in SEGMENTS) == 1  # exactly one segment active

        # adjacency: only weights in the active segment's two endpoints can be nonzero
        prob += w[(s, 0)] <= delta[(s, 0)]
        for i in range(1, n - 1):  # i = 1 .. n-2
            prob += w[(s, i)] <= delta[(s, i - 1)] + delta[(s, i)]
        prob += w[(s, n - 1)] <= delta[(s, n - 2)]  # last point bound uses last segment (n-2)

        # Flexible usage arriving at time s
        flex_use_terms = [ut_ks[idx] for idx in ut_ks_idx if idx[2] == s]
        flexible_usage = pl.lpSum(flex_use_terms) if flex_use_terms else 0.0

        # Total CPU equals inflexible + flexible, and equals convex combination over pw_x
        base_cpu = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage
        prob += total_cpu[s] == base_cpu
        prob += total_cpu[s] == pl.lpSum(pw_x[i] * w[(s, i)] for i in PW_POINTS)

        # Define cpu_power_factor from pw_y
        prob += cpu_power_factor[s] == pl.lpSum(pw_y[i] * w[(s, i)] for i in PW_POINTS)

        # Link IT power to cpu_power_factor
        prob += p_it_total_kw[s] == params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * cpu_power_factor[s]


def add_it_and_job_constraints_linear(prob, TEXT_SLOTS, T_SLOTS, K_TRANCHES,
                                      ut_ks_idx, ut_ks, total_cpu,
                                      params, data, p_it_total_kw):
    # --- Job completion (per tranche at original time t) ---
    for t in T_SLOTS:
        for k in K_TRANCHES:
            expr = []
            for s in TEXT_SLOTS:
                if (t, k, s) in ut_ks_idx:
                    expr.append(ut_ks[(t, k, s)] * params.dt_hours)
            rhs = data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0)
            prob += pl.lpSum(expr) == rhs

    # --- Total CPU and linear power relationship ---
    for s in TEXT_SLOTS:
        # Flexible usage arriving at time s
        flex_use_terms = [ut_ks[idx] for idx in ut_ks_idx if idx[2] == s]
        flexible_usage = pl.lpSum(flex_use_terms) if flex_use_terms else 0.0

        # Total CPU equals inflexible + flexible
        prob += total_cpu[s] == data['inflexibleLoadProfile_TEXT'][s] + flexible_usage

        # Linear relationship between total_cpu and p_it_total_kw
        prob += p_it_total_kw[s] == params.idle_power_kw + \
                (params.max_power_kw - params.idle_power_kw) * total_cpu[s]


def add_ups_constraints(prob, TEXT_SLOTS, e_ups_kwh, p_ups_ch_kw, p_ups_disch_kw, z_ch, z_disch, params):
    first = TEXT_SLOTS.first()
    last = TEXT_SLOTS.last()

    for s in TEXT_SLOTS:
        prev_energy = params.e_start_kwh if s == first else e_ups_kwh[s - 1]
        charge = params.eta_ch * p_ups_ch_kw[s] * params.dt_hours
        discharge = (p_ups_disch_kw[s] / params.eta_disch) * params.dt_hours
        prob += e_ups_kwh[s] == prev_energy + charge - discharge

        prob += p_ups_ch_kw[s] <= z_ch[s] * params.p_max_ch_kw
        prob += p_ups_ch_kw[s] >= z_ch[s] * params.p_min_ch_kw
        prob += p_ups_disch_kw[s] <= z_disch[s] * params.p_max_disch_kw
        prob += p_ups_disch_kw[s] >= z_disch[s] * params.p_min_disch_kw
        prob += z_ch[s] + z_disch[s] <= 1

    prob += e_ups_kwh[last] == params.e_start_kwh


def add_power_balance_constraints(prob, TEXT_SLOTS, p_it_total_kw, p_grid_it_kw, p_ups_disch_kw,
                                  p_grid_od_kw, params):
    for s in TEXT_SLOTS:
        prob += p_it_total_kw[s] == p_grid_it_kw[s] + p_ups_disch_kw[s]
        prob += p_grid_od_kw[s] == p_it_total_kw[s] * params.nominal_overhead_factor


def add_cooling_constraints(prob, TEXT_SLOTS, t_it, t_rack, t_cold_aisle, t_hot_aisle, e_tes_kwh,
                            p_chiller_hvac_w, p_chiller_tes_w, q_cool_w, q_ch_tes_w, q_dis_tes_w, t_in,
                            p_it_total_kw, params, CYCLE_TES_ENERGY):
    first = TEXT_SLOTS.first()
    last = TEXT_SLOTS.last()
    mcp = params.m_dot_air * params.c_p_air

    # Initial condition lower bounds
    prob += t_it[first]        >= params.T_IT_initial_Celsius
    prob += t_rack[first]      >= params.T_Rack_initial_Celsius
    prob += t_cold_aisle[first] >= params.T_cAisle_initial
    prob += t_hot_aisle[first]  >= params.T_hAisle_initial
    prob += e_tes_kwh[first]   >= params.TES_initial_charge_kWh

    # Average "anchor" for slot 1 (as in original)
    if len(TEXT_SLOTS) > 1:
        denom = float(len(TEXT_SLOTS) - 1)
        rhs_hvac = (pl.lpSum(p_chiller_hvac_w[k] for k in TEXT_SLOTS if k > first)) / denom
        rhs_tes  = (pl.lpSum(p_chiller_tes_w[k]  for k in TEXT_SLOTS if k > first)) / denom
        prob += p_chiller_hvac_w[first] == rhs_hvac
        prob += p_chiller_tes_w[first]  == rhs_tes

    # Dynamics for t > first
    for t in TEXT_SLOTS:
        if t == first:
            continue

        # Thermal balances (linear)
        prob += q_cool_w[t] == p_chiller_hvac_w[t] * params.COP_HVAC + q_dis_tes_w[t]
        prob += q_ch_tes_w[t] == p_chiller_tes_w[t] * params.COP_HVAC
        prob += t_in[t] == t_hot_aisle[t] - (q_cool_w[t] / mcp)
        prob += q_cool_w[t] <= (t_hot_aisle[t] - params.T_cAisle_lower_limit_Celsius) * mcp

        it_heat_watts = p_it_total_kw[t] * 1000.0

        prob += t_it[t] == t_it[t - 1] + params.dt_seconds * (
            (it_heat_watts - params.G_conv * (t_it[t - 1] - t_rack[t])) / params.C_IT
        )
        prob += t_rack[t] == t_rack[t - 1] + params.dt_seconds * (
            (params.m_dot_air * params.kappa * params.c_p_air * (t_cold_aisle[t] - t_rack[t - 1]) +
             params.G_conv * (t_it[t - 1] - t_rack[t - 1])) / params.C_Rack
        )
        prob += t_cold_aisle[t] == t_cold_aisle[t - 1] + params.dt_seconds * (
            (params.m_dot_air * params.kappa * params.c_p_air * (t_in[t] - t_cold_aisle[t - 1]) -
             params.G_cold * (t_cold_aisle[t - 1] - params.T_out_Celsius)) / params.C_cAisle
        )
        prob += t_hot_aisle[t] == t_hot_aisle[t - 1] + params.dt_seconds * (
            (params.m_dot_air * params.kappa * params.c_p_air * (t_rack[t] - t_hot_aisle[t - 1])) / params.C_hAisle
        )

        dE_tes_kwh = (q_ch_tes_w[t] * params.TES_charge_efficiency -
                      q_dis_tes_w[t] / params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        prob += e_tes_kwh[t] == e_tes_kwh[t - 1] + dE_tes_kwh

        prob += q_dis_tes_w[t] - q_dis_tes_w[t - 1] <= params.TES_p_dis_ramp
        prob += q_ch_tes_w[t]  - q_ch_tes_w[t - 1]  <= params.TES_p_ch_ramp
        prob += p_chiller_tes_w[t] + p_chiller_hvac_w[t] <= params.P_chiller_max
        prob += q_cool_w[t] >= it_heat_watts

    if CYCLE_TES_ENERGY:
        prob += e_tes_kwh[last] == e_tes_kwh[first]


def load_and_prepare_data(params: ModelParameters):
    """
    Loads input data from CSV files and prepares it for the optimization model.
    """
    try:
        load_profiles_df = pd.read_csv(DATA_DIR_INPUTS_1 / "load_profiles.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR_INPUTS_1 / "shiftability_profile.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profiles.csv' and 'shiftability_profile.csv' are in {DATA_DIR_INPUTS_1}")
        raise e

    inflexible = load_profiles_df['inflexible_load']
    base_flex = load_profiles_df['flexible_load']

    base_flex_t = base_flex.loc[list(params.T_SLOTS)]

    shiftability_df.columns = shiftability_df.columns.astype(int)
    shiftability = shiftability_df.stack().to_dict()

    RESAMPLE_FACTOR = int(900 / params.dt_seconds)

    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'flexibleLoadProfile_TEXT':   np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0),
        'shiftabilityProfile':        resample_shiftability_profile(shiftability, RESAMPLE_FACTOR)
    }

    baseFlexibleLoadProfile_T = np.insert(np.array(np.repeat(base_flex_t.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0)
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours

    baseFlexibleLoadProfile_TEXT = np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0)

    len_inflexible = len(data['inflexibleLoadProfile_TEXT'])
    len_flexible = len(baseFlexibleLoadProfile_TEXT)
    min_len = min(len_inflexible, len_flexible)

    data['Pt_IT_nom_TEXT'] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (
        data['inflexibleLoadProfile_TEXT'][:min_len] + baseFlexibleLoadProfile_TEXT[:min_len]
    )

    data['electricity_price'] = generate_tariff(params.num_steps_extended, params.dt_seconds)

    return data


def resample_shiftability_profile(shiftability_profile, repeats):
    extended_data = {}
    counter = 1
    for i in range(1, 97):
        for _ in range(repeats):
            for j in range(1, 5):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data


def _val(x):
    """PuLP value getter for scalars and returns 0.0 if None."""
    v = pl.value(x)
    return float(v) if v is not None else 0.0


def post_process_results(m, params: ModelParameters, data: dict):
    """
    Extracts results from a solved PuLP model into a DataFrame.
    """
    tranche_map = params.tranche_max_delay
    flexible_load_details = []

    for (t, k, s) in m.ut_ks_idx:
        val = _val(m.ut_ks[(t, k, s)])
        if val is not None and val > 1e-6:
            flexible_load_details.append({
                'processing_slot': s,
                'original_slot': t,
                'tranche': k,
                'cpu_load': val,
                'shiftability': tranche_map[k] - (s - t)
            })

    flex_load_origin_df = pd.DataFrame(flexible_load_details)
    if not flex_load_origin_df.empty:
        flex_load_origin_df = flex_load_origin_df.sort_values(by=['processing_slot', 'tranche']).reset_index(drop=True)

    flex_filtered = flex_load_origin_df[flex_load_origin_df['shiftability'] > 0]
    new_flexible_load_per_slot = flex_filtered.groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)
    new_inflexible_load_per_slot = flex_load_origin_df[flex_load_origin_df['shiftability'] == 0].groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)
    new_inflexible_load = data['inflexibleLoadProfile_TEXT'][1:109] + new_inflexible_load_per_slot.values

    load_profiles_df = pd.DataFrame({
        'inflexible_load': new_inflexible_load,
        'flexible_load': new_flexible_load_per_slot.values
    }, index=range(1, 109))
    load_profiles_df.index.name = 'time_slot'
    load_profiles_df.to_csv(DATA_DIR_OUTPUTS / 'load_profiles_opt.csv')

    def generate_shiftability_profile(flex_df, num_timesteps, num_tranches):
        grouped = flex_df.groupby(['processing_slot', 'shiftability'])['cpu_load'].sum().reset_index()
        shiftabilityProfile_data = {}
        for t in range(1, num_timesteps + 1):
            for k in range(1, num_tranches + 1):
                key = (t, k)
                value = grouped[(grouped['processing_slot'] == t) & (grouped['shiftability'] == k)]['cpu_load']
                shiftabilityProfile_data[key] = value.sum() if not value.empty else 0.0
        return shiftabilityProfile_data

    data['shiftabilityProfile'] = generate_shiftability_profile(flex_filtered, 108, 12)
    shiftability_df = pd.Series(data['shiftabilityProfile']).unstack()
    shiftability_df.index.name = 'time_slot'
    shiftability_df.columns.name = 'category'
    shiftability_df.to_csv(DATA_DIR_OUTPUTS / 'shiftability_profile_opt.csv')

    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'Total_Optimized_Cost': _val(m.prob.objective),
        'P_IT_Total_kW':          [_val(m.p_it_total_kw[s]) for s in params.TEXT_SLOTS],
        'P_Grid_IT_kW':           [_val(m.p_grid_it_kw[s]) for s in params.TEXT_SLOTS],
        'P_Chiller_HVAC_Watts':   [_val(m.p_chiller_hvac_w[s]) for s in params.TEXT_SLOTS],
        'P_Chiller_TES_Watts':    [_val(m.p_chiller_tes_w[s]) for s in params.TEXT_SLOTS],
        'P_Grid_Other_kW':        [_val(m.p_grid_od_kw[s]) for s in params.TEXT_SLOTS],
        'P_UPS_Charge_kW':        [_val(m.p_ups_ch_kw[s]) for s in params.TEXT_SLOTS],
        'P_UPS_Discharge_kW':     [_val(m.p_ups_disch_kw[s]) for s in params.TEXT_SLOTS],
        'E_UPS_kWh':              [_val(m.e_ups_kwh[s]) for s in params.TEXT_SLOTS],
        'T_IT_Celsius':           [_val(m.t_it[s]) for s in params.TEXT_SLOTS],
        'T_Rack_Celsius':         [_val(m.t_rack[s]) for s in params.TEXT_SLOTS],
        'T_ColdAisle_Celsius':    [_val(m.t_cold_aisle[s]) for s in params.TEXT_SLOTS],
        'T_HotAisle_Celsius':     [_val(m.t_hot_aisle[s]) for s in params.TEXT_SLOTS],
        'E_TES_kWh':              [_val(m.e_tes_kwh[s]) for s in params.TEXT_SLOTS],
        'Q_Cool_Total_Watts':     [_val(m.q_cool_w[s]) for s in params.TEXT_SLOTS],
        'Q_Charge_TES_Watts':     [_val(m.q_ch_tes_w[s]) for s in params.TEXT_SLOTS],
        'Q_Discharge_TES_Watts':  [_val(m.q_dis_tes_w[s]) for s in params.TEXT_SLOTS],
        'Total_CPU_Load':         [_val(m.total_cpu[s]) for s in params.TEXT_SLOTS],
    }

    results['P_Chiller_HVAC_kW'] = [p / 1000.0 for p in results['P_Chiller_HVAC_Watts']]
    results['P_Chiller_TES_KW']  = [p / 1000.0 for p in results['P_Chiller_TES_Watts']]
    results['P_Grid_Cooling_kW'] = [h + t for h, t in zip(results['P_Chiller_HVAC_kW'], results['P_Chiller_TES_KW'])]
    results['P_Total_kW'] = [
        results['P_Grid_IT_kW'][i - 1] + results['P_Grid_Cooling_kW'][i - 1] +
        results['P_Grid_Other_kW'][i - 1] + results['P_UPS_Charge_kW'][i - 1]
        for i in params.TEXT_SLOTS
    ]
    results['Optimized_Cost_per_Step'] = [
        params.dt_hours * results['P_Total_kW'][i - 1] * (data['electricity_price'][i] / 1000.0)
        for i in params.TEXT_SLOTS
    ]

    results['P_IT_Nominal'] = [data['Pt_IT_nom_TEXT'][s] for s in params.TEXT_SLOTS]
    results['Price_GBP_per_MWh'] = [data['electricity_price'][s] for s in params.TEXT_SLOTS]
    results['Inflexible_Load_CPU_Nom'] = [data['inflexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]
    results['Flexible_Load_CPU_Nom'] = [data['flexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]

    total_cpu_usage = results['Total_CPU_Load']
    inflexible_cpu_usage = results['Inflexible_Load_CPU_Nom']
    results['Inflexible_Load_CPU_Opt'] = inflexible_cpu_usage
    results['Flexible_Load_CPU_Opt'] = [(total - inflexible) for total, inflexible in zip(total_cpu_usage, inflexible_cpu_usage)]

    df = pd.DataFrame(results)
    df['Optimized_Cost'] = df['Optimized_Cost_per_Step'].cumsum()

    try:
        df_nominal = pd.read_csv(DATA_DIR_INPUTS_2 / "nominal_case_results.csv")
        df['Nominal_Cost'] = df_nominal['Nominal_Cost'][:len(df)].values
        df['P_Total_kW_Nominal'] = df_nominal['P_Total_kW'][:len(df)].values
    except Exception:
        print(f"Warning: Could not load or align '{DATA_DIR_INPUTS_2 / 'nominal_case_results.csv'}'. Nominal cost set to 0.")
        df['Nominal_Cost'] = 0
        df['P_Total_kW_Nominal'] = 0

    return df, flex_load_origin_df


def print_summary(params, results_df: pd.DataFrame):
    for t in range(96, 108):
        cost_diff_in_extension = results_df['Optimized_Cost_per_Step'].iloc[t] - results_df['Nominal_Cost'].iloc[t]

    print(f"Cost difference in extension period (slots 97-108): {cost_diff_in_extension:,.2f} GBP")
    nominal_cost = results_df['Nominal_Cost'].iloc[:96].sum()
    optimized_cost = results_df['Optimized_Cost_per_Step'].iloc[:96].sum()
    optimized_cost += cost_diff_in_extension
    cost_saving_abs = nominal_cost - optimized_cost if nominal_cost > 0 else 0
    cost_saving_rel = (cost_saving_abs / nominal_cost) * 100 if nominal_cost > 0 else 0

    print("\n" + "="*50)
    print("--- Optimization Results Summary ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print(f"Baseline (Nominal) Cost: {nominal_cost:,.2f} GBP")
    print(f"Absolute Cost Saving: {cost_saving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {cost_saving_rel:.2f} %")
    print("="*50 + "\n")


def create_and_save_charts(df: pd.DataFrame, flex_load_origin_df: pd.DataFrame, data: dict, params: 'ModelParameters'):
    print("Generating and saving combined chart with consistent grid...")
    plt.style.use('seaborn-v0_8-whitegrid')
    time_slots_ext = df['Time_Slot_EXT']

    TITLE_FONTSIZE = 22
    LABEL_FONTSIZE = 18
    LEGEND_FONTSIZE = 16
    TICK_FONTSIZE = 14

    time_in_hours = time_slots_ext / 4.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15),
                                        gridspec_kw={'height_ratios': [1, 1, 2]})

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))

    ax1.plot(time_in_hours, df['P_Total_kW_Nominal'], label='Nominal Total DC Power', linestyle='--', color='gray')
    ax1.plot(time_in_hours, df['P_Total_kW'], label='Optimized Total DC Power', color='crimson')
    ax1.set_ylabel('Power (kW)', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Optimized Data Center Performance and Workload Composition', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    ax2.plot(time_in_hours, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_ylabel('Energy Price (£/MWh)', color='royalblue', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='royalblue', labelsize=TICK_FONTSIZE)
    ax2.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', labelbottom=False)

    if not flex_load_origin_df.empty:
        flex_load_origin_df['lag'] = flex_load_origin_df['processing_slot'] - flex_load_origin_df['original_slot']
        flex_pivot_by_lag = flex_load_origin_df.pivot_table(
            index='processing_slot', columns='lag', values='cpu_load', aggfunc='sum'
        ).fillna(0)

        plot_df = pd.DataFrame({'Inflexible': df['Inflexible_Load_CPU_Nom'].values}, index=df['Time_Slot_EXT'])
        plot_df = plot_df.join(flex_pivot_by_lag).fillna(0)

        rename_dict = {lag: f'Flexible (Lag {int(lag)})' for lag in flex_pivot_by_lag.columns}
        plot_df.rename(columns=rename_dict, inplace=True)

        flexible_cols = [col for col in plot_df.columns if 'Flexible' in col]
        colors = ['black'] + list(plt.cm.viridis(np.linspace(0, 1, len(flexible_cols))))

        plot_df.plot(kind='bar', stacked=True, ax=ax3, width=0.8, color=colors)

        nominal_total_load = data['inflexibleLoadProfile_TEXT'][1:] + data['flexibleLoadProfile_TEXT'][1:]
        ax3.plot(range(len(nominal_total_load)), nominal_total_load, label='Nominal Total Workload', linestyle='--', color='gray')

        ax3.set_xlabel('Time (Hours)', fontsize=LABEL_FONTSIZE)
        ax3.set_ylabel('CPU Load Units', fontsize=LABEL_FONTSIZE)
        ax3.legend(title='Load Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.tick_params(axis='y', labelsize=TICK_FONTSIZE)

        tick_frequency = max(1, len(plot_df.index) // 24)
        tick_positions = range(len(plot_df.index))[::tick_frequency]

        original_tick_labels = [float(label) for label in plot_df.index[::tick_frequency]]
        hour_tick_labels = [label / 4.0 for label in original_tick_labels]
        formatted_hour_labels = [f'{label:g}' for label in hour_tick_labels]

        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(formatted_hour_labels, rotation=45, ha="right", fontsize=TICK_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig(IMAGE_DIR / 'combined_performance_final.png')
    print("✅ Final combined chart saved.")

    plt.show()
    print("\nAll charts have been generated and saved.")


def run_single_optimization(params: ModelParameters, input_data: dict, msg=False, linear=False):
    """
    Runs a single optimization instance with a given set of parameters using PuLP (CBC).
    """
    model_type = "Linear" if linear else "Piecewise-Linear"
    print(f"Building and solving model ({model_type}) with PuLP/CBC...")
    m = build_model(params, input_data, linear=linear)

    solver = pl.PULP_CBC_CMD(msg=bool(msg), gapRel=MIP_REL_GAP)
    status = m.prob.solve(solver)

    if pl.LpStatus[status] in ("Optimal", "Not Solved", "Infeasible") and pl.LpStatus[status] == "Optimal":
        results_df, flex_load_origin_df = post_process_results(m, params, input_data)
        total_cost = pl.value(m.prob.objective)
        return total_cost, results_df, flex_load_origin_df
    elif pl.LpStatus[status] == "Optimal":
        results_df, flex_load_origin_df = post_process_results(m, params, input_data)
        total_cost = pl.value(m.prob.objective)
        return total_cost, results_df, flex_load_origin_df
    else:
        print(f"Solver did not find a feasible solution. Status: {pl.LpStatus[status]}")
        return None, None, None


def run_full_optimisation(include_charts: bool, linear: bool = False):
    """
    Sets up and runs the full, baseline optimization, including generating charts.
    :param include_charts: If True, generates and saves output charts.
    :param linear: If True, uses a linear CPU-to-power model. Otherwise, uses piecewise.
    """
    print("1. Setting up model parameters...")
    params = ModelParameters()

    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)

    total_cost, results_df, flex_load_origin_df = run_single_optimization(params, input_data, msg=True, linear=linear)

    if total_cost is not None:
        print_summary(params, results_df)
        if include_charts:
            create_and_save_charts(results_df, flex_load_origin_df, input_data, params)

        output_path = DATA_DIR_OUTPUTS / "optimised_baseline.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nResults successfully exported to '{output_path}'")


if __name__ == '__main__':
    # Run the default piecewise-linear model
    run_full_optimisation(include_charts=True, linear=False)

    # Example of how to run the linear model:
    # print("\n\n--- RUNNING WITH LINEAR CPU-POWER MODEL ---")
    # run_full_optimisation(include_charts=True, linear=True)