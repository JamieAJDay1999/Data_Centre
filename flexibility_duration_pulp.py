import pathlib
import pandas as pd
import numpy as np
import pulp
import time

# --- Import user-provided dependency modules ---
from inputs.parameters_optimisation import ModelParameters, generate_tariff
# MODIFIED: Correctly import extract_detailed_results from the other script
from plotting_and_saving.flex_dur_res_plots_pulp import extract_detailed_results, plot_flex_contribution_grid, save_heatmap_from_results

# --- Path Configuration ------------------------------------------------------
DATA_DIR_INPUTS = pathlib.Path("static/data/optimisation_outputs")
DATA_DIR_OUTPUTS = pathlib.Path("static/data/flexibility_outputs")
IMAGE_DIR = pathlib.Path("static/images/flexibility_outputs")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR_INPUTS.mkdir(parents=True, exist_ok=True)
DATA_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

# --- Constants and Configuration ---------------------------------------------
DEBUG_ON_FAIL = True 
SOLVER_TIME_LIMIT_SECONDS = 60

# THIS LOCAL FUNCTION HAS BEEN REMOVED TO ENSURE THE CORRECT, IMPORTED VERSION IS USED.

def build_duration_model(params: ModelParameters, data: dict, initial_state: dict, baseline_df: pd.DataFrame, start_timestep: int, flex_target_kw: float, flex_time, linear: bool = False):
    m = pulp.LpProblem("Flex_Feasibility", pulp.LpMinimize)
    T_SLOTS_list = list(range(start_timestep, start_timestep + flex_time))
    TEXT_SLOTS_list = list(range(start_timestep, start_timestep + flex_time + 12))
    K_TRANCHES_list = list(params.K_TRANCHES)

    model_vars = {}
    model_vars['total_cpu'] = pulp.LpVariable.dicts("total_cpu", TEXT_SLOTS_list, lowBound=0, upBound=params.max_cpu_usage)
    model_vars['p_grid_it_kw'] = pulp.LpVariable.dicts("p_grid_it_kw", TEXT_SLOTS_list, lowBound=0)
    model_vars['p_grid_od_kw'] = pulp.LpVariable.dicts("p_grid_od_kw", TEXT_SLOTS_list, lowBound=0)
    model_vars['p_it_total_kw'] = pulp.LpVariable.dicts("p_it_total_kw", TEXT_SLOTS_list, lowBound=0)
    model_vars['p_ups_ch_kw'] = pulp.LpVariable.dicts("p_ups_ch_kw", TEXT_SLOTS_list, lowBound=0)
    model_vars['p_ups_disch_kw'] = pulp.LpVariable.dicts("p_ups_disch_kw", TEXT_SLOTS_list, lowBound=0)
    model_vars['e_ups_kwh'] = pulp.LpVariable.dicts("e_ups_kwh", TEXT_SLOTS_list, lowBound=params.e_min_kwh, upBound=params.e_max_kwh)
    model_vars['z_ch'] = pulp.LpVariable.dicts("z_ch", TEXT_SLOTS_list, cat='Binary')
    model_vars['z_disch'] = pulp.LpVariable.dicts("z_disch", TEXT_SLOTS_list, cat='Binary')
    ut_ks_idx = [(t, k, s) for t in T_SLOTS_list for k in K_TRANCHES_list for s in TEXT_SLOTS_list if s >= t and s <= t + params.tranche_max_delay.get(k, k)]
    model_vars['ut_ks'] = pulp.LpVariable.dicts("ut_ks", ut_ks_idx, lowBound=0)
    model_vars['t_it'] = pulp.LpVariable.dicts("t_it", TEXT_SLOTS_list, lowBound=14, upBound=60)
    model_vars['t_rack'] = pulp.LpVariable.dicts("t_rack", TEXT_SLOTS_list, lowBound=14, upBound=40)
    model_vars['t_cold_aisle'] = pulp.LpVariable.dicts("t_cold_aisle", TEXT_SLOTS_list, lowBound=18, upBound=params.T_cAisle_upper_limit_Celsius)
    model_vars['t_hot_aisle'] = pulp.LpVariable.dicts("t_hot_aisle", TEXT_SLOTS_list, lowBound=14, upBound=40)
    model_vars['e_tes_kwh'] = pulp.LpVariable.dicts("e_tes_kwh", TEXT_SLOTS_list, lowBound=params.E_TES_min_kWh, upBound=params.TES_capacity_kWh)
    model_vars['p_chiller_hvac_w'] = pulp.LpVariable.dicts("p_chiller_hvac_w", TEXT_SLOTS_list, lowBound=0)
    model_vars['p_chiller_tes_w'] = pulp.LpVariable.dicts("p_chiller_tes_w", TEXT_SLOTS_list, lowBound=0)
    model_vars['q_cool_w'] = pulp.LpVariable.dicts("q_cool_w", TEXT_SLOTS_list, lowBound=0)
    model_vars['q_ch_tes_w'] = pulp.LpVariable.dicts("q_ch_tes_w", TEXT_SLOTS_list, lowBound=0, upBound=params.TES_w_charge_max)
    model_vars['q_dis_tes_w'] = pulp.LpVariable.dicts("q_dis_tes_w", TEXT_SLOTS_list, lowBound=0, upBound=params.TES_w_discharge_max)
    model_vars['t_in'] = pulp.LpVariable.dicts("t_in", TEXT_SLOTS_list, lowBound=14, upBound=30)
    model_vars['z_tes_ch'] = pulp.LpVariable.dicts("z_tes_ch", TEXT_SLOTS_list, cat='Binary')
    model_vars['z_tes_disch'] = pulp.LpVariable.dicts("z_tes_disch", TEXT_SLOTS_list, cat='Binary')

    if not linear:
        add_it_and_job_constraints_pwl(m, params, data, model_vars, T_SLOTS_list, TEXT_SLOTS_list, K_TRANCHES_list, ut_ks_idx)
    else:
        add_it_and_job_constraints_linear(m, params, data, model_vars, T_SLOTS_list, TEXT_SLOTS_list, K_TRANCHES_list, ut_ks_idx)

    add_ups_constraints(m, params, initial_state, model_vars, TEXT_SLOTS_list, start_timestep)
    add_power_balance_constraints(m, params, model_vars, TEXT_SLOTS_list)
    add_cooling_constraints(m, params, initial_state, model_vars, TEXT_SLOTS_list, start_timestep)
    add_power_change_constraints(m, params, flex_target_kw, start_timestep, flex_time, baseline_df, model_vars)
    
    m += 1, "Feasibility_Objective"
    return m, model_vars

def add_it_and_job_constraints_linear(m, params, data, v, T_SLOTS, TEXT_SLOTS, K_TRANCHES, ut_ks_idx):
    # --- Job Completion Constraints ---
    for t in T_SLOTS:
        total_flexible_work_at_t = data['Rt'][t]
        for k in K_TRANCHES:
            workload_in_tranche_k = total_flexible_work_at_t * data['shiftabilityProfile'].get((t, k), 0)
            relevant_s = [s for s_t, s_k, s in ut_ks_idx if s_t == t and s_k == k]
            m += pulp.lpSum(v['ut_ks'][(t, k, s)] * params.dt_hours for s in relevant_s) == workload_in_tranche_k, f"JobCompletion_t{t}_k{k}"

    # --- Prepare Base CPU Load ---
    unified_base_load = data['inflexibleLoadProfile_TEXT'].copy()
    for t in T_SLOTS:
        if t < len(data['Rt']):
            total_flexible_load_rate = data['Rt'][t] / params.dt_hours
            shiftable_fraction = sum(data['shiftabilityProfile'].get((t, k), 0) for k in K_TRANCHES)
            non_shiftable_fraction = 1.0 - shiftable_fraction
            non_shiftable_load = total_flexible_load_rate * non_shiftable_fraction
            if t < len(unified_base_load):
                unified_base_load[t] += non_shiftable_load
    
    extension_slots = [s for s in TEXT_SLOTS if s not in T_SLOTS]
    for s in extension_slots:
        if s < len(unified_base_load) and s < len(data['flexibleLoadProfile_TEXT']):
             unified_base_load[s] += data['flexibleLoadProfile_TEXT'][s]

    # --- Linear CPU to Power Relationship ---
    for s in TEXT_SLOTS:
        flexible_usage = pulp.lpSum(v['ut_ks'][idx] for idx in ut_ks_idx if idx[2] == s)
        m += v['total_cpu'][s] == unified_base_load[s] + flexible_usage, f"CPU_Total_s{s}"
        
        # Define the final IT power expression using direct linear relationship
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * v['total_cpu'][s]
        m += v['p_it_total_kw'][s] == power_expr, f"IT_Power_s{s}"


def add_it_and_job_constraints_pwl(m, params, data, v, T_SLOTS, TEXT_SLOTS, K_TRANCHES, ut_ks_idx):
    # --- Job Completion Constraints ---
    for t in T_SLOTS:
        total_flexible_work_at_t = data['Rt'][t]
        for k in K_TRANCHES:
            workload_in_tranche_k = total_flexible_work_at_t * data['shiftabilityProfile'].get((t, k), 0)
            relevant_s = [s for s_t, s_k, s in ut_ks_idx if s_t == t and s_k == k]
            m += pulp.lpSum(v['ut_ks'][(t, k, s)] * params.dt_hours for s in relevant_s) == workload_in_tranche_k, f"JobCompletion_t{t}_k{k}"

    # --- Prepare Base CPU Load ---
    unified_base_load = data['inflexibleLoadProfile_TEXT'].copy()
    for t in T_SLOTS:
        if t < len(data['Rt']):
            total_flexible_load_rate = data['Rt'][t] / params.dt_hours
            shiftable_fraction = sum(data['shiftabilityProfile'].get((t, k), 0) for k in K_TRANCHES)
            non_shiftable_fraction = 1.0 - shiftable_fraction
            non_shiftable_load = total_flexible_load_rate * non_shiftable_fraction
            if t < len(unified_base_load):
                unified_base_load[t] += non_shiftable_load
    
    extension_slots = [s for s in TEXT_SLOTS if s not in T_SLOTS]
    for s in extension_slots:
        if s < len(unified_base_load) and s < len(data['flexibleLoadProfile_TEXT']):
             unified_base_load[s] += data['flexibleLoadProfile_TEXT'][s]

    # --- Piecewise Linearization for IT Power (y = x**1.32) ---
    # MODIFIED: This section now exactly replicates the methodology from optimisation_pulp.py
    # by using binary variables and adjacency constraints instead of PuLP's built-in SOS2.

    # 1. Define the (x,y) points for the approximation
    num_pw_points = 11
    pw_x = [i / (num_pw_points - 1) for i in range(num_pw_points)]
    pw_y = [x**1.32 for x in pw_x]
    PW_POINTS = range(num_pw_points)
    SEGMENTS = range(num_pw_points - 1) # Segments between points
    
    # 2. Create weighting and segment selection variables
    pw_idx = [(s, i) for s in TEXT_SLOTS for i in PW_POINTS]
    delta_idx = [(s, seg) for s in TEXT_SLOTS for seg in SEGMENTS]
    v['w'] = pulp.LpVariable.dicts("pw_weight", pw_idx, lowBound=0)
    v['delta'] = pulp.LpVariable.dicts("segment_select", delta_idx, cat='Binary')
    v['cpu_power_factor'] = pulp.LpVariable.dicts("cpu_power_factor", TEXT_SLOTS, lowBound=0)
    
    for s in TEXT_SLOTS:
        # 3. Weights must sum to 1
        m += pulp.lpSum(v['w'][(s, i)] for i in PW_POINTS) == 1, f"WeightSum_s{s}"
        
        # 4. Exactly one segment must be active
        m += pulp.lpSum(v['delta'][(s, seg)] for seg in SEGMENTS) == 1, f"SegmentSelect_s{s}"

        # 5. Adjacency constraints: only weights in the active segment can be non-zero
        n = num_pw_points
        m += v['w'][(s, 0)] <= v['delta'][(s, 0)], f"Adj_s{s}_i0"
        for i in range(1, n - 1): # i = 1 to n-2
             m += v['w'][(s, i)] <= v['delta'][(s, i - 1)] + v['delta'][(s, i)], f"Adj_s{s}_i{i}"
        m += v['w'][(s, n - 1)] <= v['delta'][(s, n - 2)], f"Adj_s{s}_i{n-1}"
        
        # 6. Link variables to the piecewise approximation
        flexible_usage = pulp.lpSum(v['ut_ks'][idx] for idx in ut_ks_idx if idx[2] == s)
        m += v['total_cpu'][s] == unified_base_load[s] + flexible_usage, f"CPU_Total_s{s}"
        m += v['total_cpu'][s] == pulp.lpSum(pw_x[i] * v['w'][(s, i)] for i in PW_POINTS), f"CPU_PW_Link_s{s}"
        m += v['cpu_power_factor'][s] == pulp.lpSum(pw_y[i] * v['w'][(s, i)] for i in PW_POINTS), f"PowerFactor_PW_Link_s{s}"
        
        # 7. Define the final IT power expression
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * v['cpu_power_factor'][s]
        m += v['p_it_total_kw'][s] == power_expr, f"IT_Power_s{s}"


def add_ups_constraints(m, params, initial_state, v, TEXT_SLOTS, start_timestep):
    for s in TEXT_SLOTS:
        prev_energy = initial_state.get('E_UPS_kWh', params.e_start_kwh) if s == start_timestep else v['e_ups_kwh'][s-1]
        charge = params.eta_ch * v['p_ups_ch_kw'][s] * params.dt_hours
        discharge = (v['p_ups_disch_kw'][s] / params.eta_disch) * params.dt_hours
        m += v['e_ups_kwh'][s] == prev_energy + charge - discharge, f"UPS_EnergyBalance_s{s}"
        m += v['p_ups_ch_kw'][s] <= v['z_ch'][s] * params.p_max_ch_kw, f"UPS_MaxCharge_s{s}"
        m += v['p_ups_ch_kw'][s] >= v['z_ch'][s] * params.p_min_ch_kw, f"UPS_MinCharge_s{s}"
        m += v['p_ups_disch_kw'][s] <= v['z_disch'][s] * params.p_max_disch_kw, f"UPS_MaxDischarge_s{s}"
        m += v['p_ups_disch_kw'][s] >= v['z_disch'][s] * params.p_min_disch_kw, f"UPS_MinDischarge_s{s}"
        m += v['z_ch'][s] + v['z_disch'][s] <= 1, f"UPS_ChargeDischargeMutex_s{s}"
    m += v['e_ups_kwh'][max(TEXT_SLOTS)] >= params.e_start_kwh, "UPS_FinalEnergy"

def add_power_balance_constraints(m, params, v, TEXT_SLOTS):
    for s in TEXT_SLOTS:
        m += v['p_it_total_kw'][s] == v['p_grid_it_kw'][s] + v['p_ups_disch_kw'][s], f"PowerBalance_IT_s{s}"
        m += v['p_grid_od_kw'][s] == v['p_it_total_kw'][s] * params.nominal_overhead_factor, f"PowerBalance_Overhead_s{s}"

def add_cooling_constraints(m, params, initial_state, v, TEXT_SLOTS, start_timestep):
    m += v['t_it'][start_timestep]   >= initial_state.get('T_IT_Celsius',   params.T_IT_initial_Celsius), "Initial_T_IT"
    m += v['t_rack'][start_timestep] >= initial_state.get('T_Rack_Celsius', params.T_Rack_initial_Celsius), "Initial_T_Rack"
    m += v['t_cold_aisle'][start_timestep] >= initial_state.get('T_cAisle_Celsius', params.T_cAisle_initial), "Initial_T_cAisle"
    m += v['t_hot_aisle'][start_timestep]  >= initial_state.get('T_hAisle_Celsius', params.T_hAisle_initial), "Initial_T_hAisle"
    m += v['e_tes_kwh'][start_timestep]   == initial_state.get('E_TES_kWh',     params.TES_initial_charge_kWh), "Initial_E_TES"
    t0 = start_timestep
    mcp = params.m_dot_air * params.c_p_air
    m += v['q_cool_w'][t0] == (v['p_chiller_hvac_w'][t0] * params.COP_HVAC) + v['q_dis_tes_w'][t0], f"Cooling_Balance_t{t0}"
    m += v['q_ch_tes_w'][t0] == v['p_chiller_tes_w'][t0] * params.COP_HVAC, f"TES_Charge_t{t0}"
    m += v['t_in'][t0] == v['t_hot_aisle'][t0] - v['q_cool_w'][t0] / mcp, f"InletTemp_t{t0}"
    m += v['q_cool_w'][t0] <= (v['t_hot_aisle'][t0] - params.T_cAisle_lower_limit_Celsius) * mcp, f"CoolingCapacity_t{t0}"
    m += v['p_chiller_tes_w'][t0] + v['p_chiller_hvac_w'][t0] <= params.P_chiller_max, f"ChillerPowerLimit_t{t0}"
    m += v['q_cool_w'][t0] >= (v['p_it_total_kw'][t0] * 1000.0), f"MeetITLoad_t{t0}"
    for s in TEXT_SLOTS:
        m += v['q_ch_tes_w'][s]   <= v['z_tes_ch'][s]    * params.TES_w_charge_max, f"TES_MaxCharge_s{s}"
        m += v['q_dis_tes_w'][s]  <= v['z_tes_disch'][s] * params.TES_w_discharge_max, f"TES_MaxDischarge_s{s}"
        m += v['z_tes_ch'][s] + v['z_tes_disch'][s] <= 1, f"TES_ChargeDischargeMutex_s{s}"
    for t in [s for s in TEXT_SLOTS if s > start_timestep]:
        m += v['q_cool_w'][t] == (v['p_chiller_hvac_w'][t] * params.COP_HVAC) + v['q_dis_tes_w'][t], f"Cooling_Balance_t{t}"
        m += v['q_ch_tes_w'][t] == v['p_chiller_tes_w'][t] * params.COP_HVAC, f"TES_Charge_t{t}"
        m += v['t_in'][t] == v['t_hot_aisle'][t] - v['q_cool_w'][t] / mcp, f"InletTemp_t{t}"
        m += v['q_cool_w'][t] <= (v['t_hot_aisle'][t] - params.T_cAisle_lower_limit_Celsius) * mcp, f"CoolingCapacity_t{t}"
        it_heat_watts = v['p_it_total_kw'][t] * 1000.0
        m += v['t_it'][t] == v['t_it'][t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (v['t_it'][t-1] - v['t_rack'][t])) / params.C_IT), f"Thermal_TIT_t{t}"
        m += v['t_rack'][t] == v['t_rack'][t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(v['t_cold_aisle'][t]-v['t_rack'][t-1]) + params.G_conv*(v['t_it'][t-1]-v['t_rack'][t-1])) / params.C_Rack), f"Thermal_TRack_t{t}"
        m += v['t_cold_aisle'][t] == v['t_cold_aisle'][t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(v['t_in'][t]-v['t_cold_aisle'][t-1]) - params.G_cold*(v['t_cold_aisle'][t-1]-params.T_out_Celsius)) / params.C_cAisle), f"Thermal_TCAisle_t{t}"
        m += v['t_hot_aisle'][t] == v['t_hot_aisle'][t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(v['t_rack'][t]-v['t_hot_aisle'][t-1])) / params.C_hAisle), f"Thermal_THAisle_t{t}"
        dE_tes_kwh = (v['q_ch_tes_w'][t]*params.TES_charge_efficiency - v['q_dis_tes_w'][t]/params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        m += v['e_tes_kwh'][t] == v['e_tes_kwh'][t-1] + dE_tes_kwh, f"TES_EnergyBalance_t{t}"
        m += v['q_dis_tes_w'][t] - v['q_dis_tes_w'][t-1] <= params.TES_p_dis_ramp, f"TES_RampDischarge_t{t}"
        m += v['q_ch_tes_w'][t] - v['q_ch_tes_w'][t-1] <= params.TES_p_ch_ramp, f"TES_RampCharge_t{t}"
        m += v['p_chiller_tes_w'][t] + v['p_chiller_hvac_w'][t] <= params.P_chiller_max, f"ChillerPowerLimit_t{t}"
        m += v['q_cool_w'][t] >= it_heat_watts, f"MeetITLoad_t{t}"

def add_power_change_constraints(m, params, flex_target_kw, start_timestep, flex_time, baseline_df, v):
    power_system_tolerance = 0.1
    for t in range(start_timestep, start_timestep + flex_time):
        baseline_power_at_t = baseline_df.loc[t, 'P_Total_kW']
        current_total_power_kw = (v['p_grid_it_kw'][t] + 
                                  (v['p_chiller_hvac_w'][t] / 1000.0) +
                                  (v['p_chiller_tes_w'][t] / 1000.0) +
                                  v['p_grid_od_kw'][t] + 
                                  v['p_ups_ch_kw'][t])
        target_power_kw = baseline_power_at_t + flex_target_kw
        m += current_total_power_kw <= target_power_kw + power_system_tolerance, f"PowerTarget_Upper_t{t}"
        m += current_total_power_kw >= target_power_kw - power_system_tolerance, f"PowerTarget_Lower_t{t}"

def load_and_prepare_data(params: ModelParameters):
    try:
        load_profiles_df = pd.read_csv(DATA_DIR_INPUTS / "load_profiles_opt.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR_INPUTS / "shiftability_profile_opt.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profiles_opt.csv' and 'shiftability_profile_opt.csv' are in {DATA_DIR_INPUTS}")
        raise e
    inflexible = load_profiles_df['inflexible_load']
    base_flex = load_profiles_df['flexible_load']
    if not hasattr(params, 'T_SLOTS'):
        params.T_SLOTS = range(1, 97)
    valid_tslots = [ts for ts in params.T_SLOTS if ts in base_flex.index]
    base_flex_t = base_flex.loc[valid_tslots]
    shiftability_df.columns = shiftability_df.columns.astype(int)
    shiftability = shiftability_df.stack().to_dict()
    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    params.sim_minutes_ext = (max(params.T_SLOTS) + 12) * (params.dt_seconds // 60)
    params.simulation_minutes = max(params.T_SLOTS) * (params.dt_seconds // 60)
    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'flexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'shiftabilityProfile': resample_shiftability_profile(shiftability, RESAMPLE_FACTOR)
    }
    baseFlexibleLoadProfile_T = np.insert(np.array(np.repeat(base_flex_t.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0)
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours
    baseFlexibleLoadProfile_TEXT = np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0)
    len_inflexible = len(data['inflexibleLoadProfile_TEXT'])
    len_flexible = len(baseFlexibleLoadProfile_TEXT)
    min_len = min(len_inflexible, len_flexible)
    data['Pt_IT_nom_TEXT'] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (data['inflexibleLoadProfile_TEXT'][:min_len] + baseFlexibleLoadProfile_TEXT[:min_len])
    params.num_steps_extended = params.sim_minutes_ext // (params.dt_seconds // 60)
    data['electricity_price'] = generate_tariff(params.num_steps_extended, params.dt_seconds)
    return data

def resample_shiftability_profile(shiftability_profile, repeats):
    extended_data = {}
    counter = 1
    for i in range(1, 97):
        for _ in range(repeats):
            for j in range(1, 13):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data

def find_max_duration(params, data, baseline_df, start_timestep, flex_kw, search_type, linear=False):
    print(f"\n--- Searching for Max Duration: Timestep {start_timestep}, Flex {flex_kw} kW ---")
    initial_state = baseline_df.loc[start_timestep].to_dict()
    low = 1
    high = len(baseline_df) - start_timestep - 12
    max_optimal_duration = 0
    start_time = time.time()
    
    while low <= high:
        mid_duration = (low + high) // 2 if search_type == 'binary' else low
        if mid_duration == 0: break
        print(f"\rTesting duration: {mid_duration} steps... ", end="")
        
        model, _ = build_duration_model(params, data, initial_state, baseline_df, start_timestep, flex_kw, mid_duration, linear=linear)
        solver = pulp.PULP_CBC_CMD(timeLimit=SOLVER_TIME_LIMIT_SECONDS, msg=False)
        model.solve(solver)
        
        if model.status == pulp.LpStatusOptimal:
            max_optimal_duration = mid_duration
            low = mid_duration + 1
            print("Feasible solution found.")
        else:
            print(f"No feasible solution found (Status: {pulp.LpStatus[model.status]}).")
            high = mid_duration - 1
            if search_type == 'linear': search_type = 'binary'
            
    end_time = time.time()
    print(f"\nSearch complete in {end_time - start_time:.2f} seconds.")
    return max_optimal_duration

def main(flex_magnitudes, timesteps, include_banked_results, search_type, generate_plots=True, linear=False):
    params = ModelParameters()
    params.tranche_max_delay = {x:x for x in range(1,13)}
    params.K_TRANCHES = range(1, 13)
    
    baseline_csv_path = DATA_DIR_INPUTS / "optimised_baseline.csv"
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        if 'Time_Slot_EXT' in baseline_df.columns and baseline_df.index.name != 'Time_Slot_EXT':
            baseline_df = baseline_df.set_index('Time_Slot_EXT')
    except FileNotFoundError:
        print(f"Error: '{baseline_csv_path}' not found. Please run 'integrated_dc_model.py' first.")
        return

    params.T_SLOTS = range(1, 97)
    data = load_and_prepare_data(params)

    if include_banked_results is None:
        old_results = pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])
    else:
        bank_file = DATA_DIR_OUTPUTS / include_banked_results
        old_results = pd.read_csv(bank_file) if bank_file.exists() else pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])
    
    old_results = old_results.set_index(['Timestep', 'Flex_Magnitude_kW'])

    results_list = []
    plot_data_list = []
    for ts in timesteps:
        for fm in flex_magnitudes:
            key = (ts, fm)
            if include_banked_results is not None and key in old_results.index:
                duration_min = float(old_results.loc[key, 'Max_Duration_Min'])
                max_dur_steps = int(duration_min / (params.dt_seconds / 60))
                print(f"Timestep {ts}, Flex {fm} kW: Using banked duration {duration_min:.0f} minutes")
            else:
                max_dur_steps = find_max_duration(params, data, baseline_df, ts, fm, search_type, linear=linear)
                duration_min = max_dur_steps * params.dt_seconds / 60
                print(f"Timestep {ts}, Flex {fm} kW: Computed max duration {duration_min:.0f} minutes")

            results_list.append({'Timestep': ts, 'Flex_Magnitude_kW': fm, 'Max_Duration_Min': duration_min})

            if max_dur_steps > 0:
                initial_state = baseline_df.loc[ts].to_dict()
                model, model_vars = build_duration_model(params, data, initial_state, baseline_df, ts, fm, max_dur_steps, linear=linear)
                solver = pulp.PULP_CBC_CMD(timeLimit=SOLVER_TIME_LIMIT_SECONDS, msg=False)
                model.solve(solver)
                
                if model.status == pulp.LpStatusOptimal:
                    # MODIFIED: This now correctly calls the imported function
                    results_df = extract_detailed_results(model_vars, params, data, ts, max_dur_steps, baseline_df)
                    
                    csv_filename = f"flex_duration_detailed_results_ts{ts}_flex{str(fm).replace('-', 'neg')}.csv"
                    results_df.to_csv(DATA_DIR_OUTPUTS / csv_filename)
                    print(f"  -> Saved detailed results to {csv_filename}")
                    if generate_plots:
                        plot_data_list.append({
                            'results_df': results_df,
                            'ts': ts,
                            'fm': fm,
                            'dur_steps': max_dur_steps
                        })

    if generate_plots and plot_data_list:
        print("\nGenerating summary grid plot...")
        plot_flex_contribution_grid(plot_data_list, timesteps, flex_magnitudes)

    save_heatmap_from_results(
        results_rows=results_list,
        csv_path=DATA_DIR_OUTPUTS / "flex_duration_results.csv",
        png_path=IMAGE_DIR / "flex_duration_heatmap.png"
    )

if __name__ == '__main__':
    timesteps = [15, 65] #[1] + list(range(5, 97, 5))  # Start at 1, then every 5th timestep up to 96
    flex_magnitudes = [-200, -150, -100] #[75, 50, 25, -100, -150, -200, -250, -300, -350, -400, -450, -500]
    include_banked_results = "flex_duration_results.csv"  # Set to None to ignore banked results
    # Run the default piecewise-linear model
    main(flex_magnitudes, timesteps, include_banked_results, search_type='binary', generate_plots=True, linear=False)

    # Example of how to run the linear model:
    # print("\n\n--- RUNNING WITH LINEAR CPU-POWER MODEL ---")
    # main(flex_magnitudes, timesteps, include_banked_results, search_type='binary', generate_plots=True, linear=True) 