import pathlib
import pandas as pd
import numpy as np
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Import user-provided dependency modules ---
from inputs.parameters_optimisation import ModelParameters, generate_tariff
# This imported function will need to be updated separately to handle Pyomo model objects
from plotting_and_saving.flexibility_duration_results_and_plots import extract_detailed_results, plot_flex_contribution_grid, save_heatmap_from_results

# --- Path Configuration ------------------------------------------------------
# Define base directories for data and images
DATA_DIR_INPUTS = pathlib.Path("static/data/optimisation_outputs")
DATA_DIR_OUTPUTS = pathlib.Path("static/data/flexibility_outputs")
IMAGE_DIR = pathlib.Path("static/images/flexibility_outputs")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR_INPUTS.mkdir(parents=True, exist_ok=True)
DATA_DIR_INPUTS.mkdir(parents=True, exist_ok=True)
DATA_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


# --- Constants and Configuration ---------------------------------------------
DURATION_SIM_HORIZON_MINUTES = 720 # 12 hours should be enough to find max duration
DEBUG_ON_FAIL = True # If True, will re-run failed simulations with verbose output
SOLVER_TIME_LIMIT_SECONDS = 20 # Return infeasible if solver exceeds 3 minutes

def build_duration_model(params: ModelParameters, data: dict, initial_state: dict, baseline_df: pd.DataFrame, start_timestep: int, flex_target_kw: float, flex_time):
    # MODIFIED: Use Pyomo's ConcreteModel
    m = pyo.ConcreteModel("Flex_Feasibility")

    # Define the time slots for this specific optimization window
    # Add a buffer of 12 steps for job shifting past the flex_time
    T_SLOTS_list = range(start_timestep, start_timestep + flex_time)
    TEXT_SLOTS_list = range(start_timestep, start_timestep + flex_time + 12)
    
    # MODIFIED: Define index sets for Pyomo
    m.T_SLOTS = pyo.Set(initialize=T_SLOTS_list)
    m.TEXT_SLOTS = pyo.Set(initialize=TEXT_SLOTS_list)
    m.K_TRANCHES = pyo.Set(initialize=params.K_TRANCHES)

    # --- Define Pyomo Variables ---
    m.total_cpu = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.max_cpu_usage))
    # Power variables in kW
    m.p_grid_it_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.p_grid_od_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.p_it_total_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.p_ups_ch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.p_ups_disch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.e_ups_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.e_min_kwh, params.e_max_kwh))
    m.z_ch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary)
    m.z_disch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary)
    ut_ks_idx = [(t, k, s) for t in m.T_SLOTS for k in m.K_TRANCHES for s in m.TEXT_SLOTS if s >= t and s <= t + params.tranche_max_delay[k]]
    m.ut_ks_idx = pyo.Set(initialize=ut_ks_idx)
    m.ut_ks = pyo.Var(m.ut_ks_idx, within=pyo.NonNegativeReals)
    m.t_it = pyo.Var(m.TEXT_SLOTS, bounds=(14, 60))
    m.t_rack = pyo.Var(m.TEXT_SLOTS, bounds=(14, 40))
    m.t_cold_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(18, params.T_cAisle_upper_limit_Celsius))
    m.t_hot_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(14, 40))
    m.e_tes_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.E_TES_min_kWh, params.TES_capacity_kWh))
    # Electrical power for cooling in Watts
    m.p_chiller_hvac_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.p_chiller_tes_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    # Thermal power in Watts
    m.q_cool_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    m.q_ch_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_charge_max))
    m.q_dis_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_discharge_max))
    m.t_in = pyo.Var(m.TEXT_SLOTS, bounds=(14, 30))
    m.z_tes_ch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary)
    m.z_tes_disch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary)
    
    # --- Add Constraints ---
    add_it_and_job_constraints(m, params, data)
    add_ups_constraints(m, params, initial_state, start_timestep)
    add_power_balance_constraints(m, params)
    add_cooling_constraints(m, params, initial_state, start_timestep)
    add_power_change_constraints(m, params, flex_target_kw, start_timestep, flex_time, baseline_df)
    
    # --- Objective Function ---
    """def objective_rule(mod):
        # Objective combines all grid-drawn power in kW
        return sum(
            mod.p_grid_it_kw[t] +
            (mod.p_chiller_hvac_w[t] / 1000.0) + # W to kW
            (mod.p_chiller_tes_w[t] / 1000.0) +  # W to kW
            mod.p_grid_od_kw[t] +
            mod.p_ups_ch_kw[t]
            for t in mod.T_SLOTS
        )
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)"""
    m.objective = pyo.Objective(expr=1, sense=pyo.minimize)
    return m


def add_it_and_job_constraints(m, params, data):
    # --- JOB COMPLETION for Shiftable Workload ---
    m.JobCompletion = pyo.ConstraintList()
    for t in m.T_SLOTS:
        total_flexible_work_at_t = data['Rt'][t]
        for k in m.K_TRANCHES:
            workload_in_tranche_k = total_flexible_work_at_t * data['shiftabilityProfile'].get((t, k), 0)
            m.JobCompletion.add(
                sum(m.ut_ks[(t, k, s)] * params.dt_hours for s in m.TEXT_SLOTS if (t, k, s) in m.ut_ks_idx) == workload_in_tranche_k
            )

    # --- UNIFIED BASE LOAD (Logic preserved exactly from original script) ---
    unified_base_load = data['inflexibleLoadProfile_TEXT'].copy()
    for t in m.T_SLOTS:
        if t < len(data['Rt']):
            total_flexible_load_rate = data['Rt'][t] / params.dt_hours
            shiftable_fraction = sum(data['shiftabilityProfile'].get((t, k), 0) for k in m.K_TRANCHES)
            non_shiftable_fraction = 1.0 - shiftable_fraction
            non_shiftable_load = total_flexible_load_rate * non_shiftable_fraction
            if t < len(unified_base_load):
                unified_base_load[t] += non_shiftable_load
    
    extension_slots = [s for s in m.TEXT_SLOTS if s not in m.T_SLOTS]
    for s in extension_slots:
        if s < len(unified_base_load) and s < len(data['flexibleLoadProfile_TEXT']):
             unified_base_load[s] += data['flexibleLoadProfile_TEXT'][s]

    # --- Manual Piecewise Linearization using SOS2 Constraints ---
    num_pw_points = 11
    pw_x = [i / (num_pw_points - 1) for i in range(num_pw_points)]
    pw_y = [x**1.32 for x in pw_x]
    m.PW_POINTS = pyo.RangeSet(0, num_pw_points - 1)
    m.w = pyo.Var(m.TEXT_SLOTS, m.PW_POINTS, within=pyo.NonNegativeReals)
    m.cpu_power_factor = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)

    m.WeightSum = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        m.WeightSum.add(sum(m.w[s, i] for i in m.PW_POINTS) == 1)

    def sos_rule(model, s):
        return [model.w[s, i] for i in model.PW_POINTS]
    m.CPU_SOS2 = pyo.SOSConstraint(m.TEXT_SLOTS, rule=sos_rule, sos=2)
    
    m.CPUandPower = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        flexible_usage = sum(m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s)
        
        # Link CPU usage to the weights and the original unified base load logic
        m.CPUandPower.add(m.total_cpu[s] == unified_base_load[s] + flexible_usage)
        m.CPUandPower.add(m.total_cpu[s] == sum(pw_x[i] * m.w[s, i] for i in m.PW_POINTS))
        
        # Define the power factor and the final linearized power expression
        m.CPUandPower.add(m.cpu_power_factor[s] == sum(pw_y[i] * m.w[s, i] for i in m.PW_POINTS))
        # p_it_total_kw is in kW
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.cpu_power_factor[s]
        m.CPUandPower.add(m.p_it_total_kw[s] == power_expr)


def add_ups_constraints(m, params, initial_state, start_timestep):
    m.UPS_Constraints = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # e_ups_kwh is in kWh, p_ups... variables are in kW
        prev_energy = initial_state.get('E_UPS_kWh', params.e_start_kwh) if s == start_timestep else m.e_ups_kwh[s-1]
        charge = params.eta_ch * m.p_ups_ch_kw[s] * params.dt_hours
        discharge = (m.p_ups_disch_kw[s] / params.eta_disch) * params.dt_hours
        m.UPS_Constraints.add(m.e_ups_kwh[s] == prev_energy + charge - discharge)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] <= m.z_ch[s] * params.p_max_ch_kw)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] >= m.z_ch[s] * params.p_min_ch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] <= m.z_disch[s] * params.p_max_disch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] >= m.z_disch[s] * params.p_min_disch_kw)
        m.UPS_Constraints.add(m.z_ch[s] + m.z_disch[s] <= 1)
    m.UPS_Constraints.add(m.e_ups_kwh[max(m.TEXT_SLOTS)] >= params.e_start_kwh)


def add_power_balance_constraints(m, params):
    m.PowerBalance = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # All power variables in this balance are in kW
        m.PowerBalance.add(m.p_it_total_kw[s] == m.p_grid_it_kw[s] + m.p_ups_disch_kw[s])
        m.PowerBalance.add(m.p_grid_od_kw[s] == m.p_it_total_kw[s] * params.nominal_overhead_factor)


def add_cooling_constraints(m, params, initial_state, start_timestep):
    m.CoolingConstraints = pyo.ConstraintList()
    # Pin initial states
    m.CoolingConstraints.add(m.t_it[start_timestep]   >= initial_state.get('T_IT_Celsius',   params.T_IT_initial_Celsius))
    m.CoolingConstraints.add(m.t_rack[start_timestep] >= initial_state.get('T_Rack_Celsius', params.T_Rack_initial_Celsius))
    m.CoolingConstraints.add(m.t_cold_aisle[start_timestep] >= initial_state.get('T_cAisle_Celsius', params.T_cAisle_initial))
    m.CoolingConstraints.add(m.t_hot_aisle[start_timestep]  >= initial_state.get('T_hAisle_Celsius', params.T_hAisle_initial))
    m.CoolingConstraints.add(m.e_tes_kwh[start_timestep]   == initial_state.get('E_TES_kWh',     params.TES_initial_charge_kWh))

    t0 = start_timestep
    mcp = params.m_dot_air * params.c_p_air
    m.CoolingConstraints.add(m.q_cool_w[t0] == (m.p_chiller_hvac_w[t0] * params.COP_HVAC) + m.q_dis_tes_w[t0])
    m.CoolingConstraints.add(m.q_ch_tes_w[t0] == m.p_chiller_tes_w[t0] * params.COP_HVAC)
    m.CoolingConstraints.add(m.t_in[t0] == m.t_hot_aisle[t0] - m.q_cool_w[t0] / mcp)
    m.CoolingConstraints.add(m.q_cool_w[t0] <= (m.t_hot_aisle[t0] - params.T_cAisle_lower_limit_Celsius) * mcp)
    m.CoolingConstraints.add(m.p_chiller_tes_w[t0] + m.p_chiller_hvac_w[t0] <= params.P_chiller_max)
    m.CoolingConstraints.add(m.q_cool_w[t0] >= (m.p_it_total_kw[t0] * 1000.0))

    for s in m.TEXT_SLOTS:
        m.CoolingConstraints.add(m.q_ch_tes_w[s]   <= m.z_tes_ch[s]    * params.TES_w_charge_max)
        m.CoolingConstraints.add(m.q_dis_tes_w[s]  <= m.z_tes_disch[s] * params.TES_w_discharge_max)
        m.CoolingConstraints.add(m.z_tes_ch[s] + m.z_tes_disch[s] <= 1)
        
    for t in list(m.TEXT_SLOTS)[1:]:
        m.CoolingConstraints.add(m.q_cool_w[t] == (m.p_chiller_hvac_w[t] * params.COP_HVAC) + m.q_dis_tes_w[t])
        m.CoolingConstraints.add(m.q_ch_tes_w[t] == m.p_chiller_tes_w[t] * params.COP_HVAC)
        m.CoolingConstraints.add(m.t_in[t] == m.t_hot_aisle[t] - m.q_cool_w[t] / mcp)
        m.CoolingConstraints.add(m.q_cool_w[t] <= (m.t_hot_aisle[t] - params.T_cAisle_lower_limit_Celsius) * mcp)
        
        it_heat_watts = m.p_it_total_kw[t] * 1000.0
        
        m.CoolingConstraints.add(m.t_it[t] == m.t_it[t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (m.t_it[t-1] - m.t_rack[t])) / params.C_IT))
        m.CoolingConstraints.add(m.t_rack[t] == m.t_rack[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_cold_aisle[t]-m.t_rack[t-1]) + params.G_conv*(m.t_it[t-1]-m.t_rack[t-1])) / params.C_Rack))
        m.CoolingConstraints.add(m.t_cold_aisle[t] == m.t_cold_aisle[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_in[t]-m.t_cold_aisle[t-1]) - params.G_cold*(m.t_cold_aisle[t-1]-params.T_out_Celsius)) / params.C_cAisle))
        m.CoolingConstraints.add(m.t_hot_aisle[t] == m.t_hot_aisle[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_rack[t]-m.t_hot_aisle[t-1])) / params.C_hAisle))
        
        # dE_tes is in kWh, so q_..._w (in W) must be divided by 1000
        dE_tes_kwh = (m.q_ch_tes_w[t]*params.TES_charge_efficiency - m.q_dis_tes_w[t]/params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        m.CoolingConstraints.add(m.e_tes_kwh[t] == m.e_tes_kwh[t-1] + dE_tes_kwh)
        
        m.CoolingConstraints.add(m.q_dis_tes_w[t] - m.q_dis_tes_w[t-1] <= params.TES_p_dis_ramp)
        m.CoolingConstraints.add(m.q_ch_tes_w[t] - m.q_ch_tes_w[t-1] <= params.TES_p_ch_ramp)
        m.CoolingConstraints.add(m.p_chiller_tes_w[t] + m.p_chiller_hvac_w[t] <= params.P_chiller_max)
        m.CoolingConstraints.add(m.q_cool_w[t] >= it_heat_watts)


def add_power_change_constraints(m, params, flex_target_kw, start_timestep, flex_time, baseline_df):
    m.PowerChange = pyo.ConstraintList()
    power_system_tolerance = 0.1
    for t in range(start_timestep, start_timestep + flex_time):
        baseline_power_at_t = baseline_df.loc[t, 'P_Total_kW']
        # Calculate current total power in kW
        current_total_power_kw = (m.p_grid_it_kw[t] + 
                                  (m.p_chiller_hvac_w[t] / 1000.0) + # W to kW
                                  (m.p_chiller_tes_w[t] / 1000.0) +   # W to kW
                                  m.p_grid_od_kw[t] + 
                                  m.p_ups_ch_kw[t])
        target_power_kw = baseline_power_at_t + flex_target_kw
        # Compare kW with kW
        m.PowerChange.add(current_total_power_kw <= target_power_kw + power_system_tolerance)
        m.PowerChange.add(current_total_power_kw >= target_power_kw - power_system_tolerance)

# Remaining data loading and helper functions are unchanged as they don't use the solver package
def load_and_prepare_data(params: ModelParameters):
    # This function remains unchanged
    try:
        load_profiles_df = pd.read_csv(DATA_DIR_INPUTS / "load_profiles_opt.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR_INPUTS / "shiftability_profile_opt.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profiles.csv' and 'shiftability_profile.csv' are in {DATA_DIR_INPUTS}")
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
    # This function remains unchanged
    extended_data = {}
    counter = 1
    for i in range(1, 97):
        for _ in range(repeats):
            for j in range(1, 13):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data

def find_max_duration(params, data, baseline_df, start_timestep, flex_kw, search_type):
    print(f"\n--- Searching for Max Duration: Timestep {start_timestep}, Flex {flex_kw} kW ---")
    initial_state = baseline_df.loc[start_timestep].to_dict()
    low = 1
    high = len(baseline_df) - start_timestep - 11
    max_optimal_duration = 0
    start_time = time.time()
    
    while low <= high:
        if search_type == 'binary':
            mid_duration = (low + high) // 2
        elif search_type == 'linear':
            mid_duration = low + 6
        if mid_duration == 0: break
        print(f"\rTesting duration: {mid_duration} steps... ", end="")
        
        model = build_duration_model(params, data, initial_state, baseline_df, start_timestep, flex_kw, mid_duration)
        
        # MODIFIED: Use Pyomo solver
        solver = pyo.SolverFactory('scip') # Or 'glpk', 'scip', etc.
        solver.options['limits/time'] = SOLVER_TIME_LIMIT_SECONDS
        results = solver.solve(model, tee=False)
        
        term_cond = results.solver.termination_condition 
        #if term_cond in [TerminationCondition.optimal, TerminationCondition.maxTimeLimit]:
        # Check the solver's formal termination condition. This is more reliable
        # than inspecting the solution object when the solver is aborted.
        if results.solver.termination_condition in [TerminationCondition.optimal]:#, TerminationCondition.maxTimeLimit]:
            max_optimal_duration = mid_duration
            low = mid_duration + 1
            print("Feasible solution found.")
        else:
            # Treat other non-optimal results (like truly infeasible) as a failure
            print("No feasible solution found.")
            high = mid_duration - 1
            search_type = 'binary'



            
    end_time = time.time()
    print(f"\nSearch complete in {end_time - start_time:.2f} seconds.")
    return max_optimal_duration

def main(flex_magnitudes, timesteps, include_banked_results, search_type,generate_plots=True):
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
        bank_file = DATA_DIR_OUTPUTS / "flex_duration_results.csv"
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
                max_dur_steps = find_max_duration(params, data, baseline_df, ts, fm, search_type)
                duration_min = max_dur_steps * params.dt_seconds / 60
                print(f"Timestep {ts}, Flex {fm} kW: Computed max duration {duration_min:.0f} minutes")

            results_list.append({'Timestep': ts, 'Flex_Magnitude_kW': fm, 'Max_Duration_Min': duration_min})

            if max_dur_steps > 0:
                initial_state = baseline_df.loc[ts].to_dict()
                model = build_duration_model(params, data, initial_state, baseline_df, ts, fm, max_dur_steps)
                
                # MODIFIED: Use Pyomo solver for the final detailed run
                solver = pyo.SolverFactory('scip')
                solver.options['limits/time'] = SOLVER_TIME_LIMIT_SECONDS
                results = solver.solve(model, tee=False)
                
                # MODIFIED: Check Pyomo termination condition
                #if results.solver.termination_condition == TerminationCondition.optimal:
                results_df = extract_detailed_results(model, params, data, ts, max_dur_steps, baseline_df)
                
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
    timesteps = [1, 40]#[1]+ list(range(5, 97, 5))  # Start at 1, then every 5th timestep up to 96
    flex_magnitudes =  [-300, -200, -100] #[75, 50, 25, -100, -150, -200, -250, -300, -350, -400, -450, -500]
    include_banked_results = None #"flex_duration_results_new_all.csv"
    main(flex_magnitudes, timesteps, include_banked_results, search_type='binary', generate_plots=True)
    #[10, 20, 25, 30, 35, 40, 50, 55, 60, 70, 75, 80, 85, 90, 95]#