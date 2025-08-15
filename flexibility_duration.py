import pathlib
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Import user-provided dependency modules ---
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles
from integrated_dc_model import ModelParameters
from flexibility_duration_results_and_plots import extract_detailed_results, plot_grid_flex_contributions, save_heatmap_from_results

# --- Path Configuration ------------------------------------------------------
# Define base directories for data and images
DATA_DIR = pathlib.Path("static/data")
IMAGE_DIR = pathlib.Path("static/images")

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# --- Constants and Configuration ---------------------------------------------
DURATION_SIM_HORIZON_MINUTES = 720 # 12 hours should be enough to find max duration
DEBUG_ON_FAIL = True # If True, will re-run failed simulations with verbose output
SOLVER_TIME_LIMIT_SECONDS = 60 # Safeguard to prevent solver from getting stuck

def build_duration_model(params: ModelParameters, data: dict, initial_state: dict, baseline_df: pd.DataFrame, start_timestep: int, flex_target_kw: float, flex_time):
    m = pulp.LpProblem("Flex_Feasibility", pulp.LpMinimize)
    
    # Define the time slots for this specific optimization window
    # Add a buffer of 12 steps for job shifting past the flex_time
    params.T_SLOTS = range(start_timestep, start_timestep + flex_time)
    TEXT_SLOTS = range(start_timestep, start_timestep + flex_time + 12) 
    params.TEXT_SLOTS = TEXT_SLOTS

    # --- Define PuLP Variables ---
    total_cpu = pulp.LpVariable.dicts("TotalCpuUsage", TEXT_SLOTS, lowBound=0, upBound=params.max_cpu_usage)
    p_grid_it = pulp.LpVariable.dicts("P_Grid_IT", TEXT_SLOTS, lowBound=0)
    p_grid_od = pulp.LpVariable.dicts("P_Grid_Other", TEXT_SLOTS, lowBound=0)
    p_it_total = pulp.LpVariable.dicts("P_IT_Total", TEXT_SLOTS, lowBound=0)
    p_ups_ch = pulp.LpVariable.dicts("P_UPS_Charge", TEXT_SLOTS, lowBound=0)
    p_ups_disch = pulp.LpVariable.dicts("P_UPS_Discharge", TEXT_SLOTS, lowBound=0)
    e_ups = pulp.LpVariable.dicts("E_UPS", TEXT_SLOTS, lowBound=params.e_min_kwh, upBound=params.e_max_kwh)
    z_ch = pulp.LpVariable.dicts("Z_Charge", TEXT_SLOTS, cat='Binary')
    z_disch = pulp.LpVariable.dicts("Z_Discharge", TEXT_SLOTS, cat='Binary')
    ut_ks_idx = [(t, k, s) for t in params.T_SLOTS for k in params.K_TRANCHES for s in TEXT_SLOTS if s >= t and s <= t + params.tranche_max_delay[k]]
    ut_ks = pulp.LpVariable.dicts("U_JobTranche", ut_ks_idx, lowBound=0)
    t_it = pulp.LpVariable.dicts("T_IT", TEXT_SLOTS, lowBound=14, upBound=60)
    t_rack = pulp.LpVariable.dicts("T_Rack", TEXT_SLOTS, lowBound=14, upBound=40)
    t_cold_aisle = pulp.LpVariable.dicts("T_ColdAisle", TEXT_SLOTS, lowBound=18, upBound=params.T_cAisle_upper_limit_Celsius)
    t_hot_aisle = pulp.LpVariable.dicts("T_HotAisle", TEXT_SLOTS, lowBound=14, upBound=40)
    e_tes = pulp.LpVariable.dicts("E_TES", TEXT_SLOTS, lowBound=params.E_TES_min_kWh, upBound=params.TES_capacity_kWh)
    p_chiller_hvac = pulp.LpVariable.dicts("P_Chiller_HVAC_Watts", TEXT_SLOTS, lowBound=0)
    p_chiller_tes = pulp.LpVariable.dicts("P_Chiller_TES_Watts", TEXT_SLOTS, lowBound=0)
    q_cool = pulp.LpVariable.dicts("Q_Cool_Watts", TEXT_SLOTS, lowBound=0)
    q_ch_tes = pulp.LpVariable.dicts("Q_Charge_TES_Watts", TEXT_SLOTS, lowBound=0, upBound=params.TES_w_charge_max)
    q_dis_tes = pulp.LpVariable.dicts("Q_Discharge_TES_Watts", TEXT_SLOTS, lowBound=0, upBound=params.TES_w_discharge_max)
    t_in = pulp.LpVariable.dicts("T_Aisle_In", TEXT_SLOTS, lowBound=14, upBound=30)
    # TES mutually-exclusive mode (like the UPS on/off vars)
    z_tes_ch   = pulp.LpVariable.dicts("Z_TES_Charge",    TEXT_SLOTS, lowBound=0, upBound=1, cat="Binary")
    z_tes_disch= pulp.LpVariable.dicts("Z_TES_Discharge", TEXT_SLOTS, lowBound=0, upBound=1, cat="Binary")

    
    # --- Add Constraints ---
    add_it_and_job_constraints(m, params, data, total_cpu, p_it_total, ut_ks)
    add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch, initial_state, start_timestep)
    add_power_balance_constraints(m, params, p_it_total, p_grid_it, p_grid_od, p_ups_disch)
    add_cooling_constraints(
    m, params, initial_state, start_timestep,
    p_it_total, t_it, t_rack, t_cold_aisle, t_hot_aisle, e_tes,
    p_chiller_hvac, p_chiller_tes, q_cool, q_ch_tes, q_dis_tes, t_in,
    z_tes_ch, z_tes_disch
)
    add_power_change_constraints(m, params, flex_target_kw, start_timestep, p_grid_it, p_chiller_hvac, p_chiller_tes, p_grid_od, p_ups_ch, flex_time, baseline_df)
    
    # --- Objective Function ---
    # Simple objective: minimize a dummy variable (or just 0) to find any feasible solution
    # Objective: minimize the sum of grid import during the flexibility period
    m += pulp.lpSum(
        p_grid_it[t] +
        (p_chiller_hvac[t] / 1000.0) +
        (p_chiller_tes[t] / 1000.0) +
        p_grid_od[t] +
        p_ups_ch[t]
        for t in params.T_SLOTS
    ), "Minimize_Grid_Import_Flex_Period"

    return m

"""def get_val(m, var_name):
    var = m.variablesDict().get(var_name)
    return pulp.value(var) if var is not None else 0"""

def add_it_and_job_constraints(m, params, data, total_cpu, p_it_total, ut_ks):
    """
    This function defines the constraints for IT workload, job scheduling, and IT power.
    
    Key Logic Change (Bug Fix):
    1.  The non-shiftable portion of the flexible load within the main flexibility period (T_SLOTS)
        was previously being discarded.
    2.  The `unified_base_load` now correctly includes this non-shiftable flexible load, ensuring
        that 100% of the workload is accounted for.
    """
    p_it_act = pulp.LpVariable.dicts("P_IT_Actual", params.TEXT_SLOTS, lowBound=0)
    p_it_act_ext = pulp.LpVariable.dicts("P_IT_Actual_Extended", params.TEXT_SLOTS, lowBound=None)

    # --- JOB COMPLETION for Shiftable Workload ---
    for t in params.T_SLOTS:
        total_flexible_work_at_t = data['Rt'][t]
        for k in params.K_TRANCHES:
            workload_in_tranche_k = total_flexible_work_at_t * data['shiftabilityProfile'].get((t, k), 0)
            m += pulp.lpSum(ut_ks[(t, k, s)] * params.dt_hours for s in params.TEXT_SLOTS if (t, k, s) in ut_ks) == \
                 workload_in_tranche_k, f"JobCompletion_{t}_{k}"

    # --- UNIFIED BASE LOAD and CPU CALCULATION ---
    # Start with the inflexible load for all periods.
    unified_base_load = data['inflexibleLoadProfile_TEXT'].copy()

    # BUG FIX: Add the non-shiftable portion of the flexible load for the main flexibility period (T_SLOTS).
    # This portion was previously being dropped from the model.
    for t in params.T_SLOTS:
        if t < len(data['Rt']):
            # Get the total flexible load rate (power) at time t
            total_flexible_load_rate = data['Rt'][t] / params.dt_hours
            
            # Calculate the fraction of the load that IS shiftable
            shiftable_fraction = sum(data['shiftabilityProfile'].get((t, k), 0) for k in params.K_TRANCHES)
            
            # The non-shiftable fraction is what's left
            non_shiftable_fraction = 1.0 - shiftable_fraction
            
            # Calculate the non-shiftable load rate and add it to the base load for that slot.
            non_shiftable_load = total_flexible_load_rate * non_shiftable_fraction
            if t < len(unified_base_load):
                unified_base_load[t] += non_shiftable_load

    # Add the entire flexible load for the extension period (it's all treated as non-shiftable).
    extension_slots = [s for s in params.TEXT_SLOTS if s not in params.T_SLOTS]
    for s in extension_slots:
        if s < len(unified_base_load) and s < len(data['flexibleLoadProfile_TEXT']):
             unified_base_load[s] += data['flexibleLoadProfile_TEXT'][s]

    # The total CPU and power calculation is now consistent for all time slots.
    for s in params.TEXT_SLOTS:
        flexible_usage = pulp.lpSum(ut_ks[idx] for idx in ut_ks if idx[2] == s)
        m += total_cpu[s] == unified_base_load[s] + flexible_usage, f"TotalCPUUsage_{s}"

        # Power calculation logic
        if s in params.T_SLOTS:
            m += p_it_act[s] == params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s], f"IT_Power_Primary_{s}"
            m += p_it_act_ext[s] == 0, f"IT_Power_Ext_Zero_{s}"
        else:
            m += p_it_act[s] == 0, f"IT_Power_Primary_Zero_{s}"
            m += p_it_act_ext[s] == (params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s]), f"IT_Power_Ext_Min_{s}"
        
        m += p_it_total[s] == p_it_act[s] + p_it_act_ext[s], f"Total_IT_Power_{s}"

def add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch, initial_state, start_timestep):
    for s in params.TEXT_SLOTS:
        if s == start_timestep:
            prev_energy = initial_state.get('E_UPS_kWh', params.e_start_kwh)
        else:
            prev_energy = e_ups[s-1]
            
        charge = params.eta_ch * p_ups_ch[s] * params.dt_hours
        discharge = (p_ups_disch[s] / params.eta_disch) * params.dt_hours
        m += e_ups[s] == prev_energy + charge - discharge, f"UPS_EnergyBalance_{s}"
        
        m += p_ups_ch[s] <= z_ch[s] * params.p_max_ch_kw, f"UPS_MaxCharge_{s}"
        m += p_ups_ch[s] >= z_ch[s] * params.p_min_ch_kw, f"UPS_MinCharge_{s}"
        m += p_ups_disch[s] <= z_disch[s] * params.p_max_disch_kw, f"UPS_MaxDischarge_{s}"
        m += p_ups_disch[s] >= z_disch[s] * params.p_min_disch_kw, f"UPS_MinDischarge_{s}"
        m += z_ch[s] + z_disch[s] <= 1, f"UPS_ChargeOrDischarge_{s}"
        
    m += e_ups[max(params.TEXT_SLOTS)] >= params.e_start_kwh, "Final_UPS_Energy_Level"

def add_power_balance_constraints(m, params, p_it_total, p_grid_it, p_grid_od, p_ups_disch):
    for s in params.TEXT_SLOTS:
        m += p_it_total[s] == p_grid_it[s] + p_ups_disch[s], f"IT_PowerBalance_{s}"
        m += p_grid_od[s] == p_it_total[s] * params.nominal_overhead_factor, f"Overhead_PowerBalance_{s}"

def add_cooling_constraints(m, params, initial_state, start_timestep,
                            p_it_total, t_it, t_rack, t_cold, t_hot, e_tes,
                            p_hvac, p_tes, q_cool, q_ch_tes, q_dis_tes, t_in,
                            z_tes_ch, z_tes_disch):
    # Pin initial states (you already had these)
    m += t_it[start_timestep]   >= initial_state.get('T_IT_Celsius',   params.T_IT_initial_Celsius)
    m += t_rack[start_timestep] >= initial_state.get('T_Rack_Celsius', params.T_Rack_initial_Celsius)
    m += t_cold[start_timestep] >= initial_state.get('T_cAisle_Celsius', params.T_cAisle_initial)
    m += t_hot[start_timestep]  >= initial_state.get('T_hAisle_Celsius', params.T_hAisle_initial)
    m += e_tes[start_timestep]   == initial_state.get('E_TES_kWh',     params.TES_initial_charge_kWh)

    # --- NEW: bind the first timestep to the physics/capacity ---
    t0 = start_timestep
    mcp = params.m_dot_air * params.c_p_air
    m += q_cool[t0] == (p_hvac[t0] * params.COP_HVAC) + q_dis_tes[t0],           f"CoolingSourceBalance_{t0}"
    m += q_ch_tes[t0] == p_tes[t0] * params.COP_HVAC,                              f"ChillerTESPower_{t0}"
    m += t_in[t0] == t_hot[t0] - q_cool[t0] / mcp,                                 f"AisleTempIn_{t0}"
    m += q_cool[t0] <= (t_hot[t0] - params.T_cAisle_lower_limit_Celsius) * mcp,    f"MaxCoolingDrop_{t0}"
    m += p_tes[t0] + p_hvac[t0] <= params.P_chiller_max,                           f"ChillerCapacity_{t0}"
    m += q_cool[t0] >= (p_it_total[t0] * 1000.0),                                  f"CoolingMeetsITHeat_{t0}"

    # --- NEW: forbid simultaneous TES charge & discharge (all steps) ---
    for s in params.TEXT_SLOTS:
        m += q_ch_tes[s]   <= z_tes_ch[s]    * params.TES_w_charge_max,     f"TES_MaxCharge_{s}"
        m += q_dis_tes[s]  <= z_tes_disch[s] * params.TES_w_discharge_max,  f"TES_MaxDischarge_{s}"
        m += z_tes_ch[s] + z_tes_disch[s] <= 1
    for t in list(params.TEXT_SLOTS)[1:]:
        mcp = params.m_dot_air * params.c_p_air
        m += q_cool[t] == (p_hvac[t] * params.COP_HVAC) + q_dis_tes[t], f"CoolingSourceBalance_{t}"
        m += q_ch_tes[t] == p_tes[t] * params.COP_HVAC, f"ChillerTESPower_{t}"
        m += t_in[t] == t_hot[t] - q_cool[t] / mcp, f"AisleTempIn_{t}"
        m += q_cool[t] <= (t_hot[t] - params.T_cAisle_lower_limit_Celsius) * mcp, f"MaxCoolingDrop_{t}"
        it_heat_watts = p_it_total[t] * 1000.0
        m += t_it[t] == t_it[t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (t_it[t-1] - t_rack[t])) / params.C_IT), f"TempUpdate_IT_{t}"
        m += t_rack[t] == t_rack[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_cold[t]-t_rack[t-1]) + params.G_conv*(t_it[t-1]-t_rack[t-1])) / params.C_Rack), f"TempUpdate_Rack_{t}"
        m += t_cold[t] == t_cold[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_in[t]-t_cold[t-1]) - params.G_cold*(t_cold[t-1]-params.T_out_Celsius)) / params.C_cAisle), f"TempUpdate_ColdAisle_{t}"
        m += t_hot[t] == t_hot[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_rack[t]-t_hot[t-1])) / params.C_hAisle), f"TempUpdate_HotAisle_{t}"
        dE_tes = (q_ch_tes[t]*params.TES_charge_efficiency - q_dis_tes[t]/params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        m += e_tes[t] == e_tes[t-1] + dE_tes, f"EnergyBalance_TES_{t}"
        m += q_dis_tes[t] - q_dis_tes[t-1] <= params.TES_p_dis_ramp, f"Ramp_TES_Discharge_{t}"
        m += q_ch_tes[t] - q_ch_tes[t-1] <= params.TES_p_ch_ramp, f"Ramp_TES_Charge_{t}"
        m += p_tes[t] + p_hvac[t] <= params.P_chiller_max
        m += q_cool[t] >= it_heat_watts
    
    # Ensure final state of TES is same as initial state
    #if 1 in params.TEXT_SLOTS and max(params.TEXT_SLOTS) in e_tes and 1 in e_tes:
    #m += e_tes[max(params.TEXT_SLOTS)] == e_tes[start_timestep], "Cycle_E_TES"

def add_power_change_constraints(m, params, flex_target_kw, start_timestep, p_grid_it, p_chiller_hvac, p_chiller_tes, p_grid_od, p_ups_ch, flex_time, baseline_df):
    
    # A small tolerance to handle floating-point inaccuracies.
    # This allows the new power to be very close to the baseline without being strictly identical.
    tolerance = 0.1 # in kW 

    for t in range(start_timestep, start_timestep + flex_time):
        baseline_power_at_t = baseline_df.loc[t, 'P_Total_kW']
        
        # Define the total power expression for convenience
        current_total_power = (p_grid_it[t] + 
                               (p_chiller_hvac[t] / 1000.0) + 
                               (p_chiller_tes[t] / 1000.0) +   
                               p_grid_od[t] + 
                               p_ups_ch[t])

        # <<< START: MODIFICATION >>>
        if flex_target_kw > 0:
            # For upward flex, power must be AT LEAST the target
            target_power = baseline_power_at_t + flex_target_kw
            m += current_total_power >= target_power, f"Power_Flex_Up_{t}"

        elif flex_target_kw < 0:
            # For downward flex, power must be AT MOST the target
            target_power = baseline_power_at_t + flex_target_kw
            m += current_total_power <= target_power, f"Power_Flex_Down_{t}"

        else: # This is the new block for flex_target_kw == 0
            # For zero flex, power must be EQUAL to the baseline, within a tolerance.
            # This requires two constraints to create an upper and lower bound.
            m += current_total_power <= baseline_power_at_t + tolerance, f"Power_Flex_Zero_Upper_{t}"
            m += current_total_power >= baseline_power_at_t - tolerance, f"Power_Flex_Zero_Lower_{t}"
        # <<< END: MODIFICATION >>>

def load_and_prepare_data(params: ModelParameters):
    """
    Loads input data from CSV files and prepares it for the optimization model.
    """
    # --- Load data from CSV files ---
    try:
        load_profiles_df = pd.read_csv(DATA_DIR / "load_profiles_opt.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR / "shiftability_profile_opt.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profiles.csv' and 'shiftability_profile.csv' are in {DATA_DIR}")
        raise e

    # --- Convert loaded data into the required formats ---
    # Extract load profiles as pandas Series
    inflexible = load_profiles_df['inflexible_load']
    base_flex = load_profiles_df['flexible_load']

    # This seems to be a placeholder, will be defined properly inside the model build
    if not hasattr(params, 'T_SLOTS'):
        params.T_SLOTS = range(1, 97) # Default to a full day if not set

    # Create the subset for the primary simulation time (T_SLOTS)
    # Handle the case where T_SLOTS might not be continuous or start at 1
    valid_tslots = [ts for ts in params.T_SLOTS if ts in base_flex.index]
    base_flex_t = base_flex.loc[valid_tslots]


    # Convert the shiftability DataFrame back into a dictionary with tuple keys
    shiftability_df.columns = shiftability_df.columns.astype(int)
    shiftability = shiftability_df.stack().to_dict()

    # --- Resample and prepare data structures for the model ---
    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    
    # These need to be set before calling this function
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
    
    # Ensure lengths match before adding
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

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)

def find_max_duration(params, data, baseline_df, start_timestep, flex_kw):
    """
    Uses a binary search to find the maximum feasible duration.
    Returns the max duration in steps.
    """
    print(f"\n--- Searching for Max Duration: Timestep {start_timestep}, Flex {flex_kw} kW ---")
    
    initial_state = baseline_df.loc[start_timestep].to_dict()
    
    low = 1
    high = len(baseline_df) - start_timestep - 11
    
    max_optimal_duration = 0

    start_time = time.time()
    
    while low <= high:
        mid_duration = (low + high) // 2
        if mid_duration == 0: break

        print(f"\rTesting duration: {mid_duration} steps... ", end="")
        
        model = build_duration_model(params, data, initial_state, baseline_df, start_timestep, flex_kw, mid_duration)
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_SECONDS)
        model.solve(solver)
        
        if pulp.LpStatus[model.status] == 'Optimal':
            max_optimal_duration = mid_duration
            low = mid_duration + 1
        else:
            high = mid_duration - 1
            
    end_time = time.time()
    print(f"\nSearch complete in {end_time - start_time:.2f} seconds.")

    return max_optimal_duration



def main(flex_magnitudes, timesteps, include_banked_results, generate_plots=True):

    params = ModelParameters()
    params.tranche_max_delay = {x:x for x in range(1,13)}
    params.K_TRANCHES = range(1, 13)
    
    baseline_csv_path = DATA_DIR / "optimised_baseline.csv"
    
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        if 'Time_Slot_EXT' in baseline_df.columns and baseline_df.index.name != 'Time_Slot_EXT':
            baseline_df = baseline_df.set_index('Time_Slot_EXT')
    except FileNotFoundError:
        print(f"Error: '{baseline_csv_path}' not found. Please run 'integrated_dc_model.py' first.")
        return  # early-out if we can't continue

    # Prepare data for the model
    params.T_SLOTS = range(1, 97)
    data = load_and_prepare_data(params)

    # Load (optional) bank of previous results
    if include_banked_results is None:
        bank_file = DATA_DIR / "flex_duration_results.csv"
        old_results = pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])
    else:
        bank_file = DATA_DIR / include_banked_results
        if bank_file.exists():
            old_results = pd.read_csv(bank_file)
        else:
            old_results = pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])

    old_results = old_results.set_index(['Timestep', 'Flex_Magnitude_kW'])

    results = []
    for ts in timesteps:
        for fm in flex_magnitudes:
            key = (ts, fm)
            if include_banked_results is not None and key in old_results.index:
                duration_min = float(old_results.loc[key, 'Max_Duration_Min'])
                max_dur_steps = int(duration_min / (params.dt_seconds / 60))
                print(f"Timestep {ts}, Flex {fm} kW: Using banked duration {duration_min:.0f} minutes")
            else:
                max_dur_steps = find_max_duration(params, data, baseline_df, ts, fm)
                duration_min = max_dur_steps * params.dt_seconds / 60
                print(f"Timestep {ts}, Flex {fm} kW: Computed max duration {duration_min:.0f} minutes")

            results.append({'Timestep': ts, 'Flex_Magnitude_kW': fm, 'Max_Duration_Min': duration_min})

            # (Optional) extract one detailed run at max duration
            if max_dur_steps > 0:
                initial_state = baseline_df.loc[ts].to_dict()
                model = build_duration_model(params, data, initial_state, baseline_df, ts, fm, max_dur_steps)
                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_SECONDS)
                model.solve(solver)
                if pulp.LpStatus[model.status] == 'Optimal':
                    results_df = extract_detailed_results(model, params, data, ts, max_dur_steps, baseline_df)
                    csv_filename = f"flex_duration_detailed_results_ts{ts}_flex{str(fm).replace('-', 'neg')}.csv"
                    results_df.to_csv(DATA_DIR / csv_filename)
                    print(f"  -> Saved detailed results to {csv_filename}")
                    if generate_plots:
                        plot_grid_flex_contributions(results_df, ts, fm, max_dur_steps)

    # After the sweep, save the long-form CSV and the heatmap
    save_heatmap_from_results(
        results_rows=results,
        csv_path=DATA_DIR / "flex_duration_results.csv",
        png_path=IMAGE_DIR / "flex_duration_heatmap.png"
    )

if __name__ == '__main__':
    timesteps = [1] # + list(range(5, 100, 5))
    flex_magnitudes = [-100]#list(range(-500, -49, 50)) + [25, 50, 75]
    include_banked_results =  None #"flex_duration_results.csv" # to reuse prior runs
    main(flex_magnitudes, timesteps, include_banked_results, generate_plots=True)