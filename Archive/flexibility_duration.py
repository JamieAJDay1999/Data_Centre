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
    p_total = pulp.LpVariable.dicts("P_Total", TEXT_SLOTS, lowBound=0)
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
    
    # --- Add Constraints ---
    add_it_and_job_constraints(m, params, data, total_cpu, p_it_total, ut_ks)
    add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch, initial_state, start_timestep)
    add_power_balance_constraints(m, params, p_it_total, p_grid_it, p_grid_od, p_ups_disch)
    add_cooling_constraints(m, params, p_it_total, initial_state, start_timestep, t_it, t_rack, t_cold_aisle, t_hot_aisle, e_tes, p_chiller_hvac, p_chiller_tes, q_cool, q_ch_tes, q_dis_tes, t_in)
    add_power_change_constraints(m, params, flex_target_kw, start_timestep, p_total, p_grid_it, p_chiller_hvac, p_chiller_tes, p_grid_od, p_ups_ch, flex_time, baseline_df)
    
    # --- Objective Function ---
        # --- Define Objective Function ---
    # Minimize the total energy cost over the optimization window
    m += pulp.lpSum(
        params.dt_hours * (
            p_grid_it[s] +
            (p_chiller_hvac[s] / 1000.0) +
            (p_chiller_tes[s] / 1000.0) +
            p_grid_od[s] +
            p_ups_ch[s]
        ) * (data['electricity_price'][s] / 1000.0)
        for s in TEXT_SLOTS
    ), "Total_Energy_Cost"

    return m


def add_it_and_job_constraints(m, params, data, total_cpu, p_it_total, ut_ks):
    p_it_act=pulp.LpVariable.dicts("P_IT_Actual",params.TEXT_SLOTS,lowBound=0)
    p_it_act_ext=pulp.LpVariable.dicts("P_IT_Actual_Extended",params.TEXT_SLOTS,lowBound=None)
    print(params.K_TRANCHES)
    print(params.tranche_max_delay)
    for t in params.T_SLOTS:
        for k in params.K_TRANCHES:
            #m+=pulp.lpSum(ut_ks[(t,k,s)]*params.dt_hours for s in params.TEXT_SLOTS if(t,k,s)in ut_ks)== data['Rt'][t] * data['shiftabilityProfile'].get((t,k),0),f"JobCompletion_{t}_{k}"
            # In flexibility_duration.py -> add_it_and_job_constraints()
            # The right-hand side is now the value from the profile, converted to kWh
            workload_in_tranche_k = data['shiftabilityProfile'].get((t, k), 0) * params.dt_hours
            m += pulp.lpSum(ut_ks[(t, k, s)] * params.dt_hours for s in params.TEXT_SLOTS if (t,k,s) in ut_ks) == \
     workload_in_tranche_k, f"JobCompletion_{t}_{k}"
    for s in params.TEXT_SLOTS:
        flexible_usage=pulp.lpSum(ut_ks[idx]for idx in ut_ks if idx[2]==s)
        m+=total_cpu[s]==data['inflexibleLoadProfile_TEXT'][s]+flexible_usage,f"TotalCPUUsage_{s}"
        
        if s in params.T_SLOTS:
            m+=p_it_act[s]==params.idle_power_kw+(params.max_power_kw-params.idle_power_kw)*total_cpu[s],f"IT_Power_Primary_{s}"
            m+=p_it_act_ext[s]==0,f"IT_Power_Ext_Zero_{s}"
        else:
            m+=p_it_act[s]==0,f"IT_Power_Primary_Zero_{s}"
            nominal_power_in_ext=data['Pt_IT_nom_TEXT'][s]
            m+=p_it_act_ext[s]>=(params.idle_power_kw+(params.max_power_kw-params.idle_power_kw)*total_cpu[s])-nominal_power_in_ext,f"IT_Power_Ext_Min_{s}"
        m+=p_it_total[s]==p_it_act[s]+p_it_act_ext[s],f"Total_IT_Power_{s}"

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

def add_cooling_constraints(m, params, p_it_total, initial_state, start_timestep, t_it, t_rack, t_cold, t_hot, e_tes, p_hvac, p_tes, q_cool, q_ch_tes, q_dis_tes, t_in):
    m += t_it[start_timestep] >= initial_state.get('T_IT_Celsius', params.T_IT_initial_Celsius)
    m += t_rack[start_timestep] >= initial_state.get('T_Rack_Celsius', params.T_Rack_initial_Celsius)
    m += t_cold[start_timestep] >= initial_state.get('T_ColdAisle_Celsius', params.T_cAisle_initial)
    m += t_hot[start_timestep] >= initial_state.get('T_HotAisle_Celsius', params.T_hAisle_initial)
    m += e_tes[start_timestep] >= initial_state.get('E_TES_kWh', params.TES_initial_charge_kWh)
    
    mcp = params.m_dot_air * params.c_p_air
    
    for t in list(params.TEXT_SLOTS)[1:]:
        m += q_cool[t] == (p_hvac[t] * params.COP_HVAC) + q_dis_tes[t], f"CoolingSourceBalance_{t}"
        m += q_ch_tes[t] == p_tes[t] * params.COP_HVAC, f"ChillerTESPower_{t}"
        m += t_in[t] == t_hot[t-1] - q_cool[t] / mcp, f"AisleTempIn_{t}"
        m += q_cool[t] <= (t_hot[t-1] - params.T_cAisle_lower_limit_Celsius) * mcp, f"MaxCoolingDrop_{t}"
        it_heat_watts = p_it_total[t] * 1000.0
        m += t_it[t] == t_it[t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (t_it[t-1] - t_rack[t-1])) / params.C_IT), f"TempUpdate_IT_{t}"
        m += t_rack[t] == t_rack[t-1] + params.dt_seconds * ((params.m_dot_air * params.kappa * params.c_p_air * (t_cold[t-1] - t_rack[t-1]) + params.G_conv * (t_it[t-1] - t_rack[t-1])) / params.C_Rack), f"TempUpdate_Rack_{t}"
        m += t_cold[t] == t_cold[t-1] + params.dt_seconds * ((params.m_dot_air * params.kappa * params.c_p_air * (t_in[t] - t_cold[t-1]) - params.G_cold * (t_cold[t-1] - params.T_out_Celsius)) / params.C_cAisle), f"TempUpdate_ColdAisle_{t}"
        m += t_hot[t] == t_hot[t-1] + params.dt_seconds * ((params.m_dot_air * params.kappa * params.c_p_air * (t_rack[t-1] - t_hot[t-1])) / params.C_hAisle), f"TempUpdate_HotAisle_{t}"
        dE_tes = (q_ch_tes[t] * params.TES_charge_efficiency - q_dis_tes[t] / params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        m += e_tes[t] == e_tes[t-1] + dE_tes, f"EnergyBalance_TES_{t}"
        m += q_dis_tes[t] - q_dis_tes[t-1] <= params.TES_p_dis_ramp, f"Ramp_TES_Discharge_{t}"
        m += q_ch_tes[t] - q_ch_tes[t-1] <= params.TES_p_ch_ramp, f"Ramp_TES_Charge_{t}"
        m += p_tes[t] + p_hvac[t] <= params.P_chiller_max
        m += q_cool[t] >= it_heat_watts

def add_power_change_constraints(m, params, flex_target_kw, start_timestep, p_total, p_grid_it, p_chiller_hvac, p_chiller_tes, p_grid_od, p_ups_ch, flex_time, baseline_df):
    for t in range(start_timestep, start_timestep + flex_time):
        m += p_total[t] == (p_grid_it[t] + 
                            (p_chiller_hvac[t] / 1000.0) + 
                            (p_chiller_tes[t] / 1000.0) + 
                            p_grid_od[t] + 
                            p_ups_ch[t]), f"Total_Power_Sum_{t}"
        
        baseline_power_at_t = baseline_df.loc[t, 'P_Total_kW']
        target_power = baseline_power_at_t - flex_target_kw
        
        if flex_target_kw > 0:
            m += p_total[t] <= target_power, f"Flex_Down_{t}"
        else:
            m += p_total[t] >= target_power, f"Flex_Up_{t}"

    # Extend p_total definition to the buffer period (no flex constraints, but for extraction)
    for t in range(start_timestep + flex_time, max(params.TEXT_SLOTS) + 1):
        m += p_total[t] == (p_grid_it[t] + 
                            (p_chiller_hvac[t] / 1000.0) + 
                            (p_chiller_tes[t] / 1000.0) + 
                            p_grid_od[t] + 
                            p_ups_ch[t]), f"Total_Power_Sum_Ext_{t}"

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

    # Create the subset for the primary simulation time (T_SLOTS)
    base_flex_t = base_flex.loc[list(params.T_SLOTS)]

    # Convert the shiftability DataFrame back into a dictionary with tuple keys
    shiftability_df.columns = shiftability_df.columns.astype(int)
    shiftability = shiftability_df.stack().to_dict()

    # --- Resample and prepare data structures for the model ---
    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    
    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'flexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0),
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
    high = len(baseline_df) - start_timestep - 15 
    
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

def extract_flex_details(m, params):
    flexible_load_details = []
    for v in m.variables():
        if v.name.startswith("U_JobTranche") and v.value() is not None and v.value() > 1e-6:
            try:
                parts = v.name.replace("U_JobTranche_", "").strip("()").split(",_")
                if len(parts) == 3:
                    t, k, s = map(int, parts)
                    lag = s - t
                    shiftability = params.tranche_max_delay[k] - lag
                    flexible_load_details.append({
                        'processing_slot': s,
                        'original_slot': t,
                        'tranche': k,
                        'cpu_load': v.value(),
                        'lag': lag,
                        'shiftability': shiftability
                    })
            except (ValueError, IndexError):
                continue
    flex_df = pd.DataFrame(flexible_load_details).sort_values(by=['processing_slot', 'tranche']).reset_index(drop=True)
    return flex_df

def extract_contribs(m, baseline_df, start_timestep, flex_time, is_flex_down):
    contribs = []
    sources = ['Grid_IT', 'Chiller_HVAC', 'Chiller_TES', 'Grid_Other', 'UPS_Charge']
    text_slots = range(start_timestep, start_timestep + flex_time + 12)  # Align with TEXT_SLOTS
    for t in text_slots:
        row = {'timestep': t}
        for source in sources:
            if source == 'Grid_IT':
                opt = pulp.value(m.variablesDict()[f'P_Grid_IT_{t}'])
                base = baseline_df.loc[t, 'P_Grid_IT_kW'] if t in baseline_df.index else 0
            elif source == 'Chiller_HVAC':
                opt = pulp.value(m.variablesDict()[f'P_Chiller_HVAC_Watts_{t}']) / 1000
                base = baseline_df.loc[t, 'P_Chiller_HVAC_kW'] if t in baseline_df.index else 0
            elif source == 'Chiller_TES':
                opt = pulp.value(m.variablesDict()[f'P_Chiller_TES_Watts_{t}']) / 1000
                base = baseline_df.loc[t, 'P_Chiller_TES_kW'] if t in baseline_df.index else 0
            elif source == 'Grid_Other':
                opt = pulp.value(m.variablesDict()[f'P_Grid_Other_{t}'])
                base = baseline_df.loc[t, 'P_Grid_Other_kW'] if t in baseline_df.index else 0
            elif source == 'UPS_Charge':
                opt = pulp.value(m.variablesDict()[f'P_UPS_Charge_{t}'])
                base = baseline_df.loc[t, 'P_UPS_Charge_kW'] if t in baseline_df.index else 0
            
            if is_flex_down:
                diff = base - opt
            else:
                diff = opt - base
            
            row[f'{source}_base'] = base
            row[f'{source}_opt'] = opt
            row[f'{source}_diff'] = diff
        
        contribs.append(row)
    df = pd.DataFrame(contribs).set_index('timestep')
    return df

def plot_contrib(df, start_ts, flex_kw, dur_steps):
    diff_cols = [col for col in df.columns if col.endswith('_diff')]
    plot_df = df[diff_cols].rename(columns={col: col.replace('_diff', '') for col in diff_cols})
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Flexibility Contributions: Start {start_ts}, Flex {flex_kw} kW, Duration {dur_steps} steps (incl. extension)")
    ax.set_ylabel("Power Deviation (kW)")
    ax.set_xlabel("Timestep")
    
    # Add horizontal dotted black line at flexibility magnitude
    ax.axhline(y=abs(flex_kw), color='black', linestyle=':')
    
    # Add scatter plot with grey "X"s at the sum of contributions
    total_flex = plot_df.sum(axis=1)
    ax.scatter(range(len(plot_df)), total_flex, color='grey', marker='X', s=50)
    
    plt.tight_layout()
    filename = f"flex_contrib_start{start_ts}_flex{str(flex_kw).replace('-', 'neg')}.png"
    plt.savefig(IMAGE_DIR / filename)
    plt.show()
    plt.close()

def plot_workload_composition(flex_df, contrib_df, baseline_df, params, data, start_ts, fm, dur_steps):
    flex_period_index = contrib_df.index  # Now includes extension
    inflexible_base = pd.Series([data['inflexibleLoadProfile_TEXT'][s] for s in flex_period_index], index=flex_period_index)
    forced_flex = flex_df[flex_df['shiftability'] == 0].groupby('processing_slot')['cpu_load'].sum().reindex(flex_period_index, fill_value=0)
    flex_pivot_by_lag = flex_df[flex_df['processing_slot'].isin(flex_period_index)].pivot_table(
        index='processing_slot',
        columns='lag',
        values='cpu_load',
        aggfunc='sum'
    ).fillna(0)
    plot_df = pd.DataFrame({
        'Inflexible': inflexible_base + forced_flex
    }, index=flex_period_index)
    plot_df = plot_df.join(flex_pivot_by_lag).fillna(0)
    rename_dict = {lag: f'Flexible (Lag {int(lag)})' for lag in flex_pivot_by_lag.columns}
    plot_df.rename(columns=rename_dict, inplace=True)
    
    fig, ax = plt.subplots(figsize=(18, 9))
    colors = ['black'] + list(plt.cm.viridis(np.linspace(0, 1, len(plot_df.columns) - 1)))
    plot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors)
    
    baseline_total = baseline_df.loc[plot_df.index.intersection(baseline_df.index), 'Total_CPU_Load'].reindex(plot_df.index, fill_value=0).values
    ax.plot(range(len(baseline_total)), baseline_total, label='Baseline Total Workload', linestyle='--', color='gray')
    
    ax.set_title(f'Optimized IT Workload Composition for Flex: Start {start_ts}, Flex {fm} kW, Duration {dur_steps} steps (incl. extension)')
    ax.set_xlabel('Processing Time Slot')
    ax.set_ylabel('CPU Load Units')
    ax.legend(title='Load Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    tick_frequency = max(1, len(plot_df) // 24)
    ax.set_xticks(range(0, len(plot_df), tick_frequency))
    ax.set_xticklabels(plot_df.index[::tick_frequency], rotation=45, ha="right")
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plot_filename = f"it_load_stacked_start{start_ts}_flex{str(fm).replace('-', 'neg')}.png"
    fig.savefig(IMAGE_DIR / plot_filename)
    plt.show()
    plt.close()

def reproduce_heatmap(csv_path):
    """
    Reads the flexibility duration results from a CSV, converts durations from minutes to slots (divide by 15),
    and reproduces the heatmap visualization.
    
    Args:
        csv_path (str): Path to the CSV file containing the results.
    """
    # Read the results
    results_df = pd.read_csv(csv_path)
    
    # Convert durations from minutes to slots (timesteps)
    results_df['Max_Duration_Slots'] = results_df['Max_Duration_Min'] / 15
    
    # Pivot for heatmap (using the new slots column)
    pivot_df = results_df.pivot(index='Flex_Magnitude_kW', columns='Timestep', values='Max_Duration_Slots')
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title('Max Flexibility Duration (Slots) Heatmap')
    plt.ylabel('Flex Magnitude (kW)')
    plt.xlabel('Timestep')
    #plt.show()

# --- Main Execution Block ----------------------------------------------------
def main(flex_magnitudes, timesteps, include_banked_results, generate_plots=True):

    params = ModelParameters()
    params.tranche_max_delay = {x:x for x in range(1,13)}
    params.K_TRANCHES = range(1, 13)
    
    # Define the path for the input CSV file
    baseline_csv_path = DATA_DIR / "optimised_baseline.csv"
    
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        if 'Time_Slot_EXT' in baseline_df.columns and baseline_df.index.name != 'Time_Slot_EXT':
             baseline_df = baseline_df.set_index('Time_Slot_EXT')
    except FileNotFoundError:
        print(f"Error: '{baseline_csv_path}' not found. Please run 'integrated_dc_model.py' first.")
        baseline_df = None

    if baseline_df is not None:
        data = load_and_prepare_data(params)

        # Fix: Add non-shiftable flexible load to inflexible profile
        """for t in params.T_SLOTS:
            shift_sum = sum(data['shiftabilityProfile'].get((t, k), 0) for k in params.K_TRANCHES)
            non_shiftable = data['Rt'][t] * (1 - shift_sum)
            data['inflexibleLoadProfile_TEXT'][t] += non_shiftable  # Assuming Rt is rate; adjust if work units"""
 
        
        # Determine bank file
        if include_banked_results is None:
            bank_file = DATA_DIR / "flex_duration_results.csv"
            old_results = pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])
        else:
            bank_file = DATA_DIR / include_banked_results
            if bank_file.exists():
                old_results = pd.read_csv(bank_file)
            else:
                old_results = pd.DataFrame(columns=['Timestep', 'Flex_Magnitude_kW', 'Max_Duration_Min'])

        # Prepare old_results for quick lookup
        old_results = old_results.set_index(['Timestep', 'Flex_Magnitude_kW'])

        results = []
        for ts in timesteps:
            for fm in flex_magnitudes:
                key = (ts, fm)
                if include_banked_results is not None and key in old_results.index:
                    duration_min = old_results.loc[key, 'Max_Duration_Min']
                    max_dur_steps = int(duration_min / (params.dt_seconds / 60))
                    print(f"Timestep {ts}, Flex {fm} kW: Using banked duration {duration_min:.0f} minutes")
                else:
                    max_dur_steps = find_max_duration(params, data, baseline_df, ts, fm)
                    duration_min = max_dur_steps * params.dt_seconds / 60
                    print(f"Timestep {ts}, Flex {fm} kW: Computed max duration {duration_min:.0f} minutes")
                results.append({'Timestep': ts, 'Flex_Magnitude_kW': fm, 'Max_Duration_Min': duration_min})

                if max_dur_steps > 0:
                    initial_state = baseline_df.loc[ts].to_dict()
                    model = build_duration_model(params, data, initial_state, baseline_df, ts, fm, max_dur_steps)
                    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_SECONDS)
                    model.solve(solver)
                    if pulp.LpStatus[model.status] == 'Optimal':
                        is_flex_down = fm > 0
                        contrib_df = extract_contribs(model, baseline_df, ts, max_dur_steps, is_flex_down)
                        flex_df = extract_flex_details(model, params)
                        flex_period_index = contrib_df.index
                        inflexible_base = pd.Series([data['inflexibleLoadProfile_TEXT'][s] for s in flex_period_index], index=flex_period_index)
                        forced_flex = flex_df[flex_df['shiftability'] == 0].groupby('processing_slot')['cpu_load'].sum().reindex(flex_period_index, fill_value=0)
                        contrib_df['Inflexible_Load_CPU_Base'] = baseline_df.loc[contrib_df.index.intersection(baseline_df.index), 'Inflexible_Load_CPU_Opt'].reindex(contrib_df.index, fill_value=0)
                        contrib_df['Flexible_Load_CPU_Base'] = baseline_df.loc[contrib_df.index.intersection(baseline_df.index), 'Flexible_Load_CPU_Opt'].reindex(contrib_df.index, fill_value=0)
                        contrib_df['Total_CPU_Load_Base'] = baseline_df.loc[contrib_df.index.intersection(baseline_df.index), 'Total_CPU_Load'].reindex(contrib_df.index, fill_value=0)
                        contrib_df['Inflexible_Load_CPU_New'] = inflexible_base + forced_flex
                        contrib_df['Flexible_Load_CPU_New'] = flex_df.groupby('processing_slot')['cpu_load'].sum().reindex(flex_period_index, fill_value=0) - forced_flex
                        contrib_df['Total_CPU_Load_New'] = contrib_df['Inflexible_Load_CPU_New'] + contrib_df['Flexible_Load_CPU_New']
                        csv_filename = f"flex_contrib_data_start{ts}_flex{str(fm).replace('-', 'neg')}.csv"
                        contrib_df.to_csv(DATA_DIR / csv_filename)
                        if generate_plots:
                            plot_contrib(contrib_df, ts, fm, max_dur_steps)
                            plot_workload_composition(flex_df, contrib_df, baseline_df, params, data, ts, fm, max_dur_steps)
        
        results_df = pd.DataFrame(results)
        
        # Update the bank: concat old and new, drop duplicates
        full_bank = pd.concat([old_results.reset_index(), results_df], ignore_index=True)
        full_bank.drop_duplicates(subset=['Timestep', 'Flex_Magnitude_kW'], keep='last', inplace=True)
        full_bank.to_csv(bank_file, index=False)
        
        # For plotting, use only user-requested combinations (which are in results_df)
        plot_df = results_df
        
        # Visualization 1: Heatmap
        pivot_df = plot_df.pivot(index='Flex_Magnitude_kW', columns='Timestep', values='Max_Duration_Min')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title('Max Flexibility Duration (Minutes) Heatmap')
        plt.ylabel('Flex Magnitude (kW)')
        plt.xlabel('Timestep')
        plt.savefig(IMAGE_DIR / "flex_duration_heatmap.png")
        #plt.show()
        
    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    flex_magnitudes = [100]   # Positive for upward (reduce power), negative for downward (increase power)
    timesteps = [1]  # Example timesteps to analyze
    include_banked_results = None #"flex_duration_results.csv"  # Filename to load/save bank, or None to not use bank
    main(flex_magnitudes, timesteps, include_banked_results, generate_plots=True)
    #reproduce_heatmap(DATA_DIR / "flex_duration_results.csv")