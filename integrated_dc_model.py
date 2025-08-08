import datetime
import json
import pathlib
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
from parameters_optimisation import setup_simulation_parameters

# --- Path Configuration ------------------------------------------------------
# Define base directories for data, images, and debugging
DATA_DIR = pathlib.Path("static/data")
IMAGE_DIR = pathlib.Path("static/images2")
DEBUG_DIR = pathlib.Path("lp_debug")

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


# --- Model Configuration -----------------------------------------------------
CYCLE_TES_ENERGY = True

# --- Parameter Management Class ----------------------------------------------
class ModelParameters:
    """A class to hold and manage all model parameters and derived constants."""
    def __init__(self, simulation_minutes=1440, dt_seconds=900, extended_horizon_minutes=180):
        # --- Time Horizon ---
        self.simulation_minutes = simulation_minutes
        self.dt_seconds = dt_seconds
        self.extended_horizon_minutes = extended_horizon_minutes
        self.dt_hours = self.dt_seconds / 3600.0
        self.sim_minutes_ext = self.simulation_minutes + self.extended_horizon_minutes
        self.num_steps_extended = int(self.sim_minutes_ext * 60 / self.dt_seconds)

        # Time slots (1-based indexing for PuLP readability)
        self.T_SLOTS = range(1, 1 + int(self.simulation_minutes * 60 / self.dt_seconds))
        self.TEXT_SLOTS = range(1, 1 + self.num_steps_extended)
        self.K_TRANCHES = range(1, 5)

        # --- IT Equipment ---
        self.idle_power_kw = 166.7
        self.max_power_kw = 1000.0
        self.max_cpu_usage = 1.0
        self.tranche_max_delay = {1: 2, 2: 4, 3: 8, 4: 12}
        self.nominal_overhead_factor = 0.1 # For other DC loads (lighting, etc.)

        # --- UPS / Battery Storage ---
        self.eta_ch = 0.82
        self.eta_disch = 0.92
        self.e_nom_kwh = 600.0
        self.soc_min = 0.5
        self.soc_max = 1.0
        self.e_start_kwh = 600.0
        self.p_max_ch_kw = 270.0
        self.p_max_disch_kw = 2700.0
        self.p_min_ch_kw = 40.0
        self.p_min_disch_kw = 100.0
        self.e_min_kwh = self.soc_min * self.e_nom_kwh
        self.e_max_kwh = self.soc_max * self.e_nom_kwh

        # --- Cooling System (from external file) ---
        cooling_params = setup_simulation_parameters("cool_down")
        self.__dict__.update(cooling_params) # Merges the cooling params into this class

        self.TES_capacity_kWh = self.TES_kwh_cap


def build_model(params: ModelParameters, data: dict):
    """
    Builds the PuLP optimization model, defining all variables, constraints, and the objective.
    """
    m = pulp.LpProblem("DC_Cost_Optimization", pulp.LpMinimize)
    TEXT_SLOTS = params.TEXT_SLOTS

    # IT & Power Grid Variables
    total_cpu = pulp.LpVariable.dicts("TotalCpuUsage", TEXT_SLOTS, lowBound=0, upBound=params.max_cpu_usage)
    p_grid_it = pulp.LpVariable.dicts("P_Grid_IT", TEXT_SLOTS, lowBound=0)
    p_grid_od = pulp.LpVariable.dicts("P_Grid_Other", TEXT_SLOTS, lowBound=0)
    p_it_total = pulp.LpVariable.dicts("P_IT_Total", TEXT_SLOTS, lowBound=0)

    # UPS Variables
    p_ups_ch = pulp.LpVariable.dicts("P_UPS_Charge", TEXT_SLOTS, lowBound=0)
    p_ups_disch = pulp.LpVariable.dicts("P_UPS_Discharge", TEXT_SLOTS, lowBound=0)
    e_ups = pulp.LpVariable.dicts("E_UPS", TEXT_SLOTS, lowBound=params.e_min_kwh, upBound=params.e_max_kwh)
    z_ch = pulp.LpVariable.dicts("Z_Charge", TEXT_SLOTS, cat='Binary')
    z_disch = pulp.LpVariable.dicts("Z_Discharge", TEXT_SLOTS, cat='Binary')

    # Job Scheduling Variables
    ut_ks_idx = [(t, k, s) for t in params.T_SLOTS for k in params.K_TRANCHES for s in TEXT_SLOTS if s >= t and s <= t + params.tranche_max_delay[k]]
    ut_ks = pulp.LpVariable.dicts("U_JobTranche", ut_ks_idx, lowBound=0)

    # Cooling System Variables
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
    add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch)
    add_power_balance_constraints(m, params, p_it_total, p_grid_it, p_grid_od, p_ups_disch)
    add_cooling_constraints(m, params, p_it_total, t_it, t_rack, t_cold_aisle, t_hot_aisle, e_tes, p_chiller_hvac, p_chiller_tes, q_cool, q_ch_tes, q_dis_tes, t_in)

    # --- Define Objective Function ---
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
    p_it_act = pulp.LpVariable.dicts("P_IT_Actual", params.TEXT_SLOTS, lowBound=0)
    p_it_act_ext = pulp.LpVariable.dicts("P_IT_Actual_Extended", params.TEXT_SLOTS, lowBound=None)
    for t in params.T_SLOTS:
        for k in params.K_TRANCHES:
            m += pulp.lpSum(ut_ks[(t, k, s)] * params.dt_hours for s in params.TEXT_SLOTS if (t,k,s) in ut_ks) == data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0), f"JobCompletion_{t}_{k}"

    for s in params.TEXT_SLOTS:
        flexible_usage = pulp.lpSum(ut_ks[idx] for idx in ut_ks if idx[2] == s)
        m += total_cpu[s] == data['inflexibleLoadProfile_TEXT'][s] + flexible_usage, f"TotalCPUUsage_{s}"
        if s in params.T_SLOTS:
            m += p_it_act[s] == params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s], f"IT_Power_Primary_{s}"
            m += p_it_act_ext[s] == 0, f"IT_Power_Ext_Zero_{s}"
        else:
            m += p_it_act[s] == 0, f"IT_Power_Primary_Zero_{s}"
            nominal_power_in_ext = data['Pt_IT_nom_TEXT'][s]
            #m += p_it_act_ext[s] >= (params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s]) - nominal_power_in_ext, f"IT_Power_Ext_Min_{s}"
            m += p_it_act_ext[s] >= (params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s]), f"IT_Power_Ext_Min_{s}"
        m += p_it_total[s] == p_it_act[s] + p_it_act_ext[s], f"Total_IT_Power_{s}"

def add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch):
    for s in params.TEXT_SLOTS:
        prev_energy = params.e_start_kwh if s == 1 else e_ups[s-1]
        charge = params.eta_ch * p_ups_ch[s] * params.dt_hours
        discharge = (p_ups_disch[s] / params.eta_disch) * params.dt_hours
        m += e_ups[s] == prev_energy + charge - discharge, f"UPS_EnergyBalance_{s}"
        m += p_ups_ch[s] <= z_ch[s] * params.p_max_ch_kw, f"UPS_MaxCharge_{s}"
        m += p_ups_ch[s] >= z_ch[s] * params.p_min_ch_kw, f"UPS_MinCharge_{s}"
        m += p_ups_disch[s] <= z_disch[s] * params.p_max_disch_kw, f"UPS_MaxDischarge_{s}"
        m += p_ups_disch[s] >= z_disch[s] * params.p_min_disch_kw, f"UPS_MinDischarge_{s}"
        m += z_ch[s] + z_disch[s] <= 1, f"UPS_ChargeOrDischarge_{s}"
    m += e_ups[max(params.TEXT_SLOTS)] == params.e_start_kwh, "Final_UPS_Energy_Level"

def add_power_balance_constraints(m, params, p_it_total, p_grid_it, p_grid_od, p_ups_disch):
    for s in params.TEXT_SLOTS:
        m += p_it_total[s] == p_grid_it[s] + p_ups_disch[s], f"IT_PowerBalance_{s}"
        m += p_grid_od[s] == p_it_total[s] * params.nominal_overhead_factor, f"Overhead_PowerBalance_{s}"

def add_cooling_constraints(m, params, p_it_total, t_it, t_rack, t_cold, t_hot, e_tes, p_hvac, p_tes, q_cool, q_ch_tes, q_dis_tes, t_in):
    m += t_it[1] >= params.T_IT_initial_Celsius
    m += t_rack[1] >= params.T_Rack_initial_Celsius
    m += t_cold[1] >= params.T_cAisle_initial
    m += t_hot[1] >= params.T_hAisle_initial
    m += e_tes[1] >= params.TES_initial_charge_kWh
    mcp = params.m_dot_air * params.c_p_air
    m += p_hvac[1] == np.mean([p_hvac[k] for k in list(p_hvac.keys())[1:]]), f"HVAC_PowerBalance_{1}"
    m += p_tes[1] == np.mean([p_tes[k] for k in list(p_tes.keys())[1:]]), f"TES_PowerBalance_{1}"
    for t in params.TEXT_SLOTS[1:]:
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
    if CYCLE_TES_ENERGY:
        m += e_tes[max(params.TEXT_SLOTS)] == e_tes[1], "Cycle_E_TES"

def load_and_prepare_data(params: ModelParameters):
    """
    Loads input data from CSV files and prepares it for the optimization model.
    """
    # --- Load data from CSV files ---
    try:
        load_profiles_df = pd.read_csv(DATA_DIR / "load_profiles.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR / "shiftability_profile.csv", index_col='time_slot')
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
            for j in range(1, 5):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)

def post_process_results(m: pulp.LpProblem, params: ModelParameters, data: dict):
    """
    Extracts results from a solved model into a DataFrame, including step-by-step costs
    and detailed workload processing information.
    """
    solved_vars = {v.name: v.value() for v in m.variables()}
    def get_val(var_name):
        return solved_vars.get(var_name, 0.0)

    # --- Extract Flexible Workload Details ---
    tranche_map = params.tranche_max_delay
    flexible_load_details = []
    for v in m.variables():
        if v.name.startswith("U_JobTranche") and v.value() is not None and v.value() > 1e-6:
            try:
                parts = v.name.replace("U_JobTranche_", "").strip("()").split(",_")
                if len(parts) == 3:
                    t, k, s = map(int, parts)
                    flexible_load_details.append({
                        'processing_slot': s,
                        'original_slot': t,
                        'tranche': k,
                        'cpu_load': v.value(),
                        'shiftability': tranche_map[k] - (s-t) 
                    })
            except (ValueError, IndexError):
                continue
    flex_load_origin_df = pd.DataFrame(flexible_load_details)
    flex_load_origin_df = flex_load_origin_df.sort_values(by=['processing_slot', 'tranche']).reset_index(drop=True)
    flex_load_origin_df.to_csv(DATA_DIR / "flex_load_origin_df.csv")


    flex_filtered = flex_load_origin_df[flex_load_origin_df['shiftability'] > 0]  # Flexible only
    new_flexible_load_per_slot = flex_filtered.groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)  # Sum cpu_load only, for 96 slots
    new_inflexible_load_per_slot = flex_load_origin_df[flex_load_origin_df['shiftability'] == 0].groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)
    new_inflexible_load = data['inflexibleLoadProfile_TEXT'][1:109] + new_inflexible_load_per_slot.values  # Align to slots 1-96
    load_profiles_df = pd.DataFrame({
        'inflexible_load': new_inflexible_load,
        'flexible_load': new_flexible_load_per_slot.values
    }, index=range(1, 109))
    load_profiles_df.index.name = 'time_slot'
    load_profiles_df.to_csv(f'{DATA_DIR}/load_profiles_opt.csv')

    # Now generate the new shiftability profile using remaining shiftability
    def generate_shiftability_profile(flex_df, num_timesteps, num_tranches):
        # Group by processing_slot and shiftability, sum cpu_load
        grouped = flex_df.groupby(['processing_slot', 'shiftability'])['cpu_load'].sum().reset_index()
        
        shiftabilityProfile_data = {}
        for t in range(1, num_timesteps + 1):
            for k in range(1, num_tranches + 1):
                key = (t, k)
                # Sum cpu_load where processing_slot == t and remaining shiftability == k
                value = grouped[(grouped['processing_slot'] == t) & (grouped['shiftability'] == k)]['cpu_load']
                shiftabilityProfile_data[key] = value.sum() if not value.empty else 0.0
        return shiftabilityProfile_data

    data['shiftabilityProfile'] = generate_shiftability_profile(flex_filtered, 108, 12)
    shiftability_df = pd.Series(data['shiftabilityProfile']).unstack()
    shiftability_df.index.name = 'time_slot'
    shiftability_df.columns.name = 'category'
    shiftability_df.to_csv(f'{DATA_DIR}/shiftability_profile_opt.csv')
    # --- Calculate Optimized Cost Profile (per time step) ---
    optimized_cost_per_step = [
        params.dt_hours * (
            get_val(f"P_Grid_IT_{s}") +
            (get_val(f"P_Chiller_HVAC_Watts_{s}") / 1000.0) +
            (get_val(f"P_Chiller_TES_Watts_{s}") / 1000.0) +
            get_val(f"P_Grid_Other_{s}") +
            get_val(f"P_UPS_Charge_{s}")
        ) * (data['electricity_price'][s] / 1000.0)
        for s in params.TEXT_SLOTS
    ]

    # --- Get CPU loads directly from model variables ---
    total_cpu_usage = [get_val(f"TotalCpuUsage_{s}") for s in params.TEXT_SLOTS]
    inflexible_cpu_usage = [data['inflexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]
    flexible_cpu_usage = [(total - inflexible) for total, inflexible in zip(total_cpu_usage, inflexible_cpu_usage)]

    # --- Build results dictionary ---
    total_power_consumption = [
        get_val(f"P_Grid_IT_{s}") +
        (get_val(f"P_Chiller_HVAC_Watts_{s}") / 1000.0) +
        (get_val(f"P_Chiller_TES_Watts_{s}") / 1000.0) +
        get_val(f"P_Grid_Other_{s}") +
        get_val(f"P_UPS_Charge_{s}")
        for s in params.TEXT_SLOTS
    ]
    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'Total_Optimized_Cost': pulp.value(m.objective),
        'Optimized_Cost_per_Step': optimized_cost_per_step,
        'P_Total_kW': total_power_consumption,
        'P_IT_Total_kW': [get_val(f"P_IT_Total_{s}") for s in params.TEXT_SLOTS],
        'P_Grid_IT_kW': [get_val(f"P_Grid_IT_{s}") for s in params.TEXT_SLOTS],
        'P_Chiller_HVAC_kW': [get_val(f"P_Chiller_HVAC_Watts_{s}") / 1000.0 for s in params.TEXT_SLOTS],
        'P_Chiller_TES_kW': [get_val(f"P_Chiller_TES_Watts_{s}") / 1000.0 for s in params.TEXT_SLOTS],
        'P_Grid_Cooling_kW': [(get_val(f"P_Chiller_HVAC_Watts_{s}") + get_val(f"P_Chiller_TES_Watts_{s}")) / 1000.0 for s in params.TEXT_SLOTS],
        'P_Grid_Other_kW': [get_val(f"P_Grid_Other_{s}") for s in params.TEXT_SLOTS],
        'P_UPS_Charge_kW': [get_val(f"P_UPS_Charge_{s}") for s in params.TEXT_SLOTS],
        'P_UPS_Discharge_kW': [get_val(f"P_UPS_Discharge_{s}") for s in params.TEXT_SLOTS],
        'E_UPS_kWh': [get_val(f"E_UPS_{s}") for s in params.TEXT_SLOTS],
        'T_IT_Celsius': [get_val(f"T_IT_{s}") for s in params.TEXT_SLOTS],
        'T_Rack_Celsius': [get_val(f"T_Rack_{s}") for s in params.TEXT_SLOTS],
        'T_ColdAisle_Celsius': [get_val(f"T_ColdAisle_{s}") for s in params.TEXT_SLOTS],
        'T_HotAisle_Celsius': [get_val(f"T_HotAisle_{s}") for s in params.TEXT_SLOTS],
        'E_TES_kWh': [get_val(f"E_TES_{s}") for s in params.TEXT_SLOTS],
        'Q_Cool_Total_Watts': [get_val(f"Q_Cool_Watts_{s}") for s in params.TEXT_SLOTS],
        'Q_Charge_TES_Watts': [get_val(f"Q_Charge_TES_Watts_{s}") for s in params.TEXT_SLOTS],
        'Q_Discharge_TES_Watts': [get_val(f"Q_Discharge_TES_Watts_{s}") for s in params.TEXT_SLOTS],
        'P_IT_Nominal': [data['Pt_IT_nom_TEXT'][s] for s in params.TEXT_SLOTS],
        'Price_GBP_per_MWh': [data['electricity_price'][s] for s in params.TEXT_SLOTS],
        'Inflexible_Load_CPU_Nom': [data['inflexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS],
        'Flexible_Load_CPU_Nom': [data['flexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS],
        'Inflexible_Load_CPU_Opt': inflexible_cpu_usage,
        'Flexible_Load_CPU_Opt': flexible_cpu_usage,
        'Total_CPU_Load': total_cpu_usage,
    }
    df = pd.DataFrame(results)
    df['Optimized_Cost'] = df['Optimized_Cost_per_Step'].cumsum()

    # --- Add nominal cost comparison ---
    try:
        df_nominal = pd.read_csv(DATA_DIR / "nominal_case_results.csv")
        target_dt = params.dt_seconds // 60
        #df_nominal = df_nominal.groupby(df_nominal.index // target_dt).sum(numeric_only=True).reset_index(drop=True)
        df['Nominal_Cost'] = df_nominal['Nominal_Cost'][:len(df)].values
        df['P_Total_kW_Nominal'] = df_nominal['P_Total_kW'][:len(df)].values
    except FileNotFoundError:
        print(f"Warning: '{DATA_DIR / 'nominal_case_results.csv'}' not found. Nominal cost set to 0.")
        df['Nominal_Cost'] = 0
    except Exception as e:
        print(f"An error occurred while processing nominal results: {e}")
        df['Nominal_Cost'] = 0

    return df, flex_load_origin_df

def print_summary(params, results_df: pd.DataFrame):
    optimized_cost = results_df['Optimized_Cost'].iloc[-1] if not results_df.empty else 0

    for t in range(96, 108):
        optimized_cost -= results_df['Nominal_Cost'].iloc[t]
    nominal_cost = results_df['Nominal_Cost'].iloc[:96].sum()

    cost_saving_abs = nominal_cost - optimized_cost if nominal_cost > 0 else 0
    cost_saving_rel = (cost_saving_abs / nominal_cost) * 100 if nominal_cost > 0 else 0

    print("\n" + "="*50)
    print("--- Optimization Results Summary ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print(f"Baseline (Nominal) Cost: {nominal_cost:,.2f} GBP")
    print(f"Absolute Cost Saving: {cost_saving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {cost_saving_rel:.2f} %")
    print("="*50 + "\n")

def create_and_save_charts(df: pd.DataFrame, flex_load_origin_df: pd.DataFrame, data: dict, params: ModelParameters):
    """Generates and saves all charts based on the results DataFrame."""
    print("Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    time_slots_ext = df['Time_Slot_EXT']

    # --- Figure 1: Power Consumption and Energy Price ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(time_slots_ext, df['P_Total_kW'], label='Nominal IT Power', linestyle='--', color='gray')
    ax1.plot(time_slots_ext, df['P_Total_kW_Nominal'], label='Optimized IT Power', color='crimson')
    #ax1.plot(time_slots_ext, df['Optimized_Cost_per_Step'], label='Optimised Cost', linestyle='--', color='gray')
    #ax1.plot(time_slots_ext, df['Nominal_Cost'], label='Nominal Cost', color='crimson')
    ax1.set_ylabel('Cost Incurred (GBP/15 min)')
    #ax1.set_xlabel('Time Slot')
    ax1.set_title('Optimized vs. Nominal DC Operational Cost')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2.plot(time_slots_ext, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Energy Price (GBP/MWh)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig1.tight_layout()
    fig1.savefig(IMAGE_DIR / 'power_consumption_comparison.png')
    print("✅ Power consumption chart saved.")

    # --- Figure 2: TES Performance ---
    fig2, (ax2_tes, ax3_tes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax2_tes.plot(time_slots_ext, df['E_TES_kWh'], label='TES Energy Level', color='mediumblue')
    ax2_tes.set_ylabel('Energy (kWh)')
    ax2_tes.set_title('Thermal Energy Storage (TES) Performance')
    ax2_tes.legend()
    ax2_tes.grid(True)
    ax3_tes.plot(time_slots_ext, df['Q_Charge_TES_Watts'], label='Charge Heat Flow', color='green')
    ax3_tes.plot(time_slots_ext, df['Q_Discharge_TES_Watts'], label='Discharge Heat Flow', color='orange')
    ax3_tes.set_xlabel('Time Slot')
    ax3_tes.set_ylabel('Heat Flow (Watts)')
    ax3_tes.legend()
    ax3_tes.grid(True)
    fig2.tight_layout()
    fig2.savefig(IMAGE_DIR / 'tes_performance.png')
    print("✅ TES performance chart saved.")

    # --- Figure 3: Data Centre Temperatures ---
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(time_slots_ext, df['T_IT_Celsius'], label='IT Equipment Temp')
    ax4.plot(time_slots_ext, df['T_Rack_Celsius'], label='Rack Temp')
    ax4.plot(time_slots_ext, df['T_HotAisle_Celsius'], label='Hot Aisle Temp')
    ax4.plot(time_slots_ext, df['T_ColdAisle_Celsius'], label='Cold Aisle Temp')
    ax4.set_xlabel('Time Slot')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Data Centre Temperatures')
    ax4.legend()
    ax4.grid(True)
    fig3.tight_layout()
    fig3.savefig(IMAGE_DIR / 'dc_temperatures.png')
    print("✅ Data centre temperatures chart saved.")

    # --- Figure 4: Cooling System Power Components ---
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(time_slots_ext, df['P_Grid_Cooling_kW'], label='Total Cooling Power (kW)', color='blue')
    ax5.plot(time_slots_ext, df['P_Chiller_HVAC_kW'], label='Chiller HVAC Power (kW)', color='red')
    ax5.plot(time_slots_ext, df['P_Chiller_TES_kW'], label='Chiller TES Power (kW)', color='green')
    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Cooling System Power Components')
    ax5.legend()
    ax5.grid(True)
    fig4.tight_layout()
    fig4.savefig(IMAGE_DIR / 'cooling_power_components.png')
    print("✅ Cooling system power components chart saved.")

    # --- Figure 5: Thermal Cooling Power (q) ---
    fig5, ax6 = plt.subplots(figsize=(12, 6))
    df['Q_Chiller_Direct_Watts'] = df['Q_Cool_Total_Watts'] - df['Q_Discharge_TES_Watts']
    ax6.stackplot(time_slots_ext, df['Q_Chiller_Direct_Watts'], df['Q_Discharge_TES_Watts'],
                  labels=['Cooling from Chiller (Direct)', 'Cooling from TES'],
                  colors=['green', 'blue'])
    ax6.plot(time_slots_ext, df['Q_Cool_Total_Watts'], label='Total Cooling Demand (Heat from IT)', color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Time Slot')
    ax6.set_ylabel('Thermal Power (Watts)')
    ax6.set_title('Cooling Power (q) by Source')
    ax6.legend(loc='upper left')
    ax6.grid(True)
    fig5.tight_layout()
    fig5.savefig(IMAGE_DIR / 'thermal_cooling_power.png')
    print("✅ Thermal cooling power chart saved.")

    # --- Figure 6: Cumulative Cost Comparison ---
    fig6, ax7 = plt.subplots(figsize=(12, 7))
    ax7.plot(time_slots_ext, df['Optimized_Cost'], label='Optimized Cost', color='crimson', linewidth=2)
    ax7.plot(time_slots_ext, df['Nominal_Cost'].cumsum(), label='Nominal Cost', color='gray', linestyle='--', linewidth=2)
    ax7.set_xlabel('Time Slot')
    ax7.set_ylabel('Cumulative Cost (£)')
    ax7.set_title('Cumulative Energy Cost: Optimized vs. Nominal')
    ax7.legend()
    ax7.grid(True)
    fig6.tight_layout()
    fig6.savefig(IMAGE_DIR / 'cumulative_cost_comparison.png')
    print("✅ Cumulative cost comparison chart saved.")

    # --- Figure 7: Stacked Bar Chart of IT Workload ---
    if not flex_load_origin_df.empty:
        flex_load_origin_df['lag'] = flex_load_origin_df['processing_slot'] - flex_load_origin_df['original_slot']
        flex_pivot_by_lag = flex_load_origin_df.pivot_table(
            index='processing_slot',
            columns='lag',
            values='cpu_load',
            aggfunc='sum'
        ).fillna(0)

        plot_df = pd.DataFrame({
            'Inflexible': df['Inflexible_Load_CPU_Nom'].values
        }, index=df['Time_Slot_EXT'])
        plot_df = plot_df.join(flex_pivot_by_lag).fillna(0)
        
        rename_dict = {lag: f'Flexible (Lag {int(lag)})' for lag in flex_pivot_by_lag.columns}
        plot_df.rename(columns=rename_dict, inplace=True)
        
        fig7, ax8 = plt.subplots(figsize=(18, 9))
        flexible_cols = [col for col in plot_df.columns if 'Flexible' in col]
        colors = ['black'] + list(plt.cm.viridis(np.linspace(0, 1, len(flexible_cols))))

        plot_df.plot(kind='bar', stacked=True, ax=ax8, width=0.8, color=colors)
        
        # Overlay nominal total load for comparison
        nominal_total_load = data['inflexibleLoadProfile_TEXT'][1:] + data['flexibleLoadProfile_TEXT'][1:]
        ax8.plot(range(len(nominal_total_load)), nominal_total_load, label='Nominal Total Workload', linestyle='--', color='gray')

        ax8.set_title('Optimized IT Workload Composition by Origin', fontsize=16)
        ax8.set_xlabel('Processing Time Slot')
        ax8.set_ylabel('CPU Load Units')
        ax8.legend(title='Load Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax8.grid(axis='y', linestyle='--', alpha=0.7)

        tick_frequency = max(1, len(time_slots_ext) // 24)
        ax8.set_xticks(ax8.get_xticks()[::tick_frequency])
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha="right")
        
        fig7.tight_layout(rect=[0, 0, 0.85, 1])
        fig7.savefig(IMAGE_DIR / 'it_load_stacked_bar.png')
        print("✅ IT workload stacked bar chart saved.")

    plt.show()
    print("\nAll charts have been generated and saved.")

def run_single_optimization(params: ModelParameters, input_data: dict, msg=False):
    """
    Runs a single optimization instance with a given set of parameters.

    Args:
        params (ModelParameters): The configuration parameters for the model.
        input_data (dict): The prepared input data (loads, prices, etc.).
        msg (bool): Whether to show solver output.

    Returns:
        A tuple containing:
        - The total optimized cost (float).
        - The full results DataFrame.
        - The flexible load origin DataFrame.
    """
    print(f"Building and solving model...")
    model = build_model(params, input_data)

    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=msg, gapRel=0.01))

    if pulp.LpStatus[model.status] == 'Optimal':
        print("Solver found an optimal solution. Post-processing...")
        results_df, flex_load_origin_df = post_process_results(model, params, input_data)
        total_cost = results_df['Total_Optimized_Cost'].iloc[0]
        return total_cost, results_df, flex_load_origin_df
    else:
        print(f"Solver did not find an optimal solution. Status: {pulp.LpStatus[model.status]}")
        return None, None, None

def run_full_optimisation():
    """
    Sets up and runs the full, baseline optimization, including generating charts.
    """
    print("1. Setting up model parameters...")
    params = ModelParameters()

    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)

    total_cost, results_df, flex_load_origin_df = run_single_optimization(params, input_data, msg=True)

    if total_cost is not None:
        print_summary(params, results_df)
        create_and_save_charts(results_df, flex_load_origin_df, input_data, params)

        output_path = DATA_DIR / "optimised_baseline.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nResults successfully exported to '{output_path}'")

if __name__ == '__main__':
    run_full_optimisation()