import datetime
import json
import pathlib
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles

# --- Constants and Configuration ---------------------------------------------
_DBGDIR = pathlib.Path("lp_debug")
_DBGDIR.mkdir(exist_ok=True)

# Model configuration switches
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
        self.tranche_max_delay = {1: 30, 2: 60, 3: 120, 4: 180}
        #self.tranche_max_delay = {1: 2, 2: 4, 3: 8, 4: 12}
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
    t_it = pulp.LpVariable.dicts("T_IT", TEXT_SLOTS, lowBound=14, upBound=75)
    t_rack = pulp.LpVariable.dicts("T_Rack", TEXT_SLOTS, lowBound=14, upBound=70)
    t_cold_aisle = pulp.LpVariable.dicts("T_ColdAisle", TEXT_SLOTS, lowBound=14, upBound=params.T_cAisle_upper_limit_Celsius)
    t_hot_aisle = pulp.LpVariable.dicts("T_HotAisle", TEXT_SLOTS, lowBound=14, upBound=80)
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

# --- Constraint Helper Functions (unchanged) ---
def add_it_and_job_constraints(m, params, data, total_cpu, p_it_total, ut_ks):
    """Adds all IT power and job scheduling constraints to the model."""
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
            m += p_it_act_ext[s] >= (params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * total_cpu[s]) - nominal_power_in_ext, f"IT_Power_Ext_Min_{s}"
        m += p_it_total[s] == p_it_act[s] + p_it_act_ext[s], f"Total_IT_Power_{s}"

def add_ups_constraints(m, params, p_ups_ch, p_ups_disch, e_ups, z_ch, z_disch):
    """Adds all UPS/battery constraints to the model."""
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
    """Adds the main power grid balance equations."""
    for s in params.TEXT_SLOTS:
        m += p_it_total[s] == p_grid_it[s] + p_ups_disch[s], f"IT_PowerBalance_{s}"
        m += p_grid_od[s] == p_it_total[s] * params.nominal_overhead_factor, f"Overhead_PowerBalance_{s}"

def add_cooling_constraints(m, params, p_it_total, t_it, t_rack, t_cold, t_hot, e_tes, p_hvac, p_tes, q_cool, q_ch_tes, q_dis_tes, t_in):
    """Adds all thermodynamic and cooling system constraints."""
    m += t_it[1] >= params.T_IT_initial_Celsius
    m += t_rack[1] >= params.T_Rack_initial_Celsius
    m += t_cold[1] >= params.T_cAisle_initial
    m += t_hot[1] >= params.T_hAisle_initial
    m += e_tes[1] >= params.TES_initial_charge_kWh
    mcp = params.m_dot_air * params.c_p_air
    for t in params.TEXT_SLOTS[1:]:
        m += q_cool[t] == (p_hvac[t] * params.COP_HVAC) + q_dis_tes[t], f"CoolingSourceBalance_{t}"
        m += q_ch_tes[t] == p_tes[t] * params.COP_HVAC, f"ChillerTESPower_{t}"
        m += t_in[t] == t_hot[t] - q_cool[t] / mcp, f"AisleTempIn_{t}"
        m += q_cool[t] <= (t_hot[t] - params.T_cAisle_lower_limit_Celsius) * mcp, f"MaxCoolingDrop_{t}"
        it_heat_watts = p_it_total[t] * 1000.0
        m += t_it[t] == t_it[t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (t_it[t] - t_rack[t])) / params.C_IT), f"TempUpdate_IT_{t}"
        m += t_rack[t] == t_rack[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_cold[t]-t_rack[t]) + params.G_conv*(t_it[t]-t_rack[t])) / params.C_Rack), f"TempUpdate_Rack_{t}"
        m += t_cold[t] == t_cold[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_in[t]-t_cold[t]) - params.G_cold*(t_cold[t]-params.T_out_Celsius)) / params.C_cAisle), f"TempUpdate_ColdAisle_{t}"
        m += t_hot[t] == t_hot[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(t_rack[t]-t_hot[t])) / params.C_hAisle), f"TempUpdate_HotAisle_{t}"
        dE_tes = (q_ch_tes[t]*params.TES_charge_efficiency - q_dis_tes[t]/params.TES_discharge_efficiency) * params.dt_hours / 1000.0
        m += e_tes[t] == e_tes[t-1] + dE_tes, f"EnergyBalance_TES_{t}"
        m += q_dis_tes[t] - q_dis_tes[t] <= params.TES_p_dis_ramp, f"Ramp_TES_Discharge_{t}"
        m += q_ch_tes[t] - q_ch_tes[t] <= params.TES_p_ch_ramp, f"Ramp_TES_Charge_{t}"
    if CYCLE_TES_ENERGY:
        m += e_tes[max(params.TEXT_SLOTS)] == e_tes[1], "Cycle_E_TES"

# --- Data Loading and Processing (unchanged) ---
def load_and_prepare_data(params: ModelParameters):
    RESAMPLE_FACTOR = int(900/ params.dt_seconds)  # Resampling factor based on dt_seconds
    print(f"Resampling factor: {RESAMPLE_FACTOR} (dt_seconds: {params.dt_seconds})")
    """Loads external data and preprocesses it for the model."""
    inflexible, base_flex, base_flex_t, shiftability, _ = get_load_and_price_profiles(params.TEXT_SLOTS, params.T_SLOTS)
    
    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'shiftabilityProfile': resample_shiftability_profile(shiftability, RESAMPLE_FACTOR)
    }
    baseFlexibleLoadProfile_T = np.insert(np.array(np.repeat(base_flex_t, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0)
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours
    baseFlexibleLoadProfile_TEXT = np.insert(np.array(np.repeat(base_flex, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0)
    data['Pt_IT_nom_TEXT'] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (data['inflexibleLoadProfile_TEXT'] + baseFlexibleLoadProfile_TEXT)
    data['electricity_price'] = generate_tariff(params.num_steps_extended, params.dt_seconds)
    return data

def resample_shiftability_profile(shiftability_profile, repeats):
    """Resamples the shiftability profile."""
    extended_data = {}
    counter = 1
    for i in range(1, 97):
        for _ in range(repeats):
            for j in range(1, 5):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    """Generates a flattened electricity price tariff."""
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)


# --- Post-Processing, Charting, and Output ------------------------------------
def post_process_results(m: pulp.LpProblem, params: ModelParameters, data: dict):
    """
    Extracts results from a solved model into a DataFrame using an efficient one-pass method.
    """
    # EFFICIENT METHOD: Extract all variable values in one pass
    solved_vars = {v.name: v.value() for v in m.variables()}

    # Helper for fast lookups in the pre-fetched dictionary
    def get_val(var_name):
        return solved_vars.get(var_name, 0) # .get is safer than []

    
    nominal_pue = 1 + 0.4 + params.nominal_overhead_factor
    total_nominal_power_profile = data["Pt_IT_nom_TEXT"] * nominal_pue
    nominal_cost = sum(
        params.dt_hours * total_nominal_power_profile[t] * (data['electricity_price'][t] / 1000.0)
        for t in params.TEXT_SLOTS
    )

    # Build results dictionary using fast lookups
    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'Optimized_Cost': pulp.value(m.objective),
        'Nominal_Cost': nominal_cost,
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
        'Price_GBP_per_MWh': [data['electricity_price'][s] for s in params.TEXT_SLOTS]
    }
    
    return pd.DataFrame(results)

def print_summary(results_df: pd.DataFrame):
    """Prints a formatted summary of the optimization results."""
    optimized_cost = results_df['Optimized_Cost'].iloc[0]
    nominal_cost = results_df['Nominal_Cost'].iloc[0]
    cost_saving_abs = nominal_cost - optimized_cost
    cost_saving_rel = (cost_saving_abs / nominal_cost) * 100 if nominal_cost else 0
    
    print("\n" + "="*50)
    print("--- Optimization Results ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print(f"Baseline (Nominal) Cost: {nominal_cost:,.2f} GBP")
    print("\n--- Savings ---")
    print(f"Absolute Cost Saving: {cost_saving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {cost_saving_rel:.2f} %")
    print("="*50 + "\n")

def create_and_save_charts(params, df: pd.DataFrame):
    """Generates and saves charts based on the results DataFrame."""
    print("6. Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')

    len_time_slots = len(params.T_SLOTS)
    time_slots_ext  = df['Time_Slot_EXT']
    time_slots_ext_only = df['Time_Slot_EXT'][len_time_slots:]
    extended_it_total_power =  df['P_IT_Nominal'][len_time_slots:].values + df['P_IT_Total_kW'][len_time_slots:].values

    # --- Figure 1: Power Consumption ---
    #df['Optimized_Total_Power_kW'] = df['P_Grid_IT_kW'] + df['P_Grid_Cooling_kW'] + df['P_Grid_Other_kW'] + df['P_UPS_Charge_kW']
    
    # --- Figure 1: Power Consumption and Energy Price (on separate subplots) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Power Consumption
    ax1.plot(time_slots_ext, df['P_IT_Nominal'], label='Nominal IT Power', linestyle='--', color='gray')
    ax1.plot(time_slots_ext, df['P_IT_Total_kW'], label='Optimized IT Power', color='crimson')
    ax1.plot(time_slots_ext_only, extended_it_total_power, label='Total IT Power in Extended time', color='green')
    ax1.set_ylabel('Total Power Consumption (kW)')
    ax1.set_title('Optimized vs. Nominal Power Consumption')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Energy Price
    ax2.plot(time_slots_ext, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_xlabel('Time Slot (minute)')
    ax2.set_ylabel('Energy Price (GBP/MWh)', color='royalblue')
    ax2.set_title('Electricity Price Profile')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    fig1.tight_layout()
    fig1.savefig('power_consumption_comparison.png')
    

    # --- Figure 2: TES Performance ---
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax2.plot(time_slots_ext, df['E_TES_kWh'], label='TES Energy Level', color='mediumblue')
    ax2.set_ylabel('Energy (kWh)')
    ax2.set_title('Thermal Energy Storage (TES) Performance')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(time_slots_ext, df['Q_Charge_TES_Watts'], label='Charge Heat Flow', color='green')
    ax3.plot(time_slots_ext, df['Q_Discharge_TES_Watts'], label='Discharge Heat Flow', color='orange')
    ax3.set_xlabel('Time Slot (minute)')
    ax3.set_ylabel('Heat Flow (Watts)')
    ax3.legend()
    ax3.grid(True)
    
    fig2.tight_layout()
    fig2.savefig('tes_performance.png')
    
    # --- Figure 3: Data Centre Temperatures ---
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(time_slots_ext, df['T_IT_Celsius'], label='IT Equipment Temp')
    ax4.plot(time_slots_ext, df['T_Rack_Celsius'], label='Rack Temp')
    ax4.plot(time_slots_ext, df['T_HotAisle_Celsius'], label='Hot Aisle Temp')
    ax4.plot(time_slots_ext, df['T_ColdAisle_Celsius'], label='Cold Aisle Temp')
    ax4.set_xlabel('Time Slot (minute)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Data Centre Temperatures')
    ax4.legend()
    ax4.grid(True)
    fig3.tight_layout()
    fig3.savefig('dc_temperatures.png')

    # --- Figure 4: Cooling System Power Components ---
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(time_slots_ext, df['P_Grid_Cooling_kW'], label='Total Cooling Power (kW)', color='blue')
    ax5.plot(time_slots_ext, df['P_Chiller_HVAC_kW'], label='Chiller HVAC Power (kW)', color='red')
    ax5.plot(time_slots_ext, df['P_Chiller_TES_kW'], label='Chiller TES Power (kW)', color='green')
    ax5.set_xlabel('Time Slot (minute)')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Cooling System Power Components')
    ax5.legend()
    ax5.grid(True)
    fig4.tight_layout()
    fig4.savefig('cooling_power_components.png')
    print("Cooling system power components chart saved as 'cooling_power_components.png'.")

    # --- NEW: Figure 5: Thermal Cooling Power (q) ---
    fig5, ax6 = plt.subplots(figsize=(12, 6))
    # Calculate direct chiller thermal power (q)
    df['Q_Chiller_Direct_Watts'] = df['Q_Cool_Total_Watts'] - df['Q_Discharge_TES_Watts']
    
    ax6.plot(time_slots_ext, df['Q_Chiller_Direct_Watts'], color='green', linewidth=2)
    ax6.plot(time_slots_ext, df['Q_Discharge_TES_Watts'], color='blue', linewidth=2)      
    ax6.plot(time_slots_ext, df['Q_Cool_Total_Watts'], label='Total Cooling Power (q)', color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Time Slot (minute)')
    ax6.set_ylabel('Thermal Power (Watts)')
    ax6.set_title('Cooling Power (q) by Source')
    ax6.legend(loc='upper left')
    ax6.grid(True)
    fig5.tight_layout()
    fig5.savefig('thermal_cooling_power.png')
    print("Thermal cooling power chart saved as 'thermal_cooling_power.png'.")

    plt.show()
    print("Charts saved as .png files.")

# --- Main Orchestrator -------------------------------------------------------
def run_full_optimisation():
    """Main function to orchestrate the optimization process."""
    print("1. Setting up model parameters...")
    params = ModelParameters()
    
    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)
    
    print(f"3. Building optimization model for {params.num_steps_extended} time steps...")
    model = build_model(params, input_data)
    
    print("4. Starting solver...")
    model.solve(pulp.PULP_CBC_CMD(msg=True, gapRel=0.01))

    if pulp.LpStatus[model.status] == 'Optimal':
        print("5. Solver found an optimal solution. Post-processing...")
        results_df = post_process_results(model, params, input_data)
        print_summary(results_df)
        
        create_and_save_charts(params, results_df)
        
        output_path = "AllResults_Refactored.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Results have been successfully exported to '{output_path}'")
        return results_df
    else:
        print(f"*** Solver did not find an optimal solution. Status: {pulp.LpStatus[model.status]} ***")
        return None

if __name__ == '__main__':
    run_full_optimisation()