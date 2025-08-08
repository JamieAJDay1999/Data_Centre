# main_flexibility_calculator.py

import pathlib
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt

# --- Import user-provided dependency modules ---
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles

# --- Constants and Configuration ---------------------------------------------
_DBGDIR = pathlib.Path("lp_debug")
_DBGDIR.mkdir(exist_ok=True)
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


# --- Optimisation Model Building -----------------------------------------------
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

# --- Data Loading and Processing ---------------------------------------------
def load_and_prepare_data(params: ModelParameters):
    """Loads external data and preprocesses it for the model."""
    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    inflexible, base_flex, base_flex_t, shiftability, _ = get_load_and_price_profiles(params.TEXT_SLOTS, params.T_SLOTS)

    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'flexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(base_flex, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0),
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

# --- Post-Processing and Baseline Generation ---------------------------------
def post_process_results(m: pulp.LpProblem, params: ModelParameters, data: dict):
    """
    Extracts key results from a solved model into a DataFrame for baseline generation.
    """
    solved_vars = {v.name: v.value() for v in m.variables()}
    def get_val(var_name):
        return solved_vars.get(var_name, 0.0)

    # Get CPU loads directly from model variables
    total_cpu_usage = [get_val(f"TotalCpuUsage_{s}") for s in params.TEXT_SLOTS]
    inflexible_cpu_usage = [data['inflexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]
    flexible_cpu_usage = [(total - inflexible) for total, inflexible in zip(total_cpu_usage, inflexible_cpu_usage)]

    # Build results dictionary with only the columns needed for flexibility calculation
    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'P_IT_Total_kW': [get_val(f"P_IT_Total_{s}") for s in params.TEXT_SLOTS],
        'P_Grid_Cooling_kW': [(get_val(f"P_Chiller_HVAC_Watts_{s}") + get_val(f"P_Chiller_TES_Watts_{s}")) / 1000.0 for s in params.TEXT_SLOTS],
        'P_Grid_Other_kW': [get_val(f"P_Grid_Other_{s}") for s in params.TEXT_SLOTS],
        'E_TES_kWh': [get_val(f"E_TES_{s}") for s in params.TEXT_SLOTS],
        'Inflexible_Load_CPU': inflexible_cpu_usage,
        'Flexible_Load_CPU': flexible_cpu_usage,
        'Total_CPU_Load': total_cpu_usage,
    }
    df = pd.DataFrame(results)
    
    df['P_IT_Total_kW'][len(params.T_SLOTS):] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * np.array(total_cpu_usage[len(params.T_SLOTS):])

    return df

# --- Optimisation Runner -----------------------------------------------------
def run_cost_optimisation():
    """Orchestrates the optimization process to generate the baseline."""
    print("1. Setting up model parameters...")
    params = ModelParameters()
    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)
    print("3. Building optimization model...")
    model = build_model(params, input_data)
    print("4. Starting solver to find optimised baseline...")
    model.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0.01))

    if pulp.LpStatus[model.status] == 'Optimal':
        print("5. Solver found an optimal solution. Extracting baseline results...")
        results_df = post_process_results(model, params, input_data)
        output_path = "optimised_baseline.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\n✅ Optimised baseline results saved to '{output_path}'")
        return results_df, params
    else:
        print(f"*** Solver did not find an optimal solution. Status: {pulp.LpStatus[model.status]} ***")
        return None, None

# --- Flexibility Calculation Functions ---------------------------------------

def calculate_power_flexibility(baseline_df: pd.DataFrame, params: ModelParameters):
    """
    Calculates the upward and downward power flexibility for each timestep.

    Args:
        baseline_df: DataFrame containing the optimised baseline results.
        params: An instance of the ModelParameters class.

    Returns:
        A DataFrame with added columns for flexibility.
    """
    upward_flex = []
    downward_flex = []

    for _, row in baseline_df.iterrows():
        # --- 1. Calculate Upward Flexibility (Power Decrease) ---
        
        # IT Power Reduction: Assume all flexible load is shifted away.
        p_it_min = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * row['Inflexible_Load_CPU']
        
        # Cooling Power Reduction: Cool the minimum IT load, using TES first.
        q_required_watts = p_it_min * 1000
        
        # Available cooling from TES (Watts), limited by max discharge rate and available energy.
        tes_energy_kwh = row['E_TES_kWh']
        max_q_from_tes_by_energy = (tes_energy_kwh - params.E_TES_min_kWh) / params.dt_hours * 1000 * params.TES_discharge_efficiency
        
        q_from_tes = min(q_required_watts, params.TES_w_discharge_max, max_q_from_tes_by_energy)
        q_from_chiller = max(0, q_required_watts - q_from_tes)
        
        # Power for remaining cooling from chiller (kW)
        p_cooling_min = (q_from_chiller / params.COP_HVAC) / 1000.0

        # Other power reduction
        p_other_min = p_it_min * params.nominal_overhead_factor
        
        # Total minimum power consumption
        p_total_min = p_it_min + p_cooling_min + p_other_min
        
        # Baseline total power for comparison
        p_baseline = row['P_IT_Total_kW'] + row['P_Grid_Cooling_kW'] + row['P_Grid_Other_kW']
        
        # Final Upward Flexibility (kW)
        up_flex = max(0, p_baseline - p_total_min)
        upward_flex.append(up_flex)

        # --- 2. Calculate Downward Flexibility (Power Increase) ---
        
        # IT Power Increase: Set CPU to 100%
        p_it_max = params.max_power_kw
        
        # Cooling Power Increase: Cool max IT load AND charge TES.
        q_cool_demand_watts = p_it_max * 1000
        p_chiller_for_hvac_kw = (q_cool_demand_watts / params.COP_HVAC) / 1000.0
        
        # Remaining chiller capacity for charging TES
        p_chiller_max_kw = params.P_chiller_max / 1000.0
        p_chiller_for_tes_kw = max(0, p_chiller_max_kw - p_chiller_for_hvac_kw)
        
        # Also limited by TES max charge rate
        max_p_chiller_for_tes_by_rate = (params.TES_w_charge_max / params.COP_HVAC) / 1000.0
        
        # And by available capacity in TES
        tes_headroom_kwh = params.TES_capacity_kWh - tes_energy_kwh
        max_p_chiller_for_tes_by_cap = (tes_headroom_kwh / params.dt_hours) / (params.COP_HVAC * params.TES_charge_efficiency)

        p_chiller_for_tes_kw = min(p_chiller_for_tes_kw, max_p_chiller_for_tes_by_rate, max_p_chiller_for_tes_by_cap)
        
        p_cooling_max = p_chiller_for_hvac_kw + p_chiller_for_tes_kw
        
        # Other power increase
        p_other_max = p_it_max * params.nominal_overhead_factor
        
        # Total maximum power
        p_total_max = p_it_max + p_cooling_max + p_other_max
        
        # Final Downward Flexibility (kW)
        down_flex = max(0, p_total_max - p_baseline)
        downward_flex.append(down_flex)

    # Add new columns to the DataFrame
    baseline_df['Upward_Flexibility_kW'] = upward_flex
    baseline_df['Downward_Flexibility_kW'] = downward_flex
    
    return baseline_df


def plot_flexibility(df: pd.DataFrame):
    """Generates and saves a chart of the calculated power flexibility."""
    print("7. Generating and saving flexibility chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    time_slots = df['Time_Slot_EXT']
    baseline_power = df['P_IT_Total_kW'] + df['P_Grid_Cooling_kW'] + df['P_Grid_Other_kW']

    # Plot baseline
    ax.plot(time_slots, baseline_power, label='Optimised Baseline Power', color='black', linewidth=2)
    
    # Create the flexibility band
    ax.fill_between(
        time_slots,
        baseline_power - df['Upward_Flexibility_kW'],
        baseline_power + df['Downward_Flexibility_kW'],
        color='skyblue',
        alpha=0.5,
        label='Power Flexibility Range'
    )
    
    ax.plot(time_slots, baseline_power - df['Upward_Flexibility_kW'], linestyle='--', color='green', label='Min Power (Max Upward Flex)')
    ax.plot(time_slots, baseline_power + df['Downward_Flexibility_kW'], linestyle='--', color='red', label='Max Power (Max Downward Flex)')

    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Total Power (kW)', fontsize=12)
    ax.set_title('Data Centre Power Flexibility', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('power_flexibility.png')
    print("✅ Power flexibility chart saved as 'power_flexibility.png'")
    plt.show()

# --- Main Orchestrator -------------------------------------------------------

if __name__ == '__main__':
    # Step 1: Run the original optimisation to get the baseline
    baseline_df, params = run_cost_optimisation()

    if baseline_df is not None:
        # Step 2: Calculate the power flexibility based on the baseline
        print("\n6. Calculating power flexibility...")
        flexibility_df = calculate_power_flexibility(baseline_df, params)
        
        # Step 3: Save the results to a new CSV
        output_path = "results_with_flexibility.csv"
        flexibility_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"✅ Results with flexibility have been exported to '{output_path}'")
        
        # Step 4: Plot the results
        plot_flexibility(flexibility_df)
        
        # Display the first few rows with the new flexibility data
        print("\n--- Results Preview with Flexibility ---")
        print(flexibility_df[['Time_Slot_EXT', 'P_IT_Total_kW', 'Upward_Flexibility_kW', 'Downward_Flexibility_kW']].head().to_string())