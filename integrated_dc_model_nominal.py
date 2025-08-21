import datetime
import json
import pathlib
import pandas as pd
import numpy as np
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from parameters_optimisation import setup_simulation_parameters

# --- Path Configuration ------------------------------------------------------
DATA_DIR = pathlib.Path("static/data")
IMAGE_DIR = pathlib.Path("static/images2")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


# --- Model Configuration -----------------------------------------------------
CYCLE_TES_ENERGY = True

# --- Parameter Management Class (Logic Preserved) ----------------------------
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

        # Time slots (1-based indexing for readability)
        self.T_SLOTS = range(1, 1 + int(self.simulation_minutes * 60 / self.dt_seconds))
        self.TEXT_SLOTS = range(1, 1 + self.num_steps_extended)
        self.K_TRANCHES = range(1, 5)

        # --- IT Equipment ---
        self.idle_power_kw = 166.7
        self.max_power_kw = 1000.0
        self.max_cpu_usage = 1.0
        self.tranche_max_delay = {1: 2, 2: 4, 3: 8, 4: 12}
        self.nominal_overhead_factor = 0.1 # For other DC loads (lighting, etc.)

        # --- UPS / Battery Storage (Effectively disabled for nominal run) ---
        self.eta_ch = 0.82
        self.eta_disch = 0.92
        self.e_nom_kwh = 0
        self.soc_min = 0.5
        self.soc_max = 0
        self.e_start_kwh = 0
        self.p_max_ch_kw = 0
        self.p_max_disch_kw = 0
        self.p_min_ch_kw = 0
        self.p_min_disch_kw = 0
        self.e_min_kwh = 0
        self.e_max_kwh = 0

        # --- Cooling System (from external file, TES disabled) ---
        cooling_params = setup_simulation_parameters("cool_down")
        self.__dict__.update(cooling_params) # Merges the cooling params into this class
        self.TES_kwh_cap = 0
        self.TES_w_discharge_max = 0
        self.TES_discharge_efficiency = 0.9
        self.TES_w_charge_max = 0
        self.TES_charge_efficiency = 0.9
        self.E_TES_min_kWh = 0.0
        self.TES_initial_charge_kWh = 0.5 * self.TES_kwh_cap
        self.TES_p_dis_ramp = 0
        self.TES_p_ch_ramp = 0
        self.TES_capacity_kWh = self.TES_kwh_cap


def build_model(params: ModelParameters, data: dict):
    """
    Builds the Pyomo model for the nominal case.
    """
    # MODIFIED: Use Pyomo's ConcreteModel
    m = pyo.ConcreteModel(name="DC_Cost_Calculation_Nominal")

    # MODIFIED: Define index sets for Pyomo
    m.TEXT_SLOTS = pyo.Set(initialize=params.TEXT_SLOTS)
    m.T_SLOTS = pyo.Set(initialize=params.T_SLOTS)
    m.K_TRANCHES = pyo.Set(initialize=params.K_TRANCHES)

    # IT & Power Grid Variables
    m.total_cpu = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.max_cpu_usage), initialize=0)
    # Power variables in kW
    m.p_grid_it_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_grid_od_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_it_total_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)

    # UPS Variables
    m.p_ups_ch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_ups_disch_kw = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.e_ups_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.e_min_kwh, params.e_max_kwh), initialize=params.e_start_kwh)
    m.z_ch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary, initialize=0)
    m.z_disch = pyo.Var(m.TEXT_SLOTS, within=pyo.Binary, initialize=0)

    # Job Scheduling Variables (set to be inflexible for nominal case)
    ut_ks_idx = [(t, k, s) for t in m.T_SLOTS for k in m.K_TRANCHES for s in m.TEXT_SLOTS if s == t]
    m.ut_ks_idx = pyo.Set(initialize=ut_ks_idx)
    m.ut_ks = pyo.Var(m.ut_ks_idx, within=pyo.NonNegativeReals, initialize=0)

    # Cooling System Variables
    m.t_it = pyo.Var(m.TEXT_SLOTS, bounds=(14, 60), initialize=25)
    m.t_rack = pyo.Var(m.TEXT_SLOTS, bounds=(14, 40), initialize=25)
    m.t_cold_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(18, params.T_cAisle_upper_limit_Celsius), initialize=20)
    m.t_hot_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(14, 40), initialize=30)
    m.e_tes_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.E_TES_min_kWh, params.TES_capacity_kWh), initialize=params.TES_initial_charge_kWh)
    # Electrical power for cooling in Watts
    m.p_chiller_hvac_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_chiller_tes_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    # Thermal power in Watts
    m.q_cool_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.q_ch_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_charge_max), initialize=0)
    m.q_dis_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_discharge_max), initialize=0)
    m.t_in = pyo.Var(m.TEXT_SLOTS, bounds=(14, 30), initialize=20)


    # --- Add Constraints ---
    add_it_and_job_constraints(m, params, data)
    add_ups_constraints(m, params)
    add_power_balance_constraints(m, params)
    add_cooling_constraints(m, params)

    # --- Define Objective Function ---
    def objective_rule(mod):
        # Objective combines all grid-drawn power in kW
        cost = sum(
            params.dt_hours * (
                mod.p_grid_it_kw[s] +
                (mod.p_chiller_hvac_w[s] / 1000.0) + # W to kW
                (mod.p_chiller_tes_w[s] / 1000.0) +  # W to kW
                mod.p_grid_od_kw[s] +
                mod.p_ups_ch_kw[s]
            ) * (data['electricity_price'][s] / 1000.0) # Price is per MWh
            for s in mod.TEXT_SLOTS
        )
        return cost
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m

def add_it_and_job_constraints(m, params, data):
  
    m.JobCompletion = pyo.ConstraintList()
    for t in m.T_SLOTS:
        for k in m.K_TRANCHES:
            # For nominal case, jobs are not shifted, so s must equal t.
            if (t, k, t) in m.ut_ks_idx:
                expr = m.ut_ks[(t, k, t)] * params.dt_hours
                m.JobCompletion.add(expr == data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0))

    m.CPUandPower = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        flexible_usage = sum(m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s)
        m.CPUandPower.add(m.total_cpu[s] == data['inflexibleLoadProfile_TEXT'][s] + flexible_usage)
        
        # p_it_total_kw is in kW
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.total_cpu[s] ** 1.32
        m.CPUandPower.add(m.p_it_total_kw[s] == power_expr)

def add_ups_constraints(m, params):
    m.UPS_Constraints = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # e_ups_kwh is in kWh, p_ups... variables are in kW
        prev_energy = params.e_start_kwh if s == m.TEXT_SLOTS.first() else m.e_ups_kwh[s-1]
        charge = params.eta_ch * m.p_ups_ch_kw[s] * params.dt_hours
        discharge = (m.p_ups_disch_kw[s] / params.eta_disch) * params.dt_hours
        m.UPS_Constraints.add(m.e_ups_kwh[s] == prev_energy + charge - discharge)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] <= m.z_ch[s] * params.p_max_ch_kw)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] >= m.z_ch[s] * params.p_min_ch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] <= m.z_disch[s] * params.p_max_disch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] >= m.z_disch[s] * params.p_min_disch_kw)
        m.UPS_Constraints.add(m.z_ch[s] + m.z_disch[s] <= 1)
    m.UPS_Constraints.add(m.e_ups_kwh[m.TEXT_SLOTS.last()] == params.e_start_kwh)

def add_power_balance_constraints(m, params):
    m.PowerBalance = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # All power variables in this balance are in kW
        m.PowerBalance.add(m.p_it_total_kw[s] == m.p_grid_it_kw[s] + m.p_ups_disch_kw[s])
        m.PowerBalance.add(m.p_grid_od_kw[s] == m.p_it_total_kw[s] * params.nominal_overhead_factor)

def add_cooling_constraints(m, params):
    m.CoolingConstraints = pyo.ConstraintList()
    m.CoolingConstraints.add(m.t_it[1] >= params.T_IT_initial_Celsius)
    m.CoolingConstraints.add(m.t_rack[1] >= params.T_Rack_initial_Celsius)
    m.CoolingConstraints.add(m.t_cold_aisle[1] >= params.T_cAisle_initial)
    m.CoolingConstraints.add(m.t_hot_aisle[1] >= params.T_hAisle_initial)
    m.CoolingConstraints.add(m.e_tes_kwh[1] >= params.TES_initial_charge_kWh)
    mcp = params.m_dot_air * params.c_p_air
    
    # This constraint can cause issues if the list is empty
    if len(m.TEXT_SLOTS) > 1:
        avg_hvac_w = sum(m.p_chiller_hvac_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) - 1)
        avg_tes_w = sum(m.p_chiller_tes_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) - 1)
        m.CoolingConstraints.add(m.p_chiller_hvac_w[1] == avg_hvac_w)
        m.CoolingConstraints.add(m.p_chiller_tes_w[1] == avg_tes_w)
    
    for t in m.TEXT_SLOTS:
        if t > 1:
            # Thermal power (q) is in Watts, Electrical power (p_chiller) is in Watts
            m.CoolingConstraints.add(m.q_cool_w[t] == (m.p_chiller_hvac_w[t] * params.COP_HVAC) + m.q_dis_tes_w[t])
            m.CoolingConstraints.add(m.q_ch_tes_w[t] == m.p_chiller_tes_w[t] * params.COP_HVAC)
            m.CoolingConstraints.add(m.t_in[t] == m.t_hot_aisle[t] - m.q_cool_w[t] / mcp)
            m.CoolingConstraints.add(m.q_cool_w[t] <= (m.t_hot_aisle[t] - params.T_cAisle_lower_limit_Celsius) * mcp)
            
            # Convert IT power from kW to Watts for thermal calculation
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

    if CYCLE_TES_ENERGY:
        m.CoolingConstraints.add(m.e_tes_kwh[m.TEXT_SLOTS.last()] == m.e_tes_kwh[1])

def load_and_prepare_data(params: ModelParameters):
    """
    Loads input data from CSV files and prepares it for the model.
    """
    try:
        load_profiles_df = pd.read_csv(DATA_DIR / "load_profile_nominal.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR / "shiftability_profile_nominal.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profile_nominal.csv' and 'shiftability_profile_nominal.csv' are in {DATA_DIR}")
        raise e

    inflexible = load_profiles_df['inflexible_load']
    base_flex = load_profiles_df['flexible_load']
    base_flex_t = base_flex.loc[list(params.T_SLOTS)]
    shiftability_df.columns = shiftability_df.columns.astype(int)
    shiftability = shiftability_df.stack().to_dict()

    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    
    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'flexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0),
        'shiftabilityProfile': resample_shiftability_profile(shiftability, RESAMPLE_FACTOR)
    }

    baseFlexibleLoadProfile_T = np.insert(np.array(np.repeat(base_flex_t.values, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0)
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours
    
    baseFlexibleLoadProfile_TEXT = np.insert(np.array(np.repeat(base_flex.values, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0)
    
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

# MODIFIED: Post-processing function adapted for Pyomo model
def post_process_results(m: pyo.ConcreteModel, params: ModelParameters, data: dict):
    """
    Extracts results from a solved Pyomo model into a DataFrame for the nominal case.
    """
    # --- Build results dictionary explicitly ---
    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'Total_Nominal_Cost': pyo.value(m.objective),
        'P_IT_Total_kW': [pyo.value(m.p_it_total_kw[s]) for s in params.TEXT_SLOTS],
        'P_Grid_IT_kW': [pyo.value(m.p_grid_it_kw[s]) for s in params.TEXT_SLOTS],
        'P_Chiller_HVAC_Watts': [pyo.value(m.p_chiller_hvac_w[s]) for s in params.TEXT_SLOTS],
        'P_Chiller_TES_Watts': [pyo.value(m.p_chiller_tes_w[s]) for s in params.TEXT_SLOTS],
        'P_Grid_Other_kW': [pyo.value(m.p_grid_od_kw[s]) for s in params.TEXT_SLOTS],
        'P_UPS_Charge_kW': [pyo.value(m.p_ups_ch_kw[s]) for s in params.TEXT_SLOTS],
        'P_UPS_Discharge_kW': [pyo.value(m.p_ups_disch_kw[s]) for s in params.TEXT_SLOTS],
        'E_UPS_kWh': [pyo.value(m.e_ups_kwh[s]) for s in params.TEXT_SLOTS],
        'T_IT_Celsius': [pyo.value(m.t_it[s]) for s in params.TEXT_SLOTS],
        'T_Rack_Celsius': [pyo.value(m.t_rack[s]) for s in params.TEXT_SLOTS],
        'T_ColdAisle_Celsius': [pyo.value(m.t_cold_aisle[s]) for s in params.TEXT_SLOTS],
        'T_HotAisle_Celsius': [pyo.value(m.t_hot_aisle[s]) for s in params.TEXT_SLOTS],
        'E_TES_kWh': [pyo.value(m.e_tes_kwh[s]) for s in params.TEXT_SLOTS],
        'Q_Cool_Total_Watts': [pyo.value(m.q_cool_w[s]) for s in params.TEXT_SLOTS],
        'Q_Charge_TES_Watts': [pyo.value(m.q_ch_tes_w[s]) for s in params.TEXT_SLOTS],
        'Q_Discharge_TES_Watts': [pyo.value(m.q_dis_tes_w[s]) for s in params.TEXT_SLOTS],
        'Total_CPU_Load': [pyo.value(m.total_cpu[s]) for s in params.TEXT_SLOTS],
    }
    
    # --- Derived calculations ---
    results['P_Chiller_HVAC_kW'] = [p / 1000.0 for p in results['P_Chiller_HVAC_Watts']]
    results['P_Chiller_TES_kW'] = [p / 1000.0 for p in results['P_Chiller_TES_Watts']]
    results['P_Grid_Cooling_kW'] = [(h + t) for h, t in zip(results['P_Chiller_HVAC_kW'], results['P_Chiller_TES_kW'])]
    results['P_Total_kW'] = [
        results['P_Grid_IT_kW'][i-1] + results['P_Grid_Cooling_kW'][i-1] +
        results['P_Grid_Other_kW'][i-1] + results['P_UPS_Charge_kW'][i-1]
        for i in params.TEXT_SLOTS
    ]
    results['Nominal_Cost'] = [
        params.dt_hours * results['P_Total_kW'][i-1] * (data['electricity_price'][i] / 1000.0)
        for i in params.TEXT_SLOTS
    ]
    
    # --- Add data from input ---
    results['Price_GBP_per_MWh'] = [data['electricity_price'][s] for s in params.TEXT_SLOTS]
    results['Inflexible_Load_CPU'] = [data['inflexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]
    results['Flexible_Load_CPU'] = [data['flexibleLoadProfile_TEXT'][s] for s in params.TEXT_SLOTS]

    df = pd.DataFrame(results)
    return df, pd.DataFrame()  # Return placeholder for flex_load_origin_df

def print_summary(results_df: pd.DataFrame):
    """Prints a summary of the nominal case results."""
    if results_df.empty:
        print("No results to summarize.")
        return

    total_nominal_cost = results_df['Nominal_Cost'].sum()

    print("\n" + "="*50)
    print("--- Nominal Case Results Summary ---")
    print(f"Total Nominal Cost: {total_nominal_cost:,.2f} GBP")
    print("="*50 + "\n")

def create_and_save_charts(df: pd.DataFrame, flex_load_origin_df: pd.DataFrame, data: dict, params: ModelParameters):
    """Generates and saves all charts based on the results DataFrame."""
    print("Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    time_slots_ext = df['Time_Slot_EXT']

    # --- Figure 1: Nominal Cost and Energy Price ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(time_slots_ext, df['Nominal_Cost'], label='Nominal Cost', color='crimson')
    ax1.set_ylabel('Cost Incurred (GBP/15 min)')
    ax1.set_title('Nominal DC Operational Cost')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2.plot(time_slots_ext, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Energy Price (GBP/MWh)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig1.tight_layout()
    fig1.savefig(IMAGE_DIR / 'nominal_cost_and_price.png')
    print("✅ Nominal cost chart saved.")

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
    ax6.plot(time_slots_ext, df['P_IT_Total_kW'] * 1000, label='Total Cooling Demand (Heat from IT)', color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Time Slot')
    ax6.set_ylabel('Thermal Power (Watts)')
    ax6.set_title('Cooling Power (q) by Source')
    ax6.legend(loc='upper left')
    ax6.grid(True)
    fig5.tight_layout()
    fig5.savefig(IMAGE_DIR / 'thermal_cooling_power.png')
    print("✅ Thermal cooling power chart saved.")

    # --- Figure 6: Cumulative Cost ---
    fig6, ax7 = plt.subplots(figsize=(12, 7))
    ax7.plot(time_slots_ext, df['Nominal_Cost'].cumsum(), label='Cumulative Nominal Cost', color='crimson', linewidth=2)
    ax7.set_xlabel('Time Slot')
    ax7.set_ylabel('Cumulative Cost (£)')
    ax7.set_title('Cumulative Nominal Energy Cost')
    ax7.legend()
    ax7.grid(True)
    fig6.tight_layout()
    fig6.savefig(IMAGE_DIR / 'cumulative_nominal_cost.png')
    print("✅ Cumulative cost chart saved.")

    # --- Figure 7: Stacked Bar Chart of IT Workload ---
    fig7, ax8 = plt.subplots(figsize=(18, 9))
    df_cpu = pd.DataFrame({
        'Inflexible': df['Inflexible_Load_CPU'],
        'Flexible': df['Flexible_Load_CPU']
    }, index=time_slots_ext)
    
    df_cpu.plot(kind='bar', stacked=True, ax=ax8, width=0.8, color=['black', 'gray'])
    
    ax8.set_title('Nominal IT Workload Composition', fontsize=16)
    ax8.set_xlabel('Processing Time Slot')
    ax8.set_ylabel('CPU Load Units')
    ax8.legend(title='Load Type')
    ax8.grid(axis='y', linestyle='--', alpha=0.7)

    tick_frequency = max(1, len(time_slots_ext) // 24)
    ax8.set_xticks(ax8.get_xticks()[::tick_frequency])
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha="right")
    
    fig7.tight_layout(rect=[0, 0, 0.9, 1])
    fig7.savefig(IMAGE_DIR / 'it_load_stacked_bar.png')
    print("✅ IT workload stacked bar chart saved.")

    plt.close('all')
    print("\nAll charts have been generated and saved.")

# MODIFIED: Updated solver invocation for Pyomo
def run_single_calculation(params: ModelParameters, input_data: dict, msg=False):
    """
    Runs a single calculation with a given set of parameters.
    """
    print(f"Building and solving model for nominal case...")
    model = build_model(params, input_data)

    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=msg)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Solver found an optimal solution. Post-processing...")
        results_df, flex_load_origin_df = post_process_results(model, params, input_data)
        total_cost = results_df['Total_Nominal_Cost'].iloc[0]
        return total_cost, results_df, flex_load_origin_df
    else:
        print(f"Solver did not find an optimal solution. Status: {results.solver.termination_condition}")
        return None, None, None

def run_nominal_case_generation():
    """
    Sets up and runs the nominal case calculation, saving results and charts.
    """
    print("1. Setting up model parameters for nominal run...")
    params = ModelParameters()
    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)

    total_cost, results_df, flex_load_origin_df = run_single_calculation(params, input_data, msg=True)

    if total_cost is not None:
        print_summary(results_df)
        create_and_save_charts(results_df, flex_load_origin_df, input_data, params)

        output_path = DATA_DIR / "nominal_case_results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nNominal case results successfully exported to '{output_path}'")

if __name__ == '__main__':
    run_nominal_case_generation()