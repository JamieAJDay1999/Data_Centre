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
    Builds the Pyomo optimization model, defining all variables, constraints, and the objective.
    """
    # MODIFIED: Use Pyomo's ConcreteModel
    m = pyo.ConcreteModel(name="DC_Cost_Optimization")

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

    # Job Scheduling Variables
    ut_ks_idx = [(t, k, s) for t in m.T_SLOTS for k in m.K_TRANCHES for s in m.TEXT_SLOTS if s >= t and s <= t + params.tranche_max_delay[k]]
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
    # --- IT Job Scheduling Constraints ---
    m.JobCompletion = pyo.ConstraintList()
    for t in m.T_SLOTS:
        for k in m.K_TRANCHES:
            expr = sum(m.ut_ks[(t, k, s)] * params.dt_hours for s in m.TEXT_SLOTS if (t,k,s) in m.ut_ks_idx)
            m.JobCompletion.add(expr == data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0))

    # --- Manual Piecewise Linearization using SOS2 Constraints ---

    # 1. Define the (x,y) points for the approximation of y = x**1.32
    num_pw_points = 11
    pw_x = [i / (num_pw_points - 1) for i in range(num_pw_points)] # x-coordinates (CPU usage)
    pw_y = [x**1.32 for x in pw_x] # y-coordinates (power factor)
    m.PW_POINTS = pyo.RangeSet(0, num_pw_points - 1)

    # 2. Create weighting variables for each time slot 's' and approximation point 'i'.
    m.w = pyo.Var(m.TEXT_SLOTS, m.PW_POINTS, within=pyo.NonNegativeReals)
    
    # 3. Add constraints for the weights.
    m.WeightSum = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # The weights for each time slot must sum to 1.
        m.WeightSum.add(sum(m.w[s, i] for i in m.PW_POINTS) == 1)

    # FIX: Correctly declare the indexed SOS2 constraint
    # We define a rule that, for each time slot 's', returns the list of variables for the SOS2 set.
    def sos_rule(model, s):
        return [model.w[s, i] for i in model.PW_POINTS]
    m.CPU_SOS2 = pyo.SOSConstraint(m.TEXT_SLOTS, rule=sos_rule, sos=2)

    # 4. Define total_cpu and the power factor based on these weights.
    m.CPUandPower = pyo.ConstraintList()
    m.PowerFactorDef = pyo.ConstraintList()
    m.cpu_power_factor = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    
    for s in m.TEXT_SLOTS:
        flexible_usage = sum(m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s)
        base_cpu = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage
        m.CPUandPower.add(m.total_cpu[s] == base_cpu)
        m.CPUandPower.add(m.total_cpu[s] == sum(pw_x[i] * m.w[s, i] for i in m.PW_POINTS))
        m.PowerFactorDef.add(m.cpu_power_factor[s] == sum(pw_y[i] * m.w[s, i] for i in m.PW_POINTS))

        # p_it_total_kw is in kW
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.cpu_power_factor[s]
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
    m.CoolingConstraints.add(m.p_chiller_hvac_w[1] == sum(m.p_chiller_hvac_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) -1))
    m.CoolingConstraints.add(m.p_chiller_tes_w[1] == sum(m.p_chiller_tes_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) -1))
    
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

# --- DATA LOADING AND CHARTING FUNCTIONS (No changes needed) ---
# The functions load_and_prepare_data, resample_shiftability_profile, 
# generate_tariff, print_summary, and create_and_save_charts remain the same.
# The only function that needs modification is post_process_results to extract
# data from the Pyomo model instead of the PuLP model.

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
    #data = [96.52, 91.65, 86.25, 84.2, 88.0, 96.86, 106.94, 107.04, 103.02, 82.14, 69.95, 68.06, 47.4, 35.0, 41.3, 69.51, 77.31, 103.84, 115.69, 131.04, 126.97, 112.82, 104.93, 99.87]
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)

# MODIFIED: A more robust version of the post-processing function
def post_process_results(m: pyo.ConcreteModel, params: ModelParameters, data: dict):
    """
    Extracts results from a solved Pyomo model into a DataFrame.
    This version is more robust and avoids errors with differently-indexed variables.
    """
    # --- Extract Flexible Workload Details (this part is correct) ---
    tranche_map = params.tranche_max_delay
    flexible_load_details = []
    # Use the variable's index set for iteration
    for (t, k, s) in m.ut_ks.index_set():
        val = pyo.value(m.ut_ks[t, k, s])
        if val is not None and val > 1e-6:
            flexible_load_details.append({
                'processing_slot': s,
                'original_slot': t,
                'tranche': k,
                'cpu_load': val,
                'shiftability': tranche_map[k] - (s-t)
            })
    flex_load_origin_df = pd.DataFrame(flexible_load_details)
    if not flex_load_origin_df.empty:
        flex_load_origin_df = flex_load_origin_df.sort_values(by=['processing_slot', 'tranche']).reset_index(drop=True)
    
    # (The rest of your flex load processing remains the same...)

    # --- Build results dictionary explicitly ---
    # This is safer than looping through all model components
    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS),
        'Total_Optimized_Cost': pyo.value(m.objective),
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

    # (The rest of the function for derived calculations and adding nominal cost remains the same...)
    # --- Derived calculations ---
    results['P_Chiller_HVAC_kW'] = [p / 1000.0 for p in results['P_Chiller_HVAC_Watts']]
    results['P_Chiller_TES_kW'] = [p / 1000.0 for p in results['P_Chiller_TES_Watts']]
    results['P_Grid_Cooling_kW'] = [(h + t) for h, t in zip(results['P_Chiller_HVAC_kW'], results['P_Chiller_TES_kW'])]
    results['P_Total_kW'] = [
        results['P_Grid_IT_kW'][i-1] + results['P_Grid_Cooling_kW'][i-1] +
        results['P_Grid_Other_kW'][i-1] + results['P_UPS_Charge_kW'][i-1]
        for i in params.TEXT_SLOTS
    ]
    results['Optimized_Cost_per_Step'] = [
        params.dt_hours * results['P_Total_kW'][i-1] * (data['electricity_price'][i] / 1000.0)
        for i in params.TEXT_SLOTS
    ]

    # --- Add data from input ---
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

    # --- Add nominal cost comparison ---
    try:
        df_nominal = pd.read_csv(DATA_DIR / "nominal_case_results.csv")
        df['Nominal_Cost'] = df_nominal['Nominal_Cost'][:len(df)].values
        df['P_Total_kW_Nominal'] = df_nominal['P_Total_kW'][:len(df)].values
    except Exception:
        print(f"Warning: Could not load or align '{DATA_DIR / 'nominal_case_results.csv'}'. Nominal cost set to 0.")
        df['Nominal_Cost'] = 0
        df['P_Total_kW_Nominal'] = 0

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

# MODIFIED: Solver invocation changed to Pyomo with Ipopt
# MODIFIED: Manually load results to handle non-optimal termination
def run_single_optimization(params: ModelParameters, input_data: dict, msg=False):
    """
    Runs a single optimization instance with a given set of parameters using Pyomo.
    """
    print("Building and solving model with Pyomo...")
    model = build_model(params, input_data)

    solver = pyo.SolverFactory('scip')
    solver.options['limits/gap'] = 0.01

    # Change 1: Add 'load_solutions=False' to the solve command.
    # This tells Pyomo to run the solver but wait to load the results.
    results = solver.solve(model, tee=msg, load_solutions=False)

    # We now check the termination condition from the raw results object.
    # The condition 'other' is acceptable if a primal bound (a solution) exists.
    if results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit] or \
       (results.solver.termination_condition == pyo.TerminationCondition.other and results.problem.lower_bound != results.problem.upper_bound):

        print("Solver found a feasible solution. Loading results for post-processing...")
        
        # Change 2: Manually load the solution from the results object into the model.
        # This is the crucial step that was being missed before.
        model.solutions.load_from(results)

        results_df, flex_load_origin_df = post_process_results(model, params, input_data)
        total_cost = pyo.value(model.objective)
        return total_cost, results_df, flex_load_origin_df
    else:
        print(f"Solver did not find a feasible solution. Status: {results.solver.termination_condition}")
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