import datetime
import json
import pathlib
import pandas as pd
import numpy as np
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from inputs.parameters_optimisation import ModelParameters, generate_tariff
# MODIFIED: Removed import of add_it_and_job_constraints, as it will be defined locally
from constraints import add_ups_constraints, add_power_balance_constraints, add_cooling_constraints
from plotting_and_saving.nom_opt_charts import gen_charts
# --- Path Configuration ------------------------------------------------------
DATA_DIR_INPUTS = pathlib.Path("static/data/inputs")
DATA_DIR_OUTPUTS = pathlib.Path("static/data/nominal_outputs")
IMAGE_DIR = pathlib.Path("static/images/nominal_outputs")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR_INPUTS.mkdir(parents=True, exist_ok=True)
DATA_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


# --- Model Configuration -----------------------------------------------------
CYCLE_TES_ENERGY = True



def build_model(params: ModelParameters, data: dict, linear: bool = False):
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
    # %%%%%%%% CRUCIAL CONSTRAINT CHANGE FOR NOMINAL CASE: s == t rather than s <= t %%%%%%%%%%%%%
    ut_ks_idx = [(t, k, s) for t in m.T_SLOTS for k in m.K_TRANCHES for s in m.TEXT_SLOTS if s == t]

    m.ut_ks_idx = pyo.Set(initialize=ut_ks_idx)
    m.ut_ks = pyo.Var(m.ut_ks_idx, within=pyo.NonNegativeReals, initialize=0)

    # Cooling System Variables
    m.t_it = pyo.Var(m.TEXT_SLOTS, bounds=(18, 60), initialize=25)
    m.t_rack = pyo.Var(m.TEXT_SLOTS, bounds=(18, 40), initialize=25)
    m.t_cold_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(18, params.T_cAisle_upper_limit_Celsius), initialize=20)
    m.t_hot_aisle = pyo.Var(m.TEXT_SLOTS, bounds=(18, 40), initialize=30)
    m.e_tes_kwh = pyo.Var(m.TEXT_SLOTS, bounds=(params.E_TES_min_kWh, params.TES_capacity_kWh), initialize=params.TES_initial_charge_kWh)
    # Electrical power for cooling in Watts
    m.p_chiller_hvac_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.p_chiller_tes_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    # Thermal power in Watts
    m.q_cool_w = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals, initialize=0)
    m.q_ch_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_charge_max), initialize=0)
    m.q_dis_tes_w = pyo.Var(m.TEXT_SLOTS, bounds=(0, params.TES_w_discharge_max), initialize=0)
    m.t_in = pyo.Var(m.TEXT_SLOTS, bounds=(18, 30), initialize=20)


    # --- Add Constraints ---
    if not linear:
        # Define Piecewise Linear helper variables and constraints
        m.num_pw_points = 11
        m.PW_POINTS = pyo.RangeSet(0, m.num_pw_points - 1)
        m.pw_x = {i: i / (m.num_pw_points - 1) for i in m.PW_POINTS}
        m.pw_y = {i: m.pw_x[i] ** 1.32 for i in m.PW_POINTS}
        
        m.w = pyo.Var(m.TEXT_SLOTS, m.PW_POINTS, within=pyo.NonNegativeReals)
        m.cpu_power_factor = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
        m.cpu_sos2 = pyo.SOSConstraint(m.TEXT_SLOTS, var=m.w, sos=2)

        add_it_and_job_constraints_pwl_nominal(m, params, data)
    else:
        add_it_and_job_constraints_linear_nominal(m, params, data)

    add_power_balance_constraints_nominal(m, params)
    add_cooling_constraints(m, params, CYCLE_TES_ENERGY)
    # --- Define Objective Function ---
    def objective_rule(mod):
        # Objective combines all grid-drawn power in kW
        cost = sum(
            params.dt_hours * (
                mod.p_grid_it_kw[s] +
                (mod.p_chiller_hvac_w[s] / 1000.0) + # W to kW
                (mod.p_chiller_tes_w[s] / 1000.0) +  # W to kW
                params.nominal_overhead_addition +
                mod.p_ups_ch_kw[s]
            ) * (data['electricity_price'][s] / 1000.0) # Price is per MWh
            for s in mod.TEXT_SLOTS
        )
        return cost
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m

def add_it_and_job_constraints_linear_nominal(m, params, data):
    """Adds job and IT constraints using a direct linear relationship."""
    m.LinearITConstraints = pyo.ConstraintList()
    
    # --- Job Completion & Total CPU ---
    for t in m.T_SLOTS:
        # Job completion for each tranche
        for k in m.K_TRANCHES:
            # For nominal case, s=t is handled by ut_ks_idx definition
            if (t, k, t) in m.ut_ks_idx:
                expr = m.ut_ks[t, k, t] * params.dt_hours
                rhs = data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0)
                m.LinearITConstraints.add(expr == rhs)

    for s in m.TEXT_SLOTS:
        # Total CPU usage
        flex_use_terms = [m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s]
        flexible_usage = sum(flex_use_terms) if flex_use_terms else 0.0
        base_cpu = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage
        m.LinearITConstraints.add(m.total_cpu[s] == base_cpu)

        # Linear relationship between CPU and IT power
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.total_cpu[s]
        m.LinearITConstraints.add(m.p_it_total_kw[s] == power_expr)

def add_it_and_job_constraints_pwl_nominal(m, params, data):
    """Adds job and IT constraints using a Piecewise Linear (PWL) relationship."""
    m.PWLITConstraints = pyo.ConstraintList()

    # --- Job Completion & Total CPU ---
    for t in m.T_SLOTS:
        # Job completion for each tranche
        for k in m.K_TRANCHES:
            if (t, k, t) in m.ut_ks_idx:
                expr = m.ut_ks[t, k, t] * params.dt_hours
                rhs = data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0)
                m.PWLITConstraints.add(expr == rhs)

    for s in m.TEXT_SLOTS:
        # Total CPU usage
        flex_use_terms = [m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s]
        flexible_usage = sum(flex_use_terms) if flex_use_terms else 0.0
        base_cpu = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage
        m.PWLITConstraints.add(m.total_cpu[s] == base_cpu)

        # --- PWL Constraints ---
        m.PWLITConstraints.add(m.total_cpu[s] == sum(m.pw_x[i] * m.w[s, i] for i in m.PW_POINTS))
        m.PWLITConstraints.add(m.cpu_power_factor[s] == sum(m.pw_y[i] * m.w[s, i] for i in m.PW_POINTS))
        m.PWLITConstraints.add(sum(m.w[s, i] for i in m.PW_POINTS) == 1)
        
        # Link IT power to the PWL power factor
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.cpu_power_factor[s]
        m.PWLITConstraints.add(m.p_it_total_kw[s] == power_expr)

def add_power_balance_constraints_nominal(m, params):
    m.PowerBalance = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # All power variables in this balance are in kW
        m.PowerBalance.add(m.p_it_total_kw[s] == m.p_grid_it_kw[s])
        #m.PowerBalance.add(m.p_grid_od_kw[s] == m.p_it_total_kw[s] * params.nominal_overhead_factor)


def load_and_prepare_data(params: ModelParameters):
    """
    Loads input data from CSV files and prepares it for the model.
    """
    try:
        load_profiles_df = pd.read_csv(DATA_DIR_INPUTS / "load_profiles.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR_INPUTS / "shiftability_profile.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profile_nominal.csv' and 'shiftability_profile_nominal.csv' are in {DATA_DIR_INPUTS}")
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
        'P_Grid_Other_kW': [params.nominal_overhead_addition] * len(params.TEXT_SLOTS),  # Fixed value for other DC loads
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

def print_summary(results_df: pd.DataFrame, params: ModelParameters):
    """
    Prints a summary of the nominal case results, including energy breakdown.
    """
    if results_df.empty:
        print("No results to summarize.")
        return

    total_nominal_cost = results_df['Nominal_Cost'].sum()

    # --- Energy Breakdown Calculation ---
    # Energy (kWh) = Power (kW) * duration (h)
    # dt_hours is the duration of a single time step
    dt_hours = params.dt_hours
    
    it_energy_kwh = results_df['P_Grid_IT_kW'].sum() * dt_hours
    cooling_energy_kwh = results_df['P_Grid_Cooling_kW'].sum() * dt_hours
    # 'Other' includes the fixed overhead and any UPS charging
    other_energy_kwh = (results_df['P_Grid_Other_kW'].sum() + results_df['P_UPS_Charge_kW'].sum()) * dt_hours
    
    total_energy_kwh = results_df['P_Total_kW'].sum() * dt_hours

    # Calculate percentages
    if total_energy_kwh > 0:
        it_percent = (it_energy_kwh / total_energy_kwh) * 100
        cooling_percent = (cooling_energy_kwh / total_energy_kwh) * 100
        other_percent = (other_energy_kwh / total_energy_kwh) * 100
    else:
        it_percent, cooling_percent, other_percent = 0, 0, 0
    # --- End of Calculation ---

    print("\n" + "="*50)
    print("--- Nominal Case Results Summary ---")
    print(f"Total Nominal Cost: {total_nominal_cost:,.2f} GBP")
    
    # --- Print Energy Breakdown ---
    print("\n--- Energy Consumption Breakdown ---")
    print(f"Total Energy Consumed: {total_energy_kwh:,.2f} kWh")
    print(f"  - IT Equipment:     {it_energy_kwh:,.2f} kWh ({it_percent:.1f}%)")
    print(f"  - Cooling System:   {cooling_energy_kwh:,.2f} kWh ({cooling_percent:.1f}%)")
    print(f"  - Other Equipment:  {other_energy_kwh:,.2f} kWh ({other_percent:.1f}%)")
    
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

    gen_charts(df, time_slots_ext, IMAGE_DIR)

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
def run_single_calculation(params: ModelParameters, input_data: dict, msg=False, linear=False):
    """
    Runs a single calculation with a given set of parameters.
    """
    model_type = "Linear" if linear else "Piecewise-Linear"
    print(f"Building and solving model for nominal case ({model_type})...")
    model = build_model(params, input_data, linear=linear)

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
    
def configure_nominal_params(params):

    params.T_cAisle_lower_limit_Celsius = 22
    params.TES_kwh_cap = 1 
    params.TES_initial_charge_kWh = 0
    params.TES_w_discharge_max = 0
    params.TES_w_charge_max = 0
    return params


def run_nominal_case_generation(include_charts, linear: bool = False):
    """
    Sets up and runs the nominal case calculation, saving results and charts.
    """
    print("1. Setting up model parameters for nominal run...")
    params = ModelParameters()
    params = configure_nominal_params(params)
    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)

    total_cost, results_df, flex_load_origin_df = run_single_calculation(params, input_data, msg=True, linear=linear)

    if total_cost is not None:
        print_summary(results_df, params)
        if include_charts:
            create_and_save_charts(results_df, flex_load_origin_df, input_data, params)
        output_path = DATA_DIR_OUTPUTS / "nominal_case_results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nNominal case results successfully exported to '{output_path}'")

if __name__ == '__main__':
    # Run the default piecewise-linear model
    run_nominal_case_generation(include_charts=True, linear=False)
    
    # Example of how to run the linear model:
    # print("\n\n--- RUNNING WITH LINEAR CPU-POWER MODEL ---")
    # run_nominal_case_generation(include_charts=True, linear=True)