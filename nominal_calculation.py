import datetime
import json
import pathlib
import pandas as pd
import numpy as np
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from inputs.parameters_optimisation import ModelParameters, generate_tariff
from constraints import add_it_and_job_constraints, add_ups_constraints, add_power_balance_constraints, add_cooling_constraints
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
    add_cooling_constraints(m, params, CYCLE_TES_ENERGY)

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

def run_nominal_case_generation(include_charts):
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
        if include_charts:
            create_and_save_charts(results_df, flex_load_origin_df, input_data, params)
        output_path = DATA_DIR_OUTPUTS / "nominal_case_results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nNominal case results successfully exported to '{output_path}'")

if __name__ == '__main__':
    run_nominal_case_generation(include_charts=True)