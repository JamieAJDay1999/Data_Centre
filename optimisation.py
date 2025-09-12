import datetime
import json
import pathlib
import pandas as pd
import numpy as np
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator # Import the ticker tool
from inputs.parameters_optimisation import ModelParameters, generate_tariff
from constraints import add_it_and_job_constraints, add_ups_constraints, add_power_balance_constraints, add_cooling_constraints
from plotting_and_saving.nom_opt_charts import gen_charts

# --- Path Configuration ------------------------------------------------------
# Define base directories for data, images, and debugging
DATA_DIR_INPUTS_1 = pathlib.Path("static/data/inputs")
DATA_DIR_INPUTS_2 = pathlib.Path("static/data/nominal_outputs")
DATA_DIR_OUTPUTS = pathlib.Path("static/data/optimisation_outputs")
IMAGE_DIR = pathlib.Path("static/images/optimisation_outputs")
DEBUG_DIR = pathlib.Path("lp_debug")

DATA_DIR_INPUTS_1.mkdir(parents=True, exist_ok=True)
DATA_DIR_INPUTS_2.mkdir(parents=True, exist_ok=True)
DATA_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

# --- Model Configuration -----------------------------------------------------
CYCLE_TES_ENERGY = True


FIGURE_SIZE = (14, 7) 
# Define the axes rectangle [left, bottom, width, height] in figure-fraction units.
# This leaves space on the right for the second chart's legend.
AXES_RECT = [0.1, 0.15, 0.75, 0.75] 

# --- Font Size Definitions (can be global) ---
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 14


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
    Loads input data from CSV files and prepares it for the optimization model.
    """
    # --- Load data from CSV files ---
    try:
        load_profiles_df = pd.read_csv(DATA_DIR_INPUTS_1 / "load_profiles.csv", index_col='time_slot')
        shiftability_df = pd.read_csv(DATA_DIR_INPUTS_1 / "shiftability_profile.csv", index_col='time_slot')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. Make sure 'load_profiles.csv' and 'shiftability_profile.csv' are in {DATA_DIR_INPUTS_1}")
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

import pandas as pd

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
    
    #flex_load_origin_df.to_csv(DATA_DIR_OUTPUTS / "flex_load_origin_df.csv")

    flex_filtered = flex_load_origin_df[flex_load_origin_df['shiftability'] > 0]  # Flexible only
    new_flexible_load_per_slot = flex_filtered.groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)  # Sum cpu_load only, for 96 slots
    new_inflexible_load_per_slot = flex_load_origin_df[flex_load_origin_df['shiftability'] == 0].groupby('processing_slot')['cpu_load'].sum().reindex(range(1, 109), fill_value=0)
    new_inflexible_load = data['inflexibleLoadProfile_TEXT'][1:109] + new_inflexible_load_per_slot.values  # Align to slots 1-96
    load_profiles_df = pd.DataFrame({
        'inflexible_load': new_inflexible_load,
        'flexible_load': new_flexible_load_per_slot.values
    }, index=range(1, 109))
    load_profiles_df.index.name = 'time_slot'
    load_profiles_df.to_csv(DATA_DIR_OUTPUTS / 'load_profiles_opt.csv')

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
    shiftability_df.to_csv(DATA_DIR_OUTPUTS / 'shiftability_profile_opt.csv')

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
        df_nominal = pd.read_csv(DATA_DIR_INPUTS_2 / "nominal_case_results.csv")
        df['Nominal_Cost'] = df_nominal['Nominal_Cost'][:len(df)].values
        df['P_Total_kW_Nominal'] = df_nominal['P_Total_kW'][:len(df)].values
    except Exception:
        print(f"Warning: Could not load or align '{DATA_DIR_INPUTS_2 / 'nominal_case_results.csv'}'. Nominal cost set to 0.")
        df['Nominal_Cost'] = 0
        df['P_Total_kW_Nominal'] = 0

    return df, flex_load_origin_df

def print_summary(params, results_df: pd.DataFrame):
    
    for t in range(96, 108):
        cost_diff_in_extension = results_df['Optimized_Cost_per_Step'].iloc[t] - results_df['Nominal_Cost'].iloc[t]

    print(f"Cost difference in extension period (slots 97-108): {cost_diff_in_extension:,.2f} GBP")
    nominal_cost = results_df['Nominal_Cost'].iloc[:96].sum()
    optimized_cost = results_df['Optimized_Cost_per_Step'].iloc[:96].sum()
    optimized_cost += cost_diff_in_extension
    cost_saving_abs = nominal_cost - optimized_cost if nominal_cost > 0 else 0
    cost_saving_rel = (cost_saving_abs / nominal_cost) * 100 if nominal_cost > 0 else 0

    print("\n" + "="*50)
    print("--- Optimization Results Summary ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print(f"Baseline (Nominal) Cost: {nominal_cost:,.2f} GBP")
    print(f"Absolute Cost Saving: {cost_saving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {cost_saving_rel:.2f} %")
    print("="*50 + "\n")

def create_power_chart(df: pd.DataFrame, IMAGE_DIR: pathlib.Path):
    """Generates and saves the power consumption chart with fixed dimensions."""
    print("Generating power consumption chart...")
    
    # --- Create the figure with a fixed size ---
    fig = plt.figure(figsize=FIGURE_SIZE)
    # --- Add axes with a fixed position and size ---
    ax1 = fig.add_axes(AXES_RECT)

    x_indices = np.arange(len(df))
    num_datapoints = len(df)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax1.plot(x_indices, df['P_Total_kW_Nominal'], label='Base DC Power', linestyle='--', color='gray')
    ax1.plot(x_indices, df['P_Total_kW'], label='Optimized DC Power', color='crimson')
    ax1.set_ylabel('Power (kW)', fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    ax2 = ax1.twinx()
    ax2.plot(x_indices, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_ylabel('Energy Price (£/MWh)', color='royalblue', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='royalblue', labelsize=TICK_FONTSIZE)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=LEGEND_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Time (Hours)', fontsize=LABEL_FONTSIZE)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/4:g}'))
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=0)
    ax1.set_xlim(-0.5, num_datapoints - 0.5)

    # REMOVED: fig.tight_layout() is no longer needed
    fig.savefig(IMAGE_DIR / 'power_consumption_chart.png')
    print("✅ Power consumption chart saved.")
    plt.show()


def create_workload_chart(df: pd.DataFrame, flex_load_origin_df: pd.DataFrame, data: dict, IMAGE_DIR: pathlib.Path):
    """Generates and saves the IT workload chart with fixed dimensions."""
    print("Generating IT workload chart...")
    if flex_load_origin_df.empty:
        print("Skipping workload chart: flex_load_origin_df is empty.")
        return

    # --- Create the figure with the same fixed size ---
    fig = plt.figure(figsize=FIGURE_SIZE)
    # --- Add axes with the same fixed position and size ---
    ax3 = fig.add_axes(AXES_RECT)
    
    num_datapoints = len(df)

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    # ... (rest of the data processing for the bar chart is identical)
    flex_load_origin_df['lag'] = flex_load_origin_df['processing_slot'] - flex_load_origin_df['original_slot']
    flex_pivot_by_lag = flex_load_origin_df.pivot_table(
        index='processing_slot', columns='lag', values='cpu_load', aggfunc='sum'
    ).fillna(0)
    plot_df = pd.DataFrame({'Inflexible': df['Inflexible_Load_CPU_Nom'].values}, index=df['Time_Slot_EXT'])
    plot_df = plot_df.join(flex_pivot_by_lag).fillna(0)
    rename_dict = {lag: f'Flexible (Lag {int(lag)})' for lag in flex_pivot_by_lag.columns}
    plot_df.rename(columns=rename_dict, inplace=True)
    flexible_cols = [col for col in plot_df.columns if 'Flexible' in col]
    num_flexible_lags = len(flexible_cols)
    power = 0.5
    non_linear_space = np.linspace(0, 1, num_flexible_lags) ** power
    cmap = plt.cm.get_cmap('plasma')
    color_indices = 0.2 + (non_linear_space * 0.8)
    flexible_colors = cmap(color_indices)
    colors = ['black'] + list(flexible_colors)
    plot_df.plot(kind='bar', stacked=True, ax=ax3, width=0.8, color=colors)
    nominal_total_load = data['inflexibleLoadProfile_TEXT'][1:] + data['flexibleLoadProfile_TEXT'][1:]
    ax3.plot(range(len(nominal_total_load)), nominal_total_load, label='Base Workload', linestyle='--', color='gray')
    # ... (rest of the plotting and formatting is identical)
    
    ax3.set_xlabel('Time (Hours)', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('CPU Load Units', fontsize=LABEL_FONTSIZE)
    # The legend is outside the axes, but our AXES_RECT leaves space for it.
    ax3.legend(title='Load Type', bbox_to_anchor=(1, 1.0), loc='upper left', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/4:g}'))
    ax3.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=0)
    ax3.set_xlim(-0.5, num_datapoints - 0.5)

    # REMOVED: fig.tight_layout() is no longer needed
    fig.savefig(IMAGE_DIR / 'it_workload_chart.png')
    print("✅ IT workload chart saved.")
    plt.show()


def create_and_save_charts(df: pd.DataFrame, flex_load_origin_df: pd.DataFrame, data: dict, params: 'ModelParameters', IMAGE_DIR: pathlib.Path):
    """Generates and saves two separate charts for power and workload with identical sizes."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    create_power_chart(df, IMAGE_DIR)
    create_workload_chart(df, flex_load_origin_df, data, IMAGE_DIR)
    
    print("\nAll charts have been generated and saved.")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pathlib

def create_power_stack_chart(df: pd.DataFrame, image_dir: pathlib.Path):
    """
    Generates and saves a stacked bar chart of power resources, showing consumption
    as positive bars and discharge as negative bars. The legend is ordered to show
    UPS items before TES items.
    """
    print("Generating final power resource chart as stacked bars...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Font Size Definitions ---
    TITLE_FONTSIZE = 22
    LABEL_FONTSIZE = 16
    LEGEND_FONTSIZE = 14
    TICK_FONTSIZE = 14

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Data Preparation ---
    timeslots_per_hour = 4
    x_hours = df.index / timeslots_per_hour
    bar_width = 1.0 / timeslots_per_hour

    # --- 1. Plot Stacked POSITIVE BARS (GRID POWER CONSUMPTION) ---
    # MODIFIED: Reordered dictionary and color list for desired legend order.
    power_components = {
        'IT Power': 'P_Grid_IT_kW',
        'HVAC Power': 'P_Chiller_HVAC_kW',
        'UPS Charging Power': 'P_UPS_Charge_kW',      # Moved Up
        'TES Charging Power': 'P_Chiller_TES_kW',      # Moved Down
        'Other DC Power': 'P_Grid_Other_kW'
    }
    # MODIFIED: Reordered colors to match the new component order.
    colors_consume = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b']

    y_consume = [df[col] for col in power_components.values()]
    labels_consume = list(power_components.keys())

    # Loop to create the stacked positive bars
    bottom_positive = pd.Series([0.0] * len(df), index=df.index)
    for i, component in enumerate(y_consume):
        ax.bar(x_hours, component, width=bar_width, label=labels_consume[i], color=colors_consume[i], bottom=bottom_positive, alpha=0.8)
        bottom_positive += component

    # --- 2. Plot Stacked NEGATIVE BARS (DISCHARGED POWER) ---
    ups_discharge_kw = df['P_UPS_Discharge_kW']
    tes_discharge_kw = df['Q_Discharge_TES_Watts'] / (1000.0 * 6)
    
    y_discharge = [-ups_discharge_kw, -tes_discharge_kw]
    labels_discharge = ['UPS Discharge', 'TES Discharge']
    # MODIFIED: Updated color indices to match the reordered colors_consume list.
    # UPS color is now at index 2, TES color is at index 3.
    colors_discharge = [colors_consume[2], colors_consume[3]]

    # Loop to create the stacked negative bars
    bottom_negative = pd.Series([0.0] * len(df), index=df.index)
    for i, component in enumerate(y_discharge):
        ax.bar(x_hours, component, width=bar_width, label=labels_discharge[i], color=colors_discharge[i], bottom=bottom_negative, alpha=0.8)
        bottom_negative += component

    # --- 3. Plot Nominal Power for Comparison ---
    ax.plot(x_hours, df['P_Total_kW_Nominal'], label='Base DC Power', linestyle=':', color='black', linewidth=2.5)

    # --- 4. Formatting and Final Touches ---
    ax.set_ylabel('Power (kW)', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel('Time (Hours)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_xlim(x_hours.min() - bar_width, x_hours.max() + bar_width)
    min_y_limit = bottom_negative.min()
    ax.set_ylim(bottom=min_y_limit * 1.1)
    
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # --- Save and Show ---
    fig.tight_layout()
    output_path = image_dir / 'power_resource_bar_chart_reordered.png'
    fig.savefig(output_path)
    print(f"✅ Power bar chart saved to '{output_path}'")
    plt.show()

def run_single_optimization(params: ModelParameters, input_data: dict, msg=False):
    """
    Runs a single optimization instance with a given set of parameters using Pyomo.
    """
    print("Building and solving model with Pyomo...")
    model = build_model(params, input_data)

    solver = pyo.SolverFactory('scip')

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

def run_full_optimisation(include_charts):
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
        if include_charts:
            create_and_save_charts(results_df, flex_load_origin_df, input_data, params, IMAGE_DIR)
            create_power_stack_chart(results_df, IMAGE_DIR)

        output_path = DATA_DIR_OUTPUTS / "optimised_baseline.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nResults successfully exported to '{output_path}'")

if __name__ == '__main__':
    run_full_optimisation(include_charts=True)