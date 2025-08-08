import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from copy import deepcopy
import time
import matplotlib.ticker as mticker

# --- Import necessary components from the original model script ---
# This assumes 'integrated_dc_model.py' is in the same directory or accessible in the python path.
try:
    from integrated_dc_model import ModelParameters, load_and_prepare_data, run_single_optimization, IMAGE_DIR
except ImportError:
    print("Error: Could not import from 'integrated_dc_model.py'.")
    print("Please ensure the file is in the same directory and you have applied the refactoring changes.")
    exit()

def plot_workload_flow(flex_load_origin_df: pd.DataFrame, price_data: np.ndarray, params: ModelParameters):
    """
    Generates and saves a chart visualizing the flow of flexible IT workload.
    This corrected version uses polygons to create a clear, non-crossing flow diagram.

    Args:
        flex_load_origin_df (pd.DataFrame): DataFrame containing the details of shifted jobs.
        price_data (np.ndarray): Array of electricity prices for each time slot.
        params (ModelParameters): The model parameters instance.
    """
    print("Generating IT workload flow chart...")
    if flex_load_origin_df.empty:
        print("No flexible load data to plot. Skipping flow chart.")
        return

    fig, ax = plt.subplots(figsize=(20, 12))
    
    # --- Prepare data for plotting ---
    # Sort by arrival time, then by the amount of delay, to get a stable, logical order.
    # This is crucial for the stacking logic to prevent crossing.
    df = flex_load_origin_df.copy()
    df['delay'] = df['processing_slot'] - df['original_slot']
    df = df.sort_values(
        by=['original_slot', 'delay', 'tranche']
    ).reset_index(drop=True)

    # Calculate the Y-position at the START of the flow (arrival time)
    df['y_start_bottom'] = df.groupby('original_slot')['cpu_load'].cumsum() - df['cpu_load']
    df['y_start_top'] = df.groupby('original_slot')['cpu_load'].cumsum()

    # Calculate the Y-position at the END of the flow (processing time)
    # The order of stacking at the end is determined by the dataframe's sorted order.
    processing_slot_heights = {t: 0 for t in df['processing_slot'].unique()}
    y_proc_bottom_list = []
    y_proc_top_list = []
    for index, row in df.iterrows():
        processing_slot = row['processing_slot']
        cpu_load = row['cpu_load']
        
        bottom = processing_slot_heights[processing_slot]
        y_proc_bottom_list.append(bottom)
        
        top = bottom + cpu_load
        y_proc_top_list.append(top)

        processing_slot_heights[processing_slot] = top
        
    df['y_proc_bottom'] = y_proc_bottom_list
    df['y_proc_top'] = y_proc_top_list
    
    # --- Create polygons for the flow plot ---
    patches = []
    colors = []

    # Use a robust quantile normalization for prices to handle outliers
    price_subset = price_data[price_data > 0]
    norm = mcolors.Normalize(
        vmin=np.percentile(price_subset, 5), 
        vmax=np.percentile(price_subset, 95)
    )
    cmap = plt.cm.viridis_r # Inverted viridis: yellow=cheap, purple=expensive

    for _, row in df.iterrows():
        # Define the 4 corners of the polygon for the flow
        verts = [
            (row['original_slot'], row['y_start_bottom']),
            (row['original_slot'], row['y_start_top']),
            (row['processing_slot'], row['y_proc_top']),
            (row['processing_slot'], row['y_proc_bottom']),
        ]
        patches.append(Polygon(verts))
        colors.append(cmap(norm(price_data[int(row['processing_slot'])])))

    # Create a PatchCollection for efficient plotting
    p = PatchCollection(patches, facecolors=colors, alpha=0.7, zorder=10)
    ax.add_collection(p)

    # --- Plot total workload profiles for context ---
    total_flex_load_per_slot = df.groupby('original_slot')['cpu_load'].sum()
    ax.fill_between(total_flex_load_per_slot.index, 0, total_flex_load_per_slot.values, 
                    color='gray', alpha=0.4, label='Arriving Flexible Workload', zorder=1)

    processed_flex_load_per_slot = df.groupby('processing_slot')['cpu_load'].sum()
    ax.plot(processed_flex_load_per_slot.index, processed_flex_load_per_slot.values, 
            color='crimson', linestyle='--', label='Processed Flexible Workload', zorder=2)

    ax.autoscale_view()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max(params.TEXT_SLOTS))
    ax.set_xlabel('Time Slot (15-minute intervals)', fontsize=12)
    ax.set_ylabel('CPU Load Units', fontsize=12)
    ax.set_title('IT Flexible Workload Flow: Arrival vs. Processing Time', fontsize=16, pad=20)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left')

    # --- Add a colorbar for the price ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Electricity Price at Processing Time (£/MWh)', fontsize=12)
    
    fig.tight_layout()
    save_path = IMAGE_DIR / 'it_workload_flow.png'
    fig.savefig(save_path, dpi=300)
    print(f"✅ Workload flow chart saved to {save_path}")
    plt.close(fig)


def run_sensitivity_analysis():
    """
    Performs a sensitivity analysis on key model parameters to understand their
    impact on the total optimized cost.

    This function iterates through a predefined set of parameters, varies them
    one at a time across a specified range, runs the optimization for each
    variation, and plots the results.
    """
    print("\n" + "="*50)
    print("--- Starting Sensitivity Analysis ---")
    print("This will run the optimization multiple times and may take a while.")
    print("="*50)

    # --- Define parameters and ranges for testing ---
    params_to_test = {
        'e_nom_kwh': {
            'values': np.linspace(300, 900, 5),
            'baseline': 600.0,
            'label': 'UPS Nominal Capacity (kWh)'
        },
        'eta_ch': {
            'values': np.linspace(0.75, 0.95, 5),
            'baseline': 0.82,
            'label': 'UPS Charging Efficiency (η_ch)'
        },
        'TES_kwh_cap': {
            'values': np.linspace(800, 2500, 5),
            'baseline': 1500.0,
            'label': 'TES Capacity (kWh)'
        },
        'COP_HVAC': {
            'values': np.linspace(3.0, 5.5, 5),
            'baseline': 4.0,
            'label': 'Chiller COP'
        },
        'p_max_ch_kw': {
            'values': np.linspace(150, 400, 5),
            'baseline': 270.0,
            'label': 'UPS Max Charge Rate (kW)'
        },
        'T_cAisle_upper_limit_Celsius': {
            'values': np.linspace(24, 30, 5),
            'baseline': 27.0,
            'label': 'Cold Aisle Temp Limit (°C)'
        },
        'TES_w_charge_max': {
            'values': np.linspace(300000, 700000, 5),
            'baseline': 500000.0,
            'label': 'TES Max Charge Rate (Watts)'
        },
        'nominal_overhead_factor': {
            'values': np.linspace(0.05, 0.25, 5),
            'baseline': 0.1,
            'label': 'DC Overhead Factor'
        }
    }

    baseline_params = ModelParameters()
    analysis_results = {}

    # --- Run a single baseline optimization to get the reference cost ---
    print("Running baseline optimization for reference cost...")
    baseline_input_data = load_and_prepare_data(baseline_params)
    baseline_total_cost, _, _ = run_single_optimization(baseline_params, baseline_input_data, msg=False)
    if baseline_total_cost is None:
        print("FATAL: Baseline optimization failed. Cannot proceed with sensitivity analysis.")
        return
    print(f"Reference Baseline Cost: £{baseline_total_cost:,.2f}")


    start_time = time.time()
    for param_name, config in params_to_test.items():
        print(f"\n--- Testing Parameter: {param_name} ---")
        costs = []
        for value in config['values']:
            print(f"  Running with {param_name} = {value:.2f}...")
            
            temp_params = deepcopy(baseline_params)
            setattr(temp_params, param_name, value)
            
            # --- CRITICAL: Recalculate all derived parameters ---
            if param_name == 'e_nom_kwh':
                temp_params.e_min_kwh = temp_params.soc_min * temp_params.e_nom_kwh
                temp_params.e_max_kwh = temp_params.soc_max * temp_params.e_nom_kwh
                # Ensure the start/end energy is valid for the new capacity
                temp_params.e_start_kwh = temp_params.e_max_kwh 
            
            if param_name == 'TES_kwh_cap':
                temp_params.TES_capacity_kWh = value
                # Assume initial/final charge is a percentage of new capacity
                temp_params.TES_initial_charge_kWh = value * 0.9 
                temp_params.E_TES_min_kWh = value * 0.1

            # --- CORRECTED LOGIC: Regenerate input data for each run ---
            # This ensures the data is consistent with the modified parameters.
            temp_input_data = load_and_prepare_data(temp_params)

            total_cost, _, _ = run_single_optimization(temp_params, temp_input_data, msg=False)
            
            if total_cost is not None:
                costs.append(total_cost)
            else:
                print("    -> Infeasible solution found.")
                costs.append(np.nan)

        analysis_results[param_name] = {'values': config['values'], 'costs': costs}

    end_time = time.time()
    print(f"\n--- Sensitivity Analysis Complete in {end_time - start_time:.2f} seconds ---")

    # --- Plot the results ---
    num_params = len(params_to_test)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 24)) # Increased figure size
    axes = axes.flatten()
    fig.suptitle('Sensitivity Analysis: Impact of Key Parameters on Total Cost', fontsize=24, y=0.99)

    for i, (param_name, config) in enumerate(params_to_test.items()):
        ax = axes[i]
        result = analysis_results[param_name]
        
        # Filter out NaN values for plotting
        valid_points = pd.DataFrame(result).dropna()
        
        if not valid_points.empty:
            # Use the parameter's label for the legend
            ax.plot(valid_points['values'], valid_points['costs'], marker='o', linestyle='-', color='royalblue', label=f'{config["label"]}')
            
        # Plot baseline value consistently on all charts
        ax.plot(config['baseline'], baseline_total_cost, 'o', color='crimson', markersize=10, label=f'Baseline Value (£{baseline_total_cost:,.0f})')
        
        # --- FORMATTING CHANGES ---
        # Remove individual chart titles
        # ax.set_title(f'Cost vs. {config["label"]}', fontsize=14)
        
        # Ensure full numerical form for axes
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='both')

        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.legend()

    # Use tight_layout with a rect to prevent the main title from overlapping subplots
    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=4, w_pad=3) # Add padding
    
    save_path = IMAGE_DIR / 'sensitivity_analysis.png'
    fig.savefig(save_path, dpi=300)
    print(f"\n✅ Sensitivity analysis plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    # --- 1. Run Baseline Optimization to get data for the Flow Chart ---
    print("--- Running Baseline Optimization to Generate Data for Charts ---")
    baseline_params = ModelParameters()
    input_data = load_and_prepare_data(baseline_params)
    total_cost, results_df, flex_load_origin_df = run_single_optimization(baseline_params, input_data, msg=False)

    if total_cost is not None:
        # --- 2. Create and save the IT Workload Flow Chart ---
        plot_workload_flow(flex_load_origin_df, input_data['electricity_price'], baseline_params)
    else:
        print("Baseline optimization failed. Cannot generate flow chart.")

    # --- 3. Run the Sensitivity Analysis ---
    run_sensitivity_analysis()