"""
flexibility_optimisation.py – calculates the duration of power flexibility
==========================================================================

* Takes the output of the first cost-optimisation as a baseline.
* For each timestep in the baseline, it calculates how long a given power
  increase or decrease can be sustained.
* This is achieved by running a step-by-step feasibility check.
* This version includes detailed logging and provides flexibility from either
  the TES tank OR the DC air's thermal inertia.
* Install requirements: pip install pulp numpy pandas matplotlib
"""

import pandas as pd
import numpy as np
import pulp
import os
import datetime
import json
import matplotlib.pyplot as plt
import itertools

# Assuming 'parameters_optimisation' is a local module you have.
from parameters_optimisation import setup_simulation_parameters


def load_original_results(filepath="static/optimisation_results.csv"):
    """Loads the results from the initial cost optimisation."""
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} was not found.")
        print("Please run the initial 'optimisation.py' script first to generate the results.")
        # Creating a dummy file for demonstration if the real one isn't found
        print("Creating a dummy 'optimisation_results.csv' for demonstration.")
        dummy_data = {
            't': np.arange(0, 500, 1), 'T_IT': 45, 'T_in': 22, 'T_Rack': 26, 'T_c': 21, 'T_h': 35,
            'E_TES': 150, 'P_HVAC': 80000, 'P_ch': 20000, 'P_dis': 0
        }
        # Ensure the static directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.DataFrame(dummy_data).to_csv(filepath, index=False)
    return pd.read_csv(filepath)

def setup_optimisation_parameters():
    """Sets up the parameters for the simulation."""
    p = setup_simulation_parameters("cool_down")
    # horizon & discretisation
    p['simulation_time_minutes'] = 500
    p['dt'] = 60
    p['simulation_time_seconds'] = p['simulation_time_minutes'] * 60
    p['num_time_points'] = int(p['simulation_time_seconds'] / p['dt'])
    p['dt_hours'] = p['dt'] / 3600.0
    # airflow
    p['m_dot_air'] = 100  # kg/s
    p['TES_capacity_kWh'] = p['TES_kwh_cap']
    # Add ramp rate if missing from parameters file
    p.setdefault('P_HVAC_ramp', 20000)
    return p

def _report_failure(reason: str, current_state: dict, details: dict, log_file):
    """A helper function to print and log detailed failure reports."""
    report_lines = []
    report_lines.append(f"\n--- FLEXIBILITY STEP FAILED (t={current_state.get('t', 'N/A'):.2f} min) ---")
    report_lines.append(f"Reason: {reason}")
    
    report_lines.append("\n[Failure Details]")
    for key, value in details.items():
        if isinstance(value, float):
            report_lines.append(f"  {key}: {value:.2f}")
        else:
            report_lines.append(f"  {key}: {value}")
            
    report_lines.append("\n[System State at Failure]")
    state_str = json.dumps(current_state, indent=4)
    report_lines.append(state_str)
    report_lines.append("---------------------------------\n")

    report_string = "\n".join(report_lines)
    
    # Print a concise message to the console
    # print(f"\n[Failure at t={current_state.get('t', 'N/A'):.2f} min: {reason}. See log for details.]")

    # Write the full detailed report to the log file
    log_file.write(report_string)
    
    return False, None

def check_step_feasibility(p, current_state, original_state, original_power, delta_P, power_diff, log_file, flexibility_source):
    """
    Checks if a power change (delta_P) is feasible for one timestep using a specified flexibility source.
    """
    # Initialize power variables from the current state
    P_ch_new = current_state['P_ch']
    P_dis_new = current_state['P_dis']
    P_HVAC_new = current_state['P_HVAC']
    tolerance = 1e-6

    # Required cooling power (in Watts, electrical) to maintain thermal equilibrium
    P_cooling_required = (current_state['T_h'] - original_state['T_in']) * \
                         (p['m_dot_air'] * p['c_p_air']) / p['COP_HVAC']
    P_cooling_required = max(0, P_cooling_required) # Cooling power cannot be negative

    # --- SIMPLIFIED LOGIC BASED ON FLEXIBILITY SOURCE ---

    if flexibility_source == 'TES':
        # Flexibility is provided by adjusting TES charge/discharge.
        # HVAC power is then set to meet the remaining cooling demand.
        if delta_P > 0: # --- POWER UP: Reduce discharge, then increase charge ---
            reduce_from_dis = min(delta_P, P_dis_new)
            P_dis_new -= reduce_from_dis
            remaining_delta_P = delta_P - reduce_from_dis
            
            if remaining_delta_P > 0:
                increase_to_ch = min(remaining_delta_P, p['TES_w_charge_max'] - P_ch_new)
                P_ch_new += increase_to_ch
                if remaining_delta_P - increase_to_ch > tolerance:
                    return _report_failure("Cannot provide power up flex: TES charge limit reached.", current_state, {"Required (W)": remaining_delta_P, "Provided (W)": increase_to_ch}, log_file)

        elif delta_P < 0: # --- POWER DOWN: Reduce charge, then increase discharge ---
            delta_P_abs = abs(delta_P)
            reduce_from_ch = min(delta_P_abs, P_ch_new)
            P_ch_new -= reduce_from_ch
            remaining_delta_P = delta_P_abs - reduce_from_ch
            
            if remaining_delta_P > 0:
                increase_to_dis = min(remaining_delta_P, p['TES_w_discharge_max'] - P_dis_new)
                P_dis_new += increase_to_dis
                if remaining_delta_P - increase_to_dis > tolerance:
                    return _report_failure("Cannot provide power down flex: TES discharge limit reached.", current_state, {"Required (W)": remaining_delta_P, "Provided (W)": increase_to_dis}, log_file)

        # Set HVAC power to meet the cooling demand not met by TES discharge
        P_HVAC_new = P_cooling_required - P_dis_new
        
    elif flexibility_source == 'DC_AIR':
        # Flexibility is provided by adjusting HVAC power, letting temperatures drift.
        # TES power levels are kept constant from the previous step.
        P_HVAC_new += delta_P

    # --- FINAL VALIDATION (common for both modes) ---
    if P_ch_new > tolerance and P_dis_new > tolerance:
        return _report_failure("TES cannot charge and discharge simultaneously.", current_state, {"P_ch": P_ch_new, "P_dis": P_dis_new}, log_file)
        
    if not (0 - tolerance <= P_HVAC_new <= p['P_HVAC_max_watts'] + tolerance):
        return _report_failure("HVAC power limit exceeded.", current_state, {"HVAC Power (kW)": P_HVAC_new / 1000, "Limit (kW)": p['P_HVAC_max_watts'] / 1000}, log_file)

    dt_s, dt_h = p['dt'], p['dt_hours']
    mcp = p['m_dot_air'] * p['c_p_air']
    
    # Calculate next state based on the new power levels
    P_cool_new = P_HVAC_new + P_dis_new
    T_in_new = current_state['T_h'] - P_cool_new * p['COP_HVAC'] / mcp
    
    # Check thermal constraints
    T_c_next = current_state['T_c'] + dt_s * ((mcp * p['kappa'] * (T_in_new - current_state['T_c']) - p['G_cold'] * (current_state['T_c'] - p['T_out_Celsius'])) / p['C_cAisle'])
    if T_c_next > p['T_cAisle_upper_limit_Celsius']:
        details = {"Predicted Cold Aisle Temp (°C)": T_c_next, "Limit (°C)": p['T_cAisle_upper_limit_Celsius']}
        return _report_failure("Cold aisle temperature upper limit exceeded.", current_state, details, log_file)
    if T_c_next < p['T_cAisle_lower_limit_Celsius']:
        details = {"Predicted Cold Aisle Temp (°C)": T_c_next, "Limit (°C)": p['T_cAisle_lower_limit_Celsius']}
        return _report_failure("Cold aisle temperature lower limit exceeded.", current_state, details, log_file)
    
    E_TES_next = current_state['E_TES'] + ((P_ch_new * p['TES_charge_efficiency'] - P_dis_new / p['TES_discharge_efficiency']) * dt_h / 1000.0)
    if not (p['E_TES_min_kWh'] - tolerance <= E_TES_next <= p['TES_capacity_kWh'] + tolerance):
        details = {"Predicted TES Energy (kWh)": E_TES_next, "Min (kWh)": p['E_TES_min_kWh'], "Max (kWh)": p['TES_capacity_kWh']}
        return _report_failure("TES energy capacity limit exceeded.", current_state, details, log_file)
    
    T_IT_next = current_state['T_IT'] + dt_s * ((p['P_IT_heat_source'] - p['G_conv'] * (current_state['T_IT'] - current_state['T_Rack'])) / p['C_IT'])
    # Add a safety check for IT temperature
    if T_IT_next > 55: # Example: Hard upper limit for IT equipment
         return _report_failure("IT equipment temperature safety limit exceeded.", current_state, {"Predicted IT Temp (°C)": T_IT_next, "Limit (°C)": 55}, log_file)

    T_Rack_next = current_state['T_Rack'] + dt_s * ((mcp * p['kappa'] * (T_c_next - current_state['T_Rack']) + p['G_conv'] * (current_state['T_IT'] - current_state['T_Rack'])) / p['C_Rack'])
    T_h_next = current_state['T_h'] + dt_s * ((mcp * p['kappa'] * (current_state['T_Rack'] - current_state['T_h'])) / p['C_hAisle'])

    new_state = {
        'T_IT': T_IT_next, 'T_in': T_in_new, 'T_Rack': T_Rack_next, 'T_c': T_c_next,
        'T_h': T_h_next, 'E_TES': E_TES_next, 'P_HVAC': P_HVAC_new, 'P_ch': P_ch_new,
        'P_dis': P_dis_new, 'delta_P': float(delta_P), 'power_diff': power_diff, 'P_cooling_required': P_cooling_required
    }
    
    log_file.write(f"\n--- Step Feasible: t={current_state.get('t', 'N/A'):.2f} -> t={current_state.get('t', 'N/A') + p['dt']/60.0:.2f} ---\n")
    log_file.write(json.dumps(new_state, indent=4) + "\n")

    return True, new_state


def calculate_flexibility_duration(p, original_results_df, start_index, delta_P, log_file, flexibility_source):
    """Calculates flexibility duration for a given mode (TES or DC_AIR)."""
    original_state = original_results_df.loc[start_index].to_dict()
    current_state = original_results_df.loc[start_index].to_dict()
    duration_steps = 0
    power_diff = 0
    
    log_file.write(f"Flexibility Source: {flexibility_source}\n")
    log_file.write("Initial State:\n")
    log_file.write(json.dumps(current_state, indent=4) + "\n")
    
    for t_step in range(start_index, len(original_results_df)):
        original_power = original_results_df.loc[t_step, ['P_HVAC', 'P_ch']].to_dict()
        is_feasible, new_state = check_step_feasibility(
            p, current_state, original_state, original_power, delta_P, power_diff, log_file, flexibility_source
        )
        
        if is_feasible:
            current_state.update(new_state)
            current_state['t'] = original_results_df.loc[start_index, 't'] + (duration_steps + 1) * (p['dt'] / 60)
            duration_steps += 1
            power_diff = original_power['P_HVAC'] + original_power['P_ch'] - (current_state['P_HVAC'] + current_state['P_ch'])
        else:
            break
            
    duration_minutes = duration_steps * (p['dt'] / 60)
    return duration_minutes


if __name__ == "__main__":
    print("Setting up simulation parameters...")
    p = setup_optimisation_parameters()

    print("Loading baseline optimisation results...")
    original_results = load_original_results()
    
    flexibility_range_kw = np.arange(10, 405, 10)
    flexibility_range_kw_negative = flexibility_range_kw * -1
    flexibility_range_kw = np.concatenate((flexibility_range_kw_negative[::-1], flexibility_range_kw))
    flexibility_range_w = flexibility_range_kw * 1000

    all_flexibility_results = []
    
    # --- Define the two flexibility modes to test ---
    flexibility_modes = ['TES', 'DC_AIR']
    
    total_calculations = len(flexibility_range_w) * len(original_results) * len(flexibility_modes)
    current_calculation = 0

    output_log_path = "flexibility_log.txt"
    print(f"\nStarting flexibility analysis... Detailed log will be saved to '{output_log_path}'")

    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Flexibility Analysis Log - {datetime.datetime.now().isoformat()}\n")
        log_file.write("="*70 + "\n")

        for flex_mode in flexibility_modes:
            print(f"\n===== ANALYSING FLEXIBILITY SOURCE: {flex_mode} =====")
            for delta_P in flexibility_range_w:
                # Running for a single start_index for demonstration speed
                for start_index in range(1):
                    current_calculation += 1
                    start_time_min = original_results.loc[start_index, 't']
                    
                    print(f"  [{current_calculation}/{total_calculations}] Calculating: "
                          f"mode={flex_mode}, start_time={int(start_time_min)} min, "
                          f"delta_P={delta_P / 1000:.1f} kW", end='\r')

                    log_file.write(f"\n\n{'='*20} NEW CALCULATION ({flex_mode}) {'='*20}\n")
                    log_file.write(f"Start Time: {start_time_min} min (index: {start_index})\n")
                    log_file.write(f"Requested Power Change (delta_P): {delta_P / 1000:.2f} kW\n")
                    log_file.write(f"{'-'*55}\n")

                    duration = calculate_flexibility_duration(
                        p, original_results, start_index, delta_P, log_file, flex_mode
                    )
                    
                    result_entry = {
                        "flexibility_source": flex_mode,
                        "start_time_minute": start_time_min,
                        "delta_P_kW": delta_P / 1000,
                        "duration_minutes": duration
                    }
                    all_flexibility_results.append(result_entry)
    
    print(f"\n\nFlexibility analysis complete. {current_calculation} scenarios calculated.")

    results_df = pd.DataFrame(all_flexibility_results)
    output_path = "flexibility_optimisation_results_simplified.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")

    # --- Plotting Results ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = {'TES': 'o', 'DC_AIR': 's'}
    colors = {'TES': 'blue', 'DC_AIR': 'green'}

    for mode in flexibility_modes:
        subset = results_df[results_df['flexibility_source'] == mode]
        if not subset.empty:
            ax.scatter(subset['duration_minutes'], subset['delta_P_kW'], 
                       marker=markers[mode], 
                       color=colors[mode],
                       label=f'Flexibility from {mode}',
                       alpha=0.8)

    ax.set_title('Flexibility Duration vs. Magnitude (by Source)')
    ax.set_xlabel('Sustainable Duration (minutes)')
    ax.set_ylabel('Power Change Magnitude (kW)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("flexibility_duration_by_source_plot.png")
    print("Plot saved to 'flexibility_duration_by_source_plot.png'")
    plt.show()

    print(f"Detailed log saved to '{output_log_path}'")