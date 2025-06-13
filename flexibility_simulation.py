"""
flexibility_optimisation.py – calculates the duration of power flexibility
==========================================================================

* Takes the output of the first cost-optimisation as a baseline.
* For each timestep in the baseline, it calculates how long a given power
  increase or decrease can be sustained.
* This is achieved by running a step-by-step feasibility check.
* This version includes detailed logging to a text file for inspection.
* Install requirements: pip install pulp numpy pandas
"""

import pandas as pd
import numpy as np
import pulp
import os
import datetime
import json
import matplotlib.pyplot as plt

# Assuming 'parameters_optimisation' is a local module you have.
# If not, you'll need to provide a mock function like in the previous examples.
from parameters_optimisation import setup_simulation_parameters
import itertools


def load_original_results(filepath="static/optimisation_results.csv"):
    """Loads the results from the initial cost optimisation."""
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} was not found.")
        print("Please run the initial 'optimisation.py' script first to generate the results.")
        # Creating a dummy file for demonstration if the real one isn't found
        print("Creating a dummy 'optimisation_results.csv' for demonstration.")
        dummy_data = {
            't': np.arange(0, 500, 1), 'T_IT': 45, 'T_in': 22, 'T_Rack': 26, 'T_c': 25, 'T_h': 35,
            'E_TES': 100, 'P_HVAC': 100000, 'P_ch': 20000, 'P_dis': 0
        }
        # Ensure the static directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.DataFrame(dummy_data).to_csv(filepath, index=False)
    return pd.read_csv(filepath)

def setup_optimisation_parameters():
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
    # Use json for a clean, consistent format in the log file
    state_str = json.dumps(current_state, indent=4)
    report_lines.append(state_str)
    report_lines.append("---------------------------------\n")

    report_string = "\n".join(report_lines)
    
    # Print a concise message to the console
    print(f"\n[Failure at t={current_state.get('t', 'N/A'):.2f} min: {reason}. See log for details.]")

    # Write the full detailed report to the log file
    log_file.write(report_string)
    
    return False, None

def check_step_feasibility(p, current_state, original_state, original_power, delta_P, power_diff, log_file):
    """
    Checks if a power change (delta_P) is feasible for one timestep, logging all outcomes.
    """


    P_ch_new = current_state['P_ch']
    P_dis_new = current_state['P_dis']
    P_HVAC_new = current_state['P_HVAC']
    P_cooling_required = (current_state['T_h'] - original_state['T_in']) * \
                          (p['m_dot_air'] * p['c_p_air']) / p['COP_HVAC']
    if P_dis_new > P_cooling_required:
        P_dis_new = max(P_cooling_required,0)



    tolerance = 1e-6
    delta_P = delta_P + power_diff
    # --- CASE 0: Power correct but cooling equilibrium not met ---
    if delta_P == 0:
        if P_cooling_required < 0:
            if P_HVAC_new > abs(P_cooling_required):
                P_HVAC_new += P_cooling_required
                if P_ch_new < p['TES_w_charge_max'] + P_cooling_required:
                    P_ch_new = P_ch_new + P_cooling_required
                elif P_ch_new < p['TES_w_charge_max']:
                    P_ch_new = p['TES_w_charge_max']
                elif P_ch_new == p['TES_w_charge_max']:
                    pass # No change needed


    # --- CASE 1: INCREASE POWER (delta_P > 0) ---
    elif delta_P > 0:
        if P_dis_new == 0:
            if P_ch_new < p["TES_w_charge_max"] - delta_P:
                P_ch_new += delta_P
            elif P_ch_new < p["TES_w_charge_max"]:
                P_HVAC_new += delta_P - (p["TES_w_charge_max"] - P_ch_new)
                P_ch_new = p["TES_w_charge_max"]
            elif P_ch_new == p["TES_w_charge_max"]:
                P_HVAC_new += delta_P
        elif P_dis_new >0:
            if P_dis_new >= delta_P:
                P_dis_new -= delta_P
                P_HVAC_new += delta_P
            elif P_dis_new < delta_P:
                P_HVAC_new += delta_P - P_dis_new
                P_dis_new = 0



    # --- CASE 2: DECREASE POWER (delta_P < 0) ---
    elif delta_P < 0:
        delta_P_abs = abs(delta_P)
        if P_ch_new == 0:
            if P_HVAC_new >= delta_P_abs:
                P_HVAC_new -= delta_P_abs
                if P_dis_new < p['TES_w_discharge_max'] + delta_P_abs:
                    P_dis_new += delta_P_abs
                else:
                    P_dis_new = p['TES_w_discharge_max']
            else:
                return _report_failure(
                    "Cannot reduce HVAC power by the required amount.",
                    current_state,
                    {"Required HVAC Reduction (kW)": delta_P_abs / 1000, "Available HVAC Power (kW)": P_HVAC_new / 1000},
                    log_file
                )
        elif P_ch_new > 0:
            if P_ch_new >= delta_P_abs:
                P_ch_new -= delta_P_abs
            else:
                if P_HVAC_new >= delta_P_abs - P_ch_new:
                    P_HVAC_new -= delta_P_abs - P_ch_new
                    P_dis_new += delta_P_abs - P_ch_new
                    P_ch_new = 0
                else:
                    return _report_failure(
                        "Cannot reduce HVAC power by the required amount.",
                        current_state,
                        {"Required HVAC Reduction (kW)": delta_P_abs / 1000, "Available HVAC Power (kW)": P_HVAC_new / 1000},
                        log_file
                    )

    P_dis_new = max(P_cooling_required - P_HVAC_new, 0) 
    if current_state['E_TES'] <  0.5: #P_dis_new * p['dt_hours'] / 1000.0:
        P_ch_new += P_cooling_required - P_HVAC_new
        P_HVAC_new = P_cooling_required
    if current_state['E_TES'] > p['TES_capacity_kWh'] - 0.5:
        P_HVAC_new += P_ch_new
        P_ch_new = 0
                                  
    # --- FINAL VALIDATION ---
    dt_s, dt_h = p['dt'], p['dt_hours']
    mcp = p['m_dot_air'] * p['c_p_air']
    P_cool_new = P_HVAC_new + P_dis_new
    T_in_new = current_state['T_h'] - P_cool_new * p['COP_HVAC'] / mcp
    T_c_next = current_state['T_c'] + dt_s * ((p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_in_new - current_state['T_c']) - p['G_cold'] * (current_state['T_c'] - p['T_out_Celsius'])) / p['C_cAisle'])
    T_c_next = max(T_c_next, T_in_new)
    if T_c_next > p['T_cAisle_upper_limit_Celsius']:
        details = {"Predicted Cold Aisle Temp (°C)": T_c_next, "Cold Aisle Temp Limit (°C)": p['T_cAisle_upper_limit_Celsius']}
        return _report_failure("Cold aisle temperature limit exceeded.", current_state, details, log_file)
    if T_c_next < p['T_cAisle_lower_limit_Celsius']:
        details = {"Predicted Cold Aisle Temp (°C)": T_c_next, "Cold Aisle Temp Lower Limit (°C)": p['T_cAisle_lower_limit_Celsius']}
        return _report_failure("Cold aisle temperature lower limit exceeded.", current_state, details, log_file)
    
    E_TES_next = current_state['E_TES'] + ((P_ch_new * p['TES_charge_efficiency'] - P_dis_new / p['TES_discharge_efficiency']) * dt_h / 1000.0)
    if not (p['E_TES_min_kWh'] - tolerance <= E_TES_next <= p['TES_capacity_kWh'] + tolerance):
        details = {"Current TES Energy (kWh)": current_state['E_TES'], "Predicted TES Energy (kWh)": E_TES_next, "TES Min (kWh)": p['E_TES_min_kWh'], "TES Max (kWh)": p['TES_capacity_kWh']}
        return _report_failure("TES energy capacity limit exceeded.", current_state, details, log_file)
    
    T_IT_next = current_state['T_IT'] + dt_s * ((p['P_IT_heat_source'] - p['G_conv'] * (current_state['T_IT'] - current_state['T_Rack'])) / p['C_IT'])
    T_Rack_next = current_state['T_Rack'] + dt_s * ((p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (current_state['T_c'] - current_state['T_Rack']) + p['G_conv'] * (current_state['T_IT'] - current_state['T_Rack'])) / p['C_Rack'])
    T_Rack_next = max(T_Rack_next, T_in_new)
    T_Rack_next = min(T_Rack_next, T_IT_next)
    T_h_next = current_state['T_h'] + dt_s * ((p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (current_state['T_Rack'] - current_state['T_h'])) / p['C_hAisle'])
    T_h_next = min(T_h_next, T_Rack_next)
    #T_h_next = min(T_h_next, T_Rack_next)

    new_state = {
        'T_IT': T_IT_next, 'T_in': T_in_new, 'T_Rack': T_Rack_next, 'T_c': T_c_next,
        'T_h': T_h_next, 'E_TES': E_TES_next, 'P_HVAC': P_HVAC_new, 'P_ch': P_ch_new,
        'P_dis': P_dis_new, 'delta_P':float(delta_P), 'power_diff': power_diff, 'P_cooling_required': P_cooling_required 
    }
    
    log_file.write(f"\n--- Step Feasible: t={current_state.get('t', 'N/A'):.2f} -> t={current_state.get('t', 'N/A') + p['dt']/60.0:.2f} ---\n")
    log_file.write(json.dumps(new_state, indent=4) + "\n")

    return True, new_state

def calculate_flexibility_duration(p, original_results_df, start_index, delta_P, log_file):
    """Calculates flexibility duration, passing the log_file object to checkers."""
    original_state = original_results_df.loc[start_index].to_dict()
    current_state = original_results_df.loc[start_index].to_dict()
    duration_steps = 0
    power_diff = 0
    
    log_file.write("Initial State:\n")
    log_file.write(json.dumps(current_state, indent=4) + "\n")
    
    for t_step in range(start_index, len(original_results_df)):
        original_power = original_results_df.loc[t_step, ['P_HVAC', 'P_ch']].to_dict()
        is_feasible, new_state = check_step_feasibility(p, current_state, original_state, original_power, delta_P, power_diff, log_file)
        
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
    #original_results = original_results.iloc[:1440,:]
    #flexibility_range_kw = np.array([400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25])
    flexibility_range_kw = np.arange(5, 400, 5)
    flexibility_range_kw_negative = flexibility_range_kw * -1
    flexibility_range_kw = np.concatenate((flexibility_range_kw_negative[::-1], flexibility_range_kw))
    flexibility_range_w = flexibility_range_kw * 1000

    all_flexibility_results = []
    total_calculations = len(flexibility_range_w) * len(original_results)
    current_calculation = 0

    output_log_path = "flexibility_log.txt"
    print(f"\nStarting flexibility analysis... Detailed log will be saved to '{output_log_path}'")

    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Flexibility Analysis Log - {datetime.datetime.now().isoformat()}\n")
        log_file.write("="*70 + "\n")

        for delta_P in flexibility_range_w:
            for start_index in range(1):#range(0, len(original_results), 10):
                current_calculation += 1
                start_time_min = original_results.loc[start_index, 't']
                
                print(f"  [{current_calculation}/{total_calculations}] Calculating: "
                      f"start_time={int(start_time_min)} min, "
                      f"delta_P={delta_P / 1000:.1f} kW", end='\r')

                log_file.write(f"\n\n{'='*20} NEW CALCULATION {'='*20}\n")
                log_file.write(f"Start Time: {start_time_min} min (index: {start_index})\n")
                log_file.write(f"Requested Power Change (delta_P): {delta_P / 1000:.2f} kW\n")
                log_file.write(f"{'-'*55}\n")

                duration = calculate_flexibility_duration(
                    p, original_results, start_index, delta_P, log_file
                )
                
                result_entry = {
                    "start_time_minute": start_time_min,
                    "delta_P_kW": delta_P / 1000,
                    "duration_minutes": duration
                }
                all_flexibility_results.append(result_entry)
                
    print(f"\n\nFlexibility analysis complete. {current_calculation} scenarios calculated.")

    results_df = pd.DataFrame(all_flexibility_results)
    results_df.to_csv("flexibility_optimisation_results_logged.csv", index=False)
    results_df = pd.read_csv("flexibility_optimisation_results_logged.csv")
    start_times = results_df['start_time_minute'].unique()
    print(start_times)
    fig, ax = plt.subplots(figsize=(12, 6))
    color_cycle = itertools.cycle(plt.cm.tab20.colors)  # 20 unique colors
    marker_cycle = itertools.cycle(('o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'x', '+', '1', '2', '3', '4', '|', '_', '.', ','))
    used_colors = set()

    for start_time in start_times[::10]:
        color = next(color_cycle)
        marker = next(marker_cycle)
        while color in used_colors:
            color = next(color_cycle)
        used_colors.add(color)

        subset = results_df[results_df['start_time_minute'] == start_time]
        ax.scatter(subset['duration_minutes'], subset['delta_P_kW'], marker=marker, label=f'Start at {start_time} min', color=color)
        if start_time > 500:
            break
        ax.set_title('Flexibility Duration vs Magnitude') 
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Magnitude (kW)')
        ax.legend(title='Start Time (min)')
        plt.grid()
        plt.tight_layout()
        plt.savefig("flexibility_duration_plot.png")
        plt.show()

    output_path = "flexibility_optimisation_results_logged.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to '{output_path}'")
    print(f"Detailed log saved to '{output_log_path}'")


    """        power_needed = delta_P
        if P_dis_new > power_needed:
            P_dis_new -= power_needed
            power_needed = 0
        elif P_dis_new > 0:
            power_needed -= P_dis_new
            P_dis_new = 0
            P_ch_new = power_needed
        else:
            pch_increase_available = min(p['TES_p_ch_ramp'], p['TES_w_charge_max'] - P_ch_new)
            pch_increase_actual = min(power_needed, pch_increase_available)
            P_ch_new += pch_increase_actual
            power_needed -= pch_increase_actual

        if power_needed > tolerance:
            phvac_increase_available = min(p['P_HVAC_ramp'], p['P_HVAC_max_watts'] - (P_HVAC_new + P_ch_new))
            phvac_increase_actual = min(power_needed, phvac_increase_available)
            P_HVAC_new += phvac_increase_actual
            power_needed -= phvac_increase_actual
        
        if power_needed > tolerance:
            details = {"Required Power Increase (kW)": delta_P / 1000, "Power Not Met (kW)": power_needed / 1000}
            return _report_failure("Unable to meet power increase target", current_state, details, log_file)
            
                elif delta_P < 0:
        power_to_shed = abs(delta_P)
        pch_decrease_available = P_ch_new
        pch_decrease_actual = min(power_to_shed, pch_decrease_available)
        P_ch_new -= pch_decrease_actual
        power_to_shed -= pch_decrease_actual

        if power_to_shed > tolerance:
            hvac_reduction_needed = power_to_shed
            if hvac_reduction_needed > P_HVAC_new + tolerance:
                details = {"Required HVAC Reduction (kW)": hvac_reduction_needed / 1000, "Available HVAC Power (kW)": P_HVAC_new / 1000}
                return _report_failure("Cannot reduce HVAC power by the required amount.", current_state, details, log_file)
            pdis_increase_available = min(p['TES_p_dis_ramp'], p['TES_w_discharge_max'] - P_dis_new)
            if hvac_reduction_needed > pdis_increase_available + tolerance:
                details = {"Required TES Discharge (kW)": hvac_reduction_needed / 1000, "Available TES Discharge Increase (kW)": pdis_increase_available / 1000}
                return _report_failure("TES cannot ramp up discharge fast enough.", current_state, details, log_file)
            P_HVAC_new -= hvac_reduction_needed
            P_dis_new += hvac_reduction_needed
            power_to_shed -= hvac_reduction_needed

        if power_to_shed > tolerance:
            details = {"Required Power Shed (kW)": abs(delta_P) / 1000, "Power Not Shed (kW)": power_to_shed / 1000}
            return _report_failure("Unable to meet power reduction target.", current_state, details, log_file)"""