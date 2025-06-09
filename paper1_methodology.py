# paper1_methodology.py (Modified plot_simulation_results function)
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import setup_simulation_parameters
import os # Added for path operations
import traceback # For detailed error logging

# ... (all your other functions: simulation_params, initialize_simulation_arrays, etc. remain unchanged)
# Make sure they are all present in your actual file. I'm only showing the modified one and its dependencies.

def simulation_params(p):
    p['simulation_time_seconds'] = p['simulation_time_minutes'] * 60
    if p['dt'] <= 0:
        p['num_time_points'] = 1
    else:
        p['num_time_points'] = max(1, int(p['simulation_time_seconds'] / p['dt']))
    p['time_points'] = np.linspace(0, p['simulation_time_seconds'], p['num_time_points'])
    if p['dt'] > 0 :
        p['num_begin_change'] = p['begin_change'] * 60 / p['dt']
    else:
        p['num_begin_change'] = 0
    return p

def initialize_simulation_arrays(num_time_points):
    T_IT_sim = np.zeros(num_time_points)
    T_Rack_sim = np.zeros(num_time_points)
    T_cAisle_sim = np.zeros(num_time_points)
    T_cWall_sim = np.zeros(num_time_points)
    T_hAisle_sim = np.zeros(num_time_points)
    T_Air_in_sim = np.zeros(num_time_points)
    P_HVAC_sim = np.zeros(num_time_points)
    P_TES_sim = np.zeros(num_time_points)
    E_TES_sim = np.zeros(num_time_points)
    P_Cooling_sim = np.zeros(num_time_points)
    return T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim,\
        P_TES_sim, E_TES_sim, P_Cooling_sim

def change_target_temp(p, num_begin_change, target_temp_change, i):
    if p['simulation_mode'] == "cool_down":
        if not target_temp_change:
            if i > num_begin_change:
                p['T_target_Air_in_Celsius'] = 14
                target_temp_change = True
    return p, target_temp_change

def _set_initial_simulation_conditions(p, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim,\
    T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, P_TES_sim, E_TES_sim, P_Cooling_sim, p_init=None):
    if p_init is None: 
        T_Air_in_sim[0] = p['T_Air_in_initial']
        T_cAisle_sim[0] = p['T_cAisle_initial']
        T_cWall_sim[0] = p['T_cWall_initial_Celsius']
        T_Rack_sim[0] = p['T_Rack_initial_Celsius']
        T_IT_sim[0] = p['T_IT_initial_Celsius']
        T_hAisle_sim[0] = p['T_hAisle_initial']
        P_HVAC_required_initial = (T_hAisle_sim[0] - p['T_target_Air_in_Celsius']) * \
                                (p['m_dot_air'] * p['c_p_air']) / p['COP_HVAC']
        P_HVAC_sim[0] = np.clip(P_HVAC_required_initial, p['P_HVAC_min_watts'], p['P_HVAC_max_watts'])
        P_TES_sim[0] = (P_HVAC_required_initial - P_HVAC_sim[0]) * p['TES_discharge_efficiency']
        E_TES_sim[0] = p['TES_initial_charge_kWh']
        P_Cooling_sim[0] = P_HVAC_sim[0] + P_TES_sim[0]
    else:
        T_Air_in_sim[0]  = p_init['T_in']          # column F in screenshot
        T_cAisle_sim[0]  = p_init['T_c']           # column D
        T_cWall_sim[0]   = p['T_cWall_initial_Celsius'] # use if column exists
        T_Rack_sim[0]    = p_init['T_Rack']        # column C
        T_IT_sim[0]      = p_init['T_IT']          # column B
        T_hAisle_sim[0]  = p_init['T_h']           # column E

        # HVAC and TES power can also come straight from the table … --------
        P_HVAC_sim[0]    = p_init['P_HVAC']        # column H
        P_TES_sim[0]     = p_init['P_ch'] - p_init['P_dis']  # net TES flow (I − J)
        E_TES_sim[0]     = p_init['E_TES']         # column G
        print(T_Rack_sim)

def _get_previous_step_temperatures(i, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim):
    T_IT_prev = T_IT_sim[i-1]
    T_Rack_prev = T_Rack_sim[i-1]
    T_cAisle_prev = T_cAisle_sim[i-1]
    T_cWall_prev = T_cWall_sim[i-1]
    T_hAisle_prev = T_hAisle_sim[i-1]
    return T_IT_prev, T_Rack_prev, T_cAisle_prev, T_cWall_prev, T_hAisle_prev

def _calculate_hvac_power_and_inlet_temp(p, T_hAisle_prev, T_cAisle_prev, i):
    P_HVAC_required_current = (T_hAisle_prev - p['T_target_Air_in_Celsius']) * \
                                   (p['m_dot_air'] * p['c_p_air']) / p['COP_HVAC']
    if p['simulation_mode'] == "warm_up" and p['warm_up'] == 'Before':
        if i > p['num_begin_change'] and T_cAisle_prev < p['T_cAisle_upper_limit_Celsius']:
            P_HVAC_required_current = p['P_HVAC_min_watts']
        if T_cAisle_prev >= p['T_cAisle_upper_limit_Celsius']:
            p['warm_up'] = 'After'
    P_HVAC_current = np.clip(P_HVAC_required_current, p['P_HVAC_min_watts'], p['P_HVAC_max_watts'])
    P_TES_required_current = (P_HVAC_required_current - P_HVAC_current) * p['TES_discharge_efficiency']
    P_TES_current = np.clip(P_TES_required_current, p['TES_w_discharge_min'], p['TES_w_discharge_max'])
    P_Cooling_current = P_HVAC_current + P_TES_current
    T_Air_in_current = T_hAisle_prev - (P_Cooling_current * p['COP_HVAC']) / \
                       (p['m_dot_air'] * p['c_p_air'])
    return P_HVAC_current, P_TES_current, P_Cooling_current, T_Air_in_current, p

def _calculate_temperature_derivatives(p, T_IT_prev, T_Rack_prev, T_cAisle_prev, T_cWall_prev, T_hAisle_prev, T_Air_in_current):
    dT_IT_dt = (p['P_IT_heat_source'] - p['G_conv'] * (T_IT_prev - T_Rack_prev)) / p['C_IT']
    dT_Rack_dt = (p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_cAisle_prev - T_Rack_prev) +
                  p['G_conv'] * (T_IT_prev - T_Rack_prev)) / p['C_Rack']
    dT_cAisle_dt = (p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_Air_in_current - T_cAisle_prev) -
                    p['G_cold'] * (T_cAisle_prev - T_cWall_prev)) / p['C_cAisle']
    dT_cWall_dt = (((p['T_out_Celsius'] - T_cWall_prev) / p['R_cold_K_per_W']) +
                   p['G_cold'] * (T_cAisle_prev - T_cWall_prev)) / p['C_cWall']
    if p['C_hAisle'] > 0:
        dT_hAisle_dt = (p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_Rack_prev - T_hAisle_prev)) / p['C_hAisle']
    else:
        dT_hAisle_dt = 0
    return (dT_IT_dt, dT_Rack_dt, dT_cAisle_dt, dT_cWall_dt, dT_hAisle_dt)

def _update_current_step_temperatures(p, i, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim,
                                   T_IT_prev, T_Rack_prev, T_cAisle_prev, T_cWall_prev, T_hAisle_prev,
                                   derivatives):
    dT_IT_dt, dT_Rack_dt, dT_cAisle_dt, dT_cWall_dt, dT_hAisle_dt = derivatives
    T_IT_sim[i] = T_IT_prev + dT_IT_dt * p['dt']
    T_Rack_sim[i] = T_Rack_prev + dT_Rack_dt * p['dt']
    T_cAisle_sim[i] = T_cAisle_prev + dT_cAisle_dt * p['dt']
    T_cWall_sim[i] =  p['T_out_Celsius']  #T_cWall_prev + dT_cWall_dt * p['dt']
    T_hAisle_sim[i] = T_hAisle_prev + dT_hAisle_dt * p['dt']

def _update_tes_charge(p, i, E_TES_sim, P_TES_sim):
    E_TES_sim[i] = E_TES_sim[i-1] - (P_TES_sim[i] * p['dt']) / 3600000
    return E_TES_sim

def _log_simulation_progress(p, i, T_IT_prev, T_Rack_prev, T_cAisle_prev, T_hAisle_prev, T_Air_in_current, P_HVAC_current):
    print_interval = int(60 / p['dt']) if p['dt'] > 0 else p['num_time_points']
    if i % print_interval == 0 or i == p['num_time_points'] -1 :
        print(f"Time: {p['time_points'][i]/60:.1f} min, T_IT: {T_IT_prev:.2f}, T_Rack: {T_Rack_prev:.2f}, "
              f"T_cAisle: {T_cAisle_prev:.2f}, T_hAisle: {T_hAisle_prev:.2f}, "
              f"T_Air_in: {T_Air_in_current:.2f}, P_HVAC: {P_HVAC_current:.0f}W, "
              f"WarmUpActive: {p.get('hvac_active_during_warmup', 'N/A')}") # Note: 'hvac_active_during_warmup' key might not exist, using p.get()

def _check_steady_state(derivatives, steady_iters, max_steady_iters):
    all_derivatives_zero = all(d == 0 for d in derivatives)
    if all_derivatives_zero:
        steady_iters += 1
    else:
        steady_iters = 0
    break_loop = False
    if steady_iters >= max_steady_iters:
        print("Simulation has reached a steady state. Exiting loop.")
        break_loop = True
    return steady_iters, break_loop

def run_simulation(p, p_init):
    p = simulation_params(p)
    T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim,\
        P_TES_sim, E_TES_sim, P_Cooling_sim = initialize_simulation_arrays(p['num_time_points'])
    _set_initial_simulation_conditions(p, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim,
        T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, P_TES_sim, E_TES_sim, P_Cooling_sim, p_init)
    steady_iters = 0
    max_steady_iters = 10
    target_temp_change = False
    for i in range(1, p['num_time_points']):
        T_IT_prev, T_Rack_prev, T_cAisle_prev, T_cWall_prev, T_hAisle_prev = \
            _get_previous_step_temperatures(i, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim)
        p, target_temp_change = change_target_temp(p, p['num_begin_change'], target_temp_change, i)
        P_HVAC_sim[i], P_TES_sim[i], P_Cooling_sim[i], T_Air_in_sim[i], p = \
            _calculate_hvac_power_and_inlet_temp(p, T_hAisle_prev, T_cAisle_prev, i)
        derivatives = _calculate_temperature_derivatives(p, T_IT_prev, T_Rack_prev, T_cAisle_prev,
                                                       T_cWall_prev, T_hAisle_prev, T_Air_in_sim[i])
        _update_current_step_temperatures(p, i, T_IT_sim, T_Rack_sim, T_cAisle_sim, T_cWall_sim, T_hAisle_sim,
                                       T_IT_prev, T_Rack_prev, T_cAisle_prev, T_cWall_prev, T_hAisle_prev,
                                       derivatives)
        _log_simulation_progress(p, i, T_IT_prev, T_Rack_prev, T_cAisle_prev, T_hAisle_prev,
                                 T_Air_in_sim[i], P_HVAC_sim[i])
        E_TES_sim = _update_tes_charge(p, i, E_TES_sim, P_TES_sim)
        steady_iters, break_loop = _check_steady_state(derivatives, steady_iters, max_steady_iters)
        if break_loop:
            # Fill remaining arrays if breaking early, to ensure consistent lengths for plotting
            for k in range(i + 1, p['num_time_points']):
                T_IT_sim[k] = T_IT_sim[i]
                T_Rack_sim[k] = T_Rack_sim[i]
                T_cAisle_sim[k] = T_cAisle_sim[i]
                T_cWall_sim[k] = T_cWall_sim[i] # Assuming T_cWall_sim exists and is relevant
                T_hAisle_sim[k] = T_hAisle_sim[i]
                T_Air_in_sim[k] = T_Air_in_sim[i]
                P_HVAC_sim[k] = P_HVAC_sim[i]
                P_TES_sim[k] = P_TES_sim[i]
                E_TES_sim[k] = E_TES_sim[i]
                P_Cooling_sim[k] = P_Cooling_sim[i]
            # Update num_time_points to reflect actual simulation length if needed for plotting logic
            # p['num_time_points'] = i + 1 # Or handle this in plotting function
            break
    return T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, \
        E_TES_sim, P_TES_sim

def calculate_flexibility(P_HVAC_sim, p, current_simulation_mode):
    # Ensure num_begin_change is an integer for indexing if used directly
    num_begin_change_idx = int(p.get('num_begin_change', 0))

    # print(p['num_begin_change']) # This refers to a step count, not index directly for steady state
    if current_simulation_mode == "warm_up":
        # Find the last point where HVAC power was effectively zero
        # This logic might need refinement based on expected P_HVAC_sim behavior in warm_up
        zero_power_indices = np.where(P_HVAC_sim < 1e-3)[0] # Consider a small threshold for "zero"
        if len(zero_power_indices) > 0:
            i = zero_power_indices[-1]
            P_HVAC_off = P_HVAC_sim[i]
        else: # HVAC never off or starts on
            i = 0 # Or handle as error/specific case
            P_HVAC_off = P_HVAC_sim[0]

        P_HVAC_steady_state_val = P_HVAC_sim[-1]
        P_HVAC_steady_state = P_HVAC_steady_state_val * np.ones(len(P_HVAC_sim))
        # Flexibility calculation needs to be based on the duration HVAC was off
        # Assuming 'i' is the index *until* which HVAC was off.
        duration_seconds = i * p['dt']
        downside_flexibility = ((P_HVAC_steady_state_val - P_HVAC_off) * duration_seconds) / 3600000 # kWh
        print(f"Downside flexibility: {np.round(downside_flexibility,3)} kWh")
    else: # cool_down or other modes
        # Ensure num_begin_change_idx is within bounds
        if num_begin_change_idx > 0 and num_begin_change_idx < len(P_HVAC_sim):
             P_HVAC_steady_state_val = P_HVAC_sim[num_begin_change_idx -1]
        elif len(P_HVAC_sim) > 0 : # Fallback if num_begin_change is 0 or out of bounds
            P_HVAC_steady_state_val = P_HVAC_sim[0]
        else: # P_HVAC_sim is empty
             P_HVAC_steady_state_val = 0

        P_HVAC_steady_state = P_HVAC_steady_state_val * np.ones(len(P_HVAC_sim))
        # Upside flexibility calculation should consider the period *after* the change
        if num_begin_change_idx < len(P_HVAC_sim):
            relevant_P_HVAC_sim = P_HVAC_sim[num_begin_change_idx:]
            relevant_P_HVAC_steady = P_HVAC_steady_state[num_begin_change_idx:]
            upside_flexibility = np.sum((relevant_P_HVAC_sim - relevant_P_HVAC_steady) * p['dt']) / 3600000
        else: # Change happens at or after end of simulation
            upside_flexibility = 0
        print(f"Upside flexibility: {np.round(upside_flexibility,3)} kWh")
    return P_HVAC_steady_state


def plot_simulation_results(T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim,\
        P_HVAC_sim, P_HVAC_steady_state, E_TES_sim, P_TES_sim, p,
        output_filename1="static/plot_temperatures_hvac.png",
        output_filename2="static/plot_tes.png"):

    print(f"\n[PLOTTER] --- Starting plot_simulation_results ---")
    print(f"[PLOTTER] Received parameters p: {p.get('simulation_mode', 'Mode N/A')}, num_time_points: {p.get('num_time_points', 'N/A')}")
    print(f"[PLOTTER] Length of T_IT_sim: {len(T_IT_sim)}")

    # Determine the correct 'static' directory path relative to this script file
    # __file__ is the path to the current script (paper1_methodology.py)
    # os.path.dirname(__file__) is the directory containing this script
    # os.path.join(...) creates a platform-independent path to the 'static' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'static')

    print(f"[PLOTTER] Script directory: {script_dir}")
    print(f"[PLOTTER] Target static directory: {static_dir}")

    if not os.path.exists(static_dir):
        print(f"[PLOTTER] Static directory '{static_dir}' does NOT exist. Attempting to create it.")
        try:
            os.makedirs(static_dir)
            print(f"[PLOTTER] Successfully created static directory: {static_dir}")
        except Exception as e_mkdir:
            print(f"[PLOTTER] ERROR: Could not create static directory '{static_dir}': {e_mkdir}")
            traceback.print_exc()
            return # Cannot proceed if static dir cannot be created
    else:
        print(f"[PLOTTER] Static directory '{static_dir}' already exists.")

    # Basenames are taken from the arguments, full paths are constructed
    base_output_filename1 = os.path.basename(output_filename1)
    base_output_filename2 = os.path.basename(output_filename2)

    output_path1 = os.path.join(static_dir, base_output_filename1)
    output_path2 = os.path.join(static_dir, base_output_filename2)

    print(f"[PLOTTER] Plot 1 will be saved to (absolute path): {os.path.abspath(output_path1)}")
    print(f"[PLOTTER] Plot 2 will be saved to (absolute path): {os.path.abspath(output_path2)}")

    actual_sim_length = len(T_IT_sim)
    if actual_sim_length == 0:
        print("[PLOTTER] CRITICAL WARNING: Simulation data (e.g., T_IT_sim) has length 0. Plotting may fail or produce blank images.")
        # Decide how to handle: return, create blank images, or let it try
        # return

    # Ensure time_points matches the actual length of simulation data
    # Default simulation_time_seconds to 0 if not found in p, for robustness
    sim_time_sec = p.get('simulation_time_seconds', 0)
    if actual_sim_length > 0:
        time_points = np.linspace(0, sim_time_sec, actual_sim_length)
    else: # actual_sim_length is 0
        time_points = np.array([]) # Create an empty array

    time_minutes_plotting = time_points / 60.0 # Ensure float division

    print(f"[PLOTTER] actual_sim_length: {actual_sim_length}")
    print(f"[PLOTTER] simulation_time_seconds from p: {sim_time_sec}")
    print(f"[PLOTTER] time_points array length: {len(time_points)}, content (first 5): {time_points[:5]}")
    print(f"[PLOTTER] time_minutes_plotting array length: {len(time_minutes_plotting)}, content (first 5): {time_minutes_plotting[:5]}")

    try:
        # Figure 1: Temperatures and HVAC Power
        print("[PLOTTER] Attempting to create Figure 1...")
        fig1 = plt.figure(figsize=(14, 12))
        print("[PLOTTER] Figure 1 object created.")

        ax1_1 = fig1.add_subplot(2, 1, 1)
        print("[PLOTTER] Added subplot 1 to Figure 1.")

        # Plotting with checks for data length
        if actual_sim_length > 0:
            ax1_1.plot(time_minutes_plotting, T_IT_sim[:actual_sim_length], label='IT Equipment ($T_{IT}$) (°C)')
            ax1_1.plot(time_minutes_plotting, T_Rack_sim[:actual_sim_length], label='Rack Air($T_{Rack}$) (°C)')
            ax1_1.plot(time_minutes_plotting, T_cAisle_sim[:actual_sim_length], label='Cold Aisle Air ($T_{cAisle}$) (°C)', linestyle=':')
            ax1_1.plot(time_minutes_plotting, T_Air_in_sim[:actual_sim_length], label='HVAC Supply Air ($T_{Air,in}$) (°C)', linestyle='--')
            ax1_1.plot(time_minutes_plotting, T_hAisle_sim[:actual_sim_length], label='Hot Aisle Air ($T_{hAisle}$) (°C)', linestyle='-.')
        else:
            print("[PLOTTER] Warning: No data to plot for Figure 1, Subplot 1 due to actual_sim_length=0.")

        ax1_1.set_xlabel('Time (minutes)')
        ax1_1.set_ylabel('Temperature (°C)')
        title_sim_mode = p.get("simulation_mode", "N/A").replace("_", " ").title()
        title_T_target = p.get("T_target_Air_in_Celsius", "N/A")
        title_T_cAisle_limit = p.get("T_cAisle_upper_limit_Celsius", "N/A")
        ax1_1.set_title(f'Data Centre Cooling ({title_sim_mode} Mode), $T_{{Air,in}}$ Target: {title_T_target}°C, Max $T_{{cAisle}}$: {title_T_cAisle_limit}°C')
        ax1_1.legend()
        ax1_1.grid(True)

        # Y-axis limits for subplot 1
        min_T_target = p.get('T_target_Air_in_Celsius', 18) # Default to 18 if not found
        min_temps_plot1 = [min_T_target - 5]
        if actual_sim_length > 0 and len(T_IT_sim) > 0: # Check len of specific arrays too
             min_temps_plot1.extend([np.min(T_IT_sim)-2, np.min(T_Rack_sim)-2, np.min(T_cAisle_sim)-2, np.min(T_Air_in_sim)-2])
        min_y_plot1 = min(min_temps_plot1) if min_temps_plot1 else 0

        max_temps_plot1 = [min_T_target + 10]
        if actual_sim_length > 0 and len(T_IT_sim) > 0:
            max_temps_plot1.extend([np.max(T_IT_sim)+5, np.max(T_Rack_sim)+5, np.max(T_cAisle_sim)+5, np.max(T_Air_in_sim)+5])
        if p.get('simulation_mode') == "warm_up" and p.get('T_cAisle_upper_limit_Celsius') is not None:
            max_temps_plot1.append(p.get('T_cAisle_upper_limit_Celsius') + 5)
        max_y_plot1 = max(max_temps_plot1) if max_temps_plot1 else 30
        ax1_1.set_ylim(bottom=min_y_plot1, top=max_y_plot1)
        print("[PLOTTER] Subplot 1 for Figure 1 configured.")

        ax1_2 = fig1.add_subplot(2, 1, 2)
        print("[PLOTTER] Added subplot 2 to Figure 1.")
        if actual_sim_length > 0:
            ax1_2.plot(time_minutes_plotting, P_HVAC_sim[:actual_sim_length], label='HVAC Power ($P_{HVAC}$) (Watts)', color='purple')
            ax1_2.plot(time_minutes_plotting, P_HVAC_steady_state[:actual_sim_length], label='HVAC Steady State Power (Watts)', linestyle='--', color='red')
        else:
            print("[PLOTTER] Warning: No data to plot for Figure 1, Subplot 2 due to actual_sim_length=0.")
        ax1_2.set_xlabel('Time (minutes)')
        ax1_2.set_ylabel('HVAC Power (Watts)', color='purple')
        ax1_2.tick_params(axis='y', labelcolor='purple')
        ax1_2.grid(True)
        p_hvac_max = p.get('P_HVAC_max_watts', 10000.0)
        ax1_2.set_ylim(bottom=-0.05 * p_hvac_max, top=p_hvac_max * 1.1 if p_hvac_max > 0 else 100)
        handles_ax1_2, labels_ax1_2 = ax1_2.get_legend_handles_labels()
        ax1_2.set_title('Dynamic HVAC Power')
        ax1_2.legend(handles_ax1_2, labels_ax1_2, loc='upper right')
        print("[PLOTTER] Subplot 2 for Figure 1 configured.")

        fig1.tight_layout()
        print(f"[PLOTTER] Attempting to save Figure 1 to: {output_path1}")
        fig1.savefig(output_path1)
        plt.close(fig1)
        print(f"[PLOTTER] Successfully saved Figure 1 to {output_path1} (verified by checking os.path.exists: {os.path.exists(output_path1)})")

        # Figure 2: TES Energy and Power
        print("[PLOTTER] Attempting to create Figure 2...")
        fig2, ax2_1 = plt.subplots(figsize=(10,6))
        print("[PLOTTER] Figure 2 object created.")
        if actual_sim_length > 0:
            ax2_1.plot(time_minutes_plotting, E_TES_sim[:actual_sim_length], label='TES Energy (kWh)', color='orange')
        else:
            print("[PLOTTER] Warning: No data for E_TES_sim plot due to actual_sim_length=0.")
        ax2_1.set_xlabel('Time (minutes)')
        ax2_1.set_ylabel('TES Energy (kWh)', color='orange')
        ax2_1.tick_params(axis='y', labelcolor='orange')
        ax2_1.grid(True)

        ax2_2 = ax2_1.twinx()
        if actual_sim_length > 0:
            ax2_2.plot(time_minutes_plotting, P_TES_sim[:actual_sim_length], label='TES Power (W)', linestyle='--', color='green')
        else:
            print("[PLOTTER] Warning: No data for P_TES_sim plot due to actual_sim_length=0.")
        ax2_2.set_ylabel('TES Power (W)', color='green')
        ax2_2.tick_params(axis='y', labelcolor='green')

        lines_1, labels_1 = ax2_1.get_legend_handles_labels()
        lines_2, labels_2 = ax2_2.get_legend_handles_labels()
        ax2_1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        fig2.suptitle('TES Energy and Power Over Time')
        fig2.tight_layout()
        print(f"[PLOTTER] Attempting to save Figure 2 to: {output_path2}")
        fig2.savefig(output_path2)
        plt.close(fig2)
        print(f"[PLOTTER] Successfully saved Figure 2 to {output_path2} (verified by checking os.path.exists: {os.path.exists(output_path2)})")

    except Exception as e_plot:
        print(f"[PLOTTER] MAJOR ERROR during plotting or saving figures: {e_plot}")
        traceback.print_exc()

    print(f"[PLOTTER] --- Finished plot_simulation_results ---")

def main(init_data=False):
    p = setup_simulation_parameters()
    if init_data:
        p_init = pd.read_csv('static/optimisation_results.csv').iloc[0].to_dict()  # Load initial conditions from CSV
    else:
        p_init = None
    T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, E_TES_sim, P_TES_sim = run_simulation(p, p_init)
    P_HVAC_steady_state = calculate_flexibility(P_HVAC_sim, p, p['simulation_mode'])
    plot_simulation_results(T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim,
                            P_HVAC_sim, P_HVAC_steady_state, E_TES_sim, P_TES_sim, p)

if __name__ == "__main__":
    main(True)
    
