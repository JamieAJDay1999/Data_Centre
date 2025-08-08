
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import time
import numpy as np
import traceback # For detailed error logging in /get_animation_data

from parameters import setup_simulation_parameters
from paper1_methodology import run_simulation, calculate_flexibility, plot_simulation_results, simulation_params

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_for_session_management' # Choose a strong secret key

# Define parameter groups and order (must match keys in setup_simulation_parameters)
PARAMETER_GROUPS = {
    "Simulation Control": [
        'simulation_time_minutes', 'dt', 'begin_change'
    ],
    "Target Temperatures": [
        'T_target_Air_in_Celsius', 'T_cAisle_upper_limit_Celsius'
    ],
    "Initial State Temperatures (Mode Dependent Defaults)": [
        'T_Air_in_initial', 'T_cAisle_initial', 'T_cWall_initial_Celsius',
        'T_Rack_initial_Celsius', 'T_IT_initial_Celsius', 'T_hAisle_initial'
    ],
    "IT Equipment": ['P_IT_heat_source', 'C_IT', 'G_conv'],
    "Rack System": ['V_freeSpace_per_rack_in_formula', 'n_racks'],
    "HVAC System": ['m_dot_air', 'kappa', 'COP_HVAC', 'P_HVAC_max_watts', 'P_HVAC_min_watts'],
    "Aisles & Walls": [
        'V_cAisle', 'V_hAisle', 'alpha_cAisle', 'A_cAisle',
        'R_cold_K_per_W', 'C_cWall', 'T_out_Celsius'
    ],
    "Thermal Energy Storage (TES)": [
        'TES_kwh_cap', 'TES_w_discharge_max',
        'TES_discharge_efficiency', 'TES_w_charge_max',
        'TES_charge_efficiency', 'TES_initial_charge'
    ],
    "Physical Constants": ['rho_air', 'c_p_air']
}

# Parameters that are calculated or set internally and shouldn't be directly edited in the form
NON_EDITABLE_PARAMS = [
    'C_Rack', 'C_cAisle', 'C_hAisle', 'G_cold', 'T_out_Kelvin', 'warm_up',
    'num_time_points', 'time_points', 'num_begin_change', 'simulation_time_seconds'
]

def get_param_type(value):
    """Determines the HTML input type based on the parameter's Python type."""
    if isinstance(value, bool):
        return "checkbox"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    return "text" # Default for strings or None

def convert_form_value(key, value_str, default_params_for_type_inference):
    """
    Converts a form string value to its appropriate Python type
    based on the type of the default value for that parameter.
    """
    if key not in default_params_for_type_inference:
        return value_str # Should not happen if form is based on default_params

    default_value = default_params_for_type_inference[key]

    if value_str is None: # Handle cases where value_str might be None (e.g. from JSON)
        return default_value if default_value is not None else None


    # Handle specific case for T_cAisle_upper_limit_Celsius which can be None
    if key == 'T_cAisle_upper_limit_Celsius' and (value_str == "" or value_str.lower() == 'none'):
        return None

    if isinstance(default_value, bool):
        return value_str.lower() in ['true', 'on', '1', True]
    elif isinstance(default_value, int):
        try:
            return int(value_str)
        except (ValueError, TypeError):
            return default_value # Fallback
    elif isinstance(default_value, float):
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return default_value # Fallback
    elif default_value is None and value_str.strip() == "": # For params that can be None
        return None
    return str(value_str) # Default to string if no other type matches

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_mode = session.get('simulation_mode', 'cool_down')

    if request.method == 'POST': # If user changed mode via dropdown
        selected_mode = request.form.get('simulation_mode_select', 'cool_down')
        session['simulation_mode'] = selected_mode
        # Always fetch fresh defaults when mode changes via POST
        default_params = setup_simulation_parameters(mode=selected_mode)
        session['default_params'] = default_params
    else: # GET request or initial load
        # In debug mode, always load fresh parameters from source on a GET request
        # This ensures changes in parameters.py are picked up on browser refresh (GET)
        if app.debug:
            default_params = setup_simulation_parameters(mode=selected_mode)
            session['default_params'] = default_params # Update session for consistency
        else: # In production, rely on session for performance and consistency
            if 'default_params' in session and session.get('simulation_mode') == selected_mode:
                default_params = session['default_params']
            else:
                default_params = setup_simulation_parameters(mode=selected_mode)
                session['simulation_mode'] = selected_mode
                session['default_params'] = default_params

    editable_params = {k: v for k, v in default_params.items() if k not in NON_EDITABLE_PARAMS}

    return render_template('index.html',
                           parameter_groups=PARAMETER_GROUPS,
                           params=editable_params,
                           get_param_type=get_param_type,
                           current_mode=selected_mode,
                           non_editable_keys=NON_EDITABLE_PARAMS)

@app.route('/run_simulation', methods=['POST'])
def run_simulation_route():
    """Handles running the simulation for static plot generation."""
    form_data = request.form
    current_mode = form_data.get('simulation_mode', 'cool_down')

    p = setup_simulation_parameters(mode=current_mode)

    # Override parameters with form data, converting types
    original_defaults_for_mode = setup_simulation_parameters(mode=current_mode)
    for key, value_str in form_data.items():
        if key == 'simulation_mode': continue
        if key in p:
            p[key] = convert_form_value(key, value_str, original_defaults_for_mode)

    # Manually re-calculate derived parameters based on (potentially modified) inputs
    if 'V_freeSpace_per_rack_in_formula' in p and 'n_racks' in p and 'rho_air' in p and 'c_p_air' in p:
         p['C_Rack'] = p['V_freeSpace_per_rack_in_formula'] * p['n_racks'] * p['rho_air'] * p['c_p_air']
    if 'V_cAisle' in p and 'rho_air' in p and 'c_p_air' in p:
        p['C_cAisle'] = p['V_cAisle'] * p['rho_air'] * p['c_p_air']
    if 'V_hAisle' in p and 'rho_air' in p and 'c_p_air' in p:
        p['C_hAisle'] = p['V_hAisle'] * p['rho_air'] * p['c_p_air']
    if 'alpha_cAisle' in p and 'A_cAisle' in p and 'R_cold_K_per_W' in p:
        term1_Gcold_inv = 1 / (p['alpha_cAisle'] * p['A_cAisle']) if (p['alpha_cAisle'] * p['A_cAisle']) > 0 else float('inf')
        p['G_cold'] = 1 / (term1_Gcold_inv + p['R_cold_K_per_W']) if (term1_Gcold_inv + p['R_cold_K_per_W']) > 0 else p['alpha_cAisle'] * p['A_cAisle']
    if 'T_out_Celsius' in p:
        p['T_out_Kelvin'] = p['T_out_Celsius'] + 273.15

    p = simulation_params(p) # Critical: calculates time steps, num_time_points, etc.

    T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, \
    E_TES_sim, P_TES_sim = run_simulation(p.copy())

    P_HVAC_steady_state = calculate_flexibility(P_HVAC_sim, p.copy(), current_mode)

    plot_filename1 = "plot_temperatures_hvac.png"
    plot_filename2 = "plot_tes.png"

    plot_simulation_results(T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim,
                            P_HVAC_sim, P_HVAC_steady_state, E_TES_sim, P_TES_sim, p.copy(),
                            output_filename1=plot_filename1,
                            output_filename2=plot_filename2)

    return redirect(url_for('results_page',
                            plot1=plot_filename1,
                            plot2=plot_filename2,
                            v=time.time())) # Cache buster

@app.route('/results')
def results_page():
    """Displays the static plots generated by the simulation."""
    plot1 = request.args.get('plot1')
    plot2 = request.args.get('plot2')
    cache_version = request.args.get('v')
    return render_template('results.html', plot1=plot1, plot2=plot2, version=cache_version)


@app.route('/live_simulation_page')
def live_simulation_page_route():
    """Serves the HTML page for the live animation."""
    return render_template('live_simulation.html')

@app.route('/get_animation_data', methods=['POST'])
def get_animation_data():
    """
    Runs the simulation with provided parameters (from JSON)
    and returns data formatted for live animation.
    """
    try:
        form_params = request.json # Parameters sent as JSON from frontend
        if not form_params:
            return jsonify({"error": "No parameters provided"}), 400

        current_mode = form_params.get('simulation_mode', 'cool_down')
        p = setup_simulation_parameters(mode=current_mode)

        # Override parameters with form_params, converting types
        # Use original_defaults_for_mode for correct type inference during conversion
        original_defaults_for_mode = setup_simulation_parameters(mode=current_mode)
        for key, value in form_params.items():
            if key == 'simulation_mode': continue
            if key in p:
                 # Value from JSON might already be correct type, but ensure conversion if it's string
                p[key] = convert_form_value(key, str(value) if not isinstance(value, (bool, int, float)) and value is not None else value, original_defaults_for_mode)


        # Manually re-calculate derived parameters based on (potentially modified) inputs
        if 'V_freeSpace_per_rack_in_formula' in p and 'n_racks' in p and 'rho_air' in p and 'c_p_air' in p:
             p['C_Rack'] = p['V_freeSpace_per_rack_in_formula'] * p['n_racks'] * p['rho_air'] * p['c_p_air']
        if 'V_cAisle' in p and 'rho_air' in p and 'c_p_air' in p:
            p['C_cAisle'] = p['V_cAisle'] * p['rho_air'] * p['c_p_air']
        if 'V_hAisle' in p and 'rho_air' in p and 'c_p_air' in p:
            p['C_hAisle'] = p['V_hAisle'] * p['rho_air'] * p['c_p_air']
        if 'alpha_cAisle' in p and 'A_cAisle' in p and 'R_cold_K_per_W' in p:
            term1_Gcold_inv = 1 / (p['alpha_cAisle'] * p['A_cAisle']) if (p['alpha_cAisle'] * p['A_cAisle']) > 0 else float('inf')
            p['G_cold'] = 1 / (term1_Gcold_inv + p['R_cold_K_per_W']) if (term1_Gcold_inv + p['R_cold_K_per_W']) > 0 else p['alpha_cAisle'] * p['A_cAisle']
        if 'T_out_Celsius' in p:
            p['T_out_Kelvin'] = p['T_out_Celsius'] + 273.15

        p = simulation_params(p) # Critical: calculates time steps, num_time_points, etc.

        T_IT_sim, T_Rack_sim, T_cAisle_sim, T_hAisle_sim, T_Air_in_sim, P_HVAC_sim, \
        _, _ = run_simulation(p.copy()) # E_TES_sim, P_TES_sim not directly used in this animation

        # Prepare data for animation (sample per simulated minute)
        time_points_seconds = p.get('time_points', np.array([]))
        num_actual_points = len(T_IT_sim) # Use length of actual simulation results

        animation_output = {
            'time_steps_minutes': [],
            'T1_air_in': [], 'T2_cold_aisle': [], 'T3_rack_air': [],
            'T4_it_equip': [], 'T5_hot_aisle': [], 'P_HVAC': [],
            'P_IT_heat_source': p.get('P_IT_heat_source'),
            'total_duration_minutes': p.get('simulation_time_minutes', 0),
            'min_temp_observed': 10.0, # Default placeholder
            'max_temp_observed': 40.0  # Default placeholder
        }

        last_minute_recorded = -1
        temp_values_for_scaling = []

        if num_actual_points > 0 and len(time_points_seconds) == num_actual_points:
            for i in range(num_actual_points):
                current_sim_minute = int(time_points_seconds[i] / 60)
                if current_sim_minute > last_minute_recorded or i == 0:
                    animation_output['time_steps_minutes'].append(current_sim_minute)
                    animation_output['T1_air_in'].append(round(T_Air_in_sim[i], 1))
                    animation_output['T2_cold_aisle'].append(round(T_cAisle_sim[i], 1))
                    animation_output['T3_rack_air'].append(round(T_Rack_sim[i], 1))
                    animation_output['T4_it_equip'].append(round(T_IT_sim[i], 1))
                    animation_output['T5_hot_aisle'].append(round(T_hAisle_sim[i], 1))
                    animation_output['P_HVAC'].append(round(P_HVAC_sim[i], 0))
                    last_minute_recorded = current_sim_minute

                    temp_values_for_scaling.extend([
                        T_cAisle_sim[i], T_Rack_sim[i], T_IT_sim[i], T_hAisle_sim[i]
                    ])
        elif num_actual_points > 0 and len(time_points_seconds) != num_actual_points:
             print(f"Warning: Mismatch in length of time_points_seconds ({len(time_points_seconds)}) and simulation results ({num_actual_points}). Animation data might be incomplete.")
             # Fallback or error handling if lengths don't match. For now, proceed with caution.
             # This could happen if simulation exits early and time_points isn't truncated.
             # The plotting function handles this by using actual_sim_length for time_points.
             # Here, we might need a similar adjustment or ensure `run_simulation` returns truncated time_points.

        if temp_values_for_scaling:
            animation_output['min_temp_observed'] = round(float(np.min(temp_values_for_scaling)), 1)
            animation_output['max_temp_observed'] = round(float(np.max(temp_values_for_scaling)), 1)
            if animation_output['min_temp_observed'] == animation_output['max_temp_observed']:
                animation_output['min_temp_observed'] -= 1.0 # Ensure a range for color scaling
                animation_output['max_temp_observed'] += 1.0
        
        # Ensure there's always some data to prevent JS errors if simulation yields nothing
        if not animation_output['time_steps_minutes']:
            animation_output['time_steps_minutes'].append(0)
            animation_output['T1_air_in'].append(0)
            animation_output['T2_cold_aisle'].append(0)
            animation_output['T3_rack_air'].append(0)
            animation_output['T4_it_equip'].append(0)
            animation_output['T5_hot_aisle'].append(0)
            animation_output['P_HVAC'].append(0)


        return jsonify(animation_output)

    except Exception as e:
        print(f"Error in /get_animation_data: {e}")
        traceback.print_exc() # Print full traceback to Flask console
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == '__main__':
    # Ensure the static directory exists, relative to app.py's location
    static_folder_path = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_folder_path):
        os.makedirs(static_folder_path)
        print(f"Created static directory at: {static_folder_path}")

    # Ensure the static/images directory exists
    images_folder_path = os.path.join(static_folder_path, 'images')
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path)
        print(f"Created static/images directory at: {images_folder_path}")

    app.run(debug=True)
