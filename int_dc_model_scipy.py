import pathlib
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import matplotlib.pyplot as plt

# --- Enabled Imports for your Supporting Modules ---
# Ensure these modules are in your Python path
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles


# --- Constants and Configuration ---------------------------------------------
_DBGDIR = pathlib.Path("nlp_debug")
_DBGDIR.mkdir(exist_ok=True)

# Model configuration switches
CYCLE_TES_ENERGY = True

# --- Parameter Management Class ----------------------------------------------
class ModelParameters:
    """A class to hold and manage all model parameters and derived constants."""
    def __init__(self, simulation_minutes=360, dt_seconds=900, extended_horizon_minutes=180):
        # --- Time Horizon ---
        self.simulation_minutes = simulation_minutes
        self.dt_seconds = dt_seconds
        self.extended_horizon_minutes = extended_horizon_minutes
        self.dt_hours = self.dt_seconds / 3600.0
        self.sim_minutes_ext = self.simulation_minutes + self.extended_horizon_minutes
        self.num_steps_extended = int(self.sim_minutes_ext * 60 / self.dt_seconds)

        # Time slots (0-based indexing for NumPy compatibility)
        self.T_SLOTS_IDX = range(int(self.simulation_minutes * 60 / self.dt_seconds))
        self.TEXT_SLOTS_IDX = range(self.num_steps_extended)
        self.K_TRANCHES = range(1, 5)

        # --- IT Equipment ---
        self.idle_power_kw = 166.7
        self.max_power_kw = 1000.0
        self.max_cpu_usage = 1.0
        self.tranche_max_delay_steps = {k: int(d * 60 / self.dt_seconds) for k, d in {1: 30, 2: 60, 3: 120, 4: 180}.items()}
        self.nominal_overhead_factor = 0.1

        # --- UPS / Battery Storage ---
        self.eta_ch = 0.82
        self.eta_disch = 0.92
        self.e_nom_kwh = 600.0
        self.soc_min = 0.5
        self.soc_max = 1.0
        self.e_start_kwh = 600.0
        self.p_max_ch_kw = 270.0
        self.p_max_disch_kw = 270.0
        self.p_min_ch_kw = 40.0
        self.p_min_disch_kw = 100.0
        self.e_min_kwh = self.soc_min * self.e_nom_kwh
        self.e_max_kwh = self.soc_max * self.e_nom_kwh

        # --- Cooling System (from external file) ---
        cooling_params = setup_simulation_parameters("cool_down")
        self.__dict__.update(cooling_params)
        self.TES_capacity_kWh = self.TES_kwh_cap

# --- SciPy NLP Model Implementation ------------------------------------------

class VariableManager:
    """Helper class to manage the flat variable vector for SciPy."""
    def __init__(self, params: ModelParameters):
        self.params = params
        self.num_steps = params.num_steps_extended
        self.var_map = {}
        self.total_vars = 0

        # Define all variables and their sizes
        self.var_sizes = {
            'total_cpu': self.num_steps,
            'p_grid_it': self.num_steps,
            'p_grid_od': self.num_steps,
            'p_it_total': self.num_steps,
            'p_ups_ch': self.num_steps,
            'p_ups_disch': self.num_steps,
            'e_ups': self.num_steps,
            'z_ch': self.num_steps, # Relaxed binary
            'z_disch': self.num_steps, # Relaxed binary
            't_it': self.num_steps, 't_rack': self.num_steps, 't_cold_aisle': self.num_steps,
            't_hot_aisle': self.num_steps, 'e_tes': self.num_steps,
            'p_chiller_hvac': self.num_steps, 'p_chiller_tes': self.num_steps
        }
        # Calculate indices for job scheduling variables
        self.ut_ks_indices = [(t, k, s) for t in self.params.T_SLOTS_IDX for k in self.params.K_TRANCHES
                              for s in self.params.TEXT_SLOTS_IDX if s >= t and s <= t + self.params.tranche_max_delay_steps[k]]
        self.var_sizes['ut_ks'] = len(self.ut_ks_indices)


        # Create a map to locate each variable group in the flat vector
        start_idx = 0
        for name, size in self.var_sizes.items():
            self.var_map[name] = slice(start_idx, start_idx + size)
            start_idx += size
        self.total_vars = start_idx

    def unpack(self, x: np.ndarray) -> dict:
        """Unpacks the flat vector x into a dictionary of named variables."""
        return {name: x[sl] for name, sl in self.var_map.items()}

def build_and_solve_nlp(params: ModelParameters, data: dict):
    """Builds and solves the non-linear optimization problem using SciPy."""
    vm = VariableManager(params)

    # --- 1. Objective Function ---
    def objective_function(x, params, data):
        vars = vm.unpack(x)
        p_grid_it = vars['p_grid_it']
        p_chiller_hvac = vars['p_chiller_hvac']
        p_chiller_tes = vars['p_chiller_tes']
        p_grid_od = vars['p_grid_od']
        p_ups_ch = vars['p_ups_ch']
        prices = data['electricity_price'] # 0-indexed

        total_cost = np.sum(
            params.dt_hours * (
                p_grid_it +
                (p_chiller_hvac / 1000.0) +
                (p_chiller_tes / 1000.0) +
                p_grid_od +
                p_ups_ch
            ) * (prices / 1000.0)
        )
        return total_cost

    # --- 2. Bounds ---
    low_bounds = np.zeros(vm.total_vars)
    up_bounds = np.full(vm.total_vars, np.inf)

    # Set specific bounds based on variable definitions
    vars_unpacked = vm.unpack(low_bounds) # Use as a template
    vars_unpacked['total_cpu'][:] = 0.0
    vars_unpacked['e_ups'][:] = params.e_min_kwh
    vars_unpacked['z_ch'][:] = 0.0 # Relaxed binary
    vars_unpacked['z_disch'][:] = 0.0 # Relaxed binary
    vars_unpacked['t_it'][:] = 14.0
    vars_unpacked['t_rack'][:] = 14.0
    vars_unpacked['t_cold_aisle'][:] = 14.0
    vars_unpacked['t_hot_aisle'][:] = 14.0
    vars_unpacked['e_tes'][:] = params.E_TES_min_kWh

    vars_unpacked_up = vm.unpack(up_bounds) # Template for upper bounds
    vars_unpacked_up['total_cpu'][:] = params.max_cpu_usage
    vars_unpacked_up['e_ups'][:] = params.e_max_kwh
    vars_unpacked_up['z_ch'][:] = 1.0 # Relaxed binary
    vars_unpacked_up['z_disch'][:] = 1.0 # Relaxed binary
    vars_unpacked_up['t_it'][:] = 75.0
    vars_unpacked_up['t_rack'][:] = 70.0
    vars_unpacked_up['t_cold_aisle'][:] = params.T_cAisle_upper_limit_Celsius
    vars_unpacked_up['t_hot_aisle'][:] = 80.0
    vars_unpacked_up['e_tes'][:] = params.TES_capacity_kWh

    bounds = Bounds(low_bounds, up_bounds)

    # --- 3. Constraints ---
    constraints = []

    # Helper function to get a variable from the flat vector x
    def get_var(x, name):
        return x[vm.var_map[name]]

    # A. Job Scheduling and CPU Usage (Equality)
    def job_constraints_eq(x):
        ut_ks = get_var(x, 'ut_ks')
        total_cpu = get_var(x, 'total_cpu')

        # Job Completion
        job_completion_vals = []
        for t in params.T_SLOTS_IDX:
            for k in params.K_TRANCHES:
                relevant_indices = [i for i, (t_idx, k_val, _) in enumerate(vm.ut_ks_indices) if t_idx == t and k_val == k]
                sum_val = np.sum(ut_ks[relevant_indices]) * params.dt_hours
                target_val = data['Rt'][t] * data['shiftabilityProfile'].get((t + 1, k), 0)
                job_completion_vals.append(sum_val - target_val)

        # Total CPU Usage
        cpu_usage_vals = []
        for s in params.TEXT_SLOTS_IDX:
            relevant_indices = [i for i, (_, _, s_idx) in enumerate(vm.ut_ks_indices) if s_idx == s]
            flexible_usage = np.sum(ut_ks[relevant_indices])
            cpu_def = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage - total_cpu[s]
            cpu_usage_vals.append(cpu_def)

        return np.concatenate([job_completion_vals, cpu_usage_vals])

    constraints.append(NonlinearConstraint(job_constraints_eq, 0, 0))

    # B. IT Power Definition (Non-Linear Equality)
    def it_power_constraint_eq(x):
        p_it_total = get_var(x, 'p_it_total')
        total_cpu = get_var(x, 'total_cpu')
        # THIS IS THE KEY NON-LINEAR CONSTRAINT
        power_calc = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (total_cpu ** 1.32)
        return p_it_total - power_calc

    constraints.append(NonlinearConstraint(it_power_constraint_eq, 0, 0))

    # C. Power Balance and UPS Constraints
    def power_and_ups_constraints(x):
        vars = vm.unpack(x)
        p_it_total, p_grid_it, p_grid_od, p_ups_disch = vars['p_it_total'], vars['p_grid_it'], vars['p_grid_od'], vars['p_ups_disch']
        p_ups_ch, e_ups, z_ch, z_disch = vars['p_ups_ch'], vars['e_ups'], vars['z_ch'], vars['z_disch']

        it_balance = p_grid_it + p_ups_disch - p_it_total
        overhead_balance = p_it_total * params.nominal_overhead_factor - p_grid_od
        e_ups_prev = np.roll(e_ups, 1)
        e_ups_prev[0] = params.e_start_kwh
        ups_balance = e_ups_prev + (params.eta_ch * p_ups_ch - p_ups_disch / params.eta_disch) * params.dt_hours - e_ups
        ups_balance[-1] = e_ups[-1] - params.e_start_kwh # Enforce final state
        ups_max_ch = z_ch * params.p_max_ch_kw - p_ups_ch
        ups_min_ch = p_ups_ch - z_ch * params.p_min_ch_kw
        ups_max_disch = z_disch * params.p_max_disch_kw - p_ups_disch
        ups_min_disch = p_ups_disch - z_disch * params.p_min_disch_kw
        charge_or_discharge = 1 - (z_ch + z_disch)

        eq_constraints = np.concatenate([it_balance, overhead_balance, ups_balance])
        ineq_constraints = np.concatenate([ups_max_ch, ups_min_ch, ups_max_disch, ups_min_disch, charge_or_discharge])
        return eq_constraints, ineq_constraints

    def eq_con_func(x): return power_and_ups_constraints(x)[0]
    def ineq_con_func(x): return power_and_ups_constraints(x)[1]

    constraints.append(NonlinearConstraint(eq_con_func, 0, 0))
    constraints.append(NonlinearConstraint(ineq_con_func, 0, np.inf))

    # D. Cooling Constraints (Simplified)
    def cooling_constraints_eq(x):
        p_it_total = get_var(x, 'p_it_total')
        p_chiller_hvac = get_var(x, 'p_chiller_hvac')
        cooling_power_demand_kw = p_it_total / params.COP_HVAC
        return (p_chiller_hvac / 1000.0) - cooling_power_demand_kw

    print("WARNING: Using simplified cooling model for NLP conversion.")
    constraints.append(NonlinearConstraint(cooling_constraints_eq, 0, 0))


    # --- 4. Initial Guess (x0) ---
    x0 = np.zeros(vm.total_vars)
    x0[vm.var_map['p_it_total']] = data['Pt_IT_nom_TEXT']
    x0[vm.var_map['p_grid_it']] = data['Pt_IT_nom_TEXT']
    x0[vm.var_map['e_ups']] = params.e_start_kwh
    x0[vm.var_map['e_tes']] = params.TES_initial_charge_kWh
    x0[vm.var_map['t_it']] = params.T_IT_initial_Celsius
    x0[vm.var_map['t_rack']] = params.T_Rack_initial_Celsius
    x0[vm.var_map['t_cold_aisle']] = params.T_cAisle_initial
    x0[vm.var_map['t_hot_aisle']] = params.T_hAisle_initial

    # --- 5. Solve ---
    print("4. Starting NLP solver (SciPy SLSQP)...")
    result = minimize(
        objective_function, x0, args=(params, data), method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'disp': True, 'ftol': 1e-4, 'maxiter': 200}
    )

    return result, vm

# --- Data Loading and Processing ---------------------------------------------
def load_and_prepare_data(params: ModelParameters):
    """Loads external data and preprocesses it for the model.
    This version is robust to both upsampling and downsampling."""
    BASE_DT_SECONDS = 900  # Assumed base interval of the source data in seconds.

    # Call the actual user module to get data at its native resolution.
    inflexible, base_flex, base_flex_t, shiftability, _ = get_load_and_price_profiles(params.TEXT_SLOTS_IDX, params.T_SLOTS_IDX)

    num_original_t_slots = len(base_flex_t)
    resample_factor, agg_factor = None, None

    # Determine if we are upsampling or downsampling
    if params.dt_seconds < BASE_DT_SECONDS:
        # --- UPSAMPLING ---
        if BASE_DT_SECONDS % params.dt_seconds != 0:
            raise ValueError("BASE_DT_SECONDS must be divisible by dt_seconds for upsampling.")
        resample_factor = BASE_DT_SECONDS // params.dt_seconds
        inflexible_resampled = np.repeat(inflexible, resample_factor)
        base_flex_resampled = np.repeat(base_flex, resample_factor)
        base_flex_t_resampled = np.repeat(base_flex_t, resample_factor)
        shiftability_processed = resample_shiftability_profile(shiftability, resample_factor, num_original_t_slots)
        print(f"Data resampled with factor: {resample_factor}")

    elif params.dt_seconds >= BASE_DT_SECONDS:
        # --- DOWNSAMPLING ---
        if params.dt_seconds % BASE_DT_SECONDS != 0:
            raise ValueError("dt_seconds must be a multiple of BASE_DT_SECONDS for downsampling.")
        agg_factor = params.dt_seconds // BASE_DT_SECONDS
        inflexible_resampled = pd.Series(inflexible).rolling(window=agg_factor).mean().iloc[agg_factor-1::agg_factor].to_numpy()
        base_flex_resampled = pd.Series(base_flex).rolling(window=agg_factor).mean().iloc[agg_factor-1::agg_factor].to_numpy()
        base_flex_t_resampled = pd.Series(base_flex_t).rolling(window=agg_factor).mean().iloc[agg_factor-1::agg_factor].to_numpy()
        shiftability_processed = aggregate_shiftability_profile(shiftability, agg_factor, num_original_t_slots)
        print(f"Data aggregated with factor: {agg_factor}")

    data = {
        'inflexibleLoadProfile_TEXT': np.resize(inflexible_resampled, params.num_steps_extended),
        'shiftabilityProfile': shiftability_processed
    }
    baseFlexibleLoadProfile_T = np.resize(base_flex_t_resampled, len(params.T_SLOTS_IDX))
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours
    baseFlexibleLoadProfile_TEXT = np.resize(base_flex_resampled, params.num_steps_extended)
    data['Pt_IT_nom_TEXT'] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (data['inflexibleLoadProfile_TEXT'] + baseFlexibleLoadProfile_TEXT)
    data['electricity_price'] = generate_tariff(params.num_steps_extended, params.dt_seconds)

    return data

def resample_shiftability_profile(shiftability_profile, repeats, num_original_t_slots):
    """Upsamples the shiftability profile by repeating entries."""
    extended_data = {}
    for i in range(num_original_t_slots):
        for r in range(repeats):
            new_t_slot_idx = i * repeats + r
            for k in range(1, 5):
                extended_data[(new_t_slot_idx + 1, k)] = shiftability_profile.get((i + 1, k), 0)
    return extended_data

def aggregate_shiftability_profile(shiftability_profile, agg_factor, num_original_t_slots):
    """Downsamples the shiftability profile by taking the first value in each window."""
    aggregated_data = {}
    num_new_t_slots = num_original_t_slots // agg_factor
    for i in range(num_new_t_slots):
        original_t_slot = i * agg_factor + 1
        for k in range(1, 5):
            aggregated_data[(i + 1, k)] = shiftability_profile.get((original_t_slot, k), 0)
    return aggregated_data

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    """Generates a flattened electricity price tariff."""
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    if dt_seconds == 0: return np.array([])
    steps_per_hour = 3600 // dt_seconds
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, steps_per_hour)
    return price_per_step[:num_steps]


# --- Post-Processing, Charting, and Output ------------------------------------
def post_process_results(result: dict, vm: VariableManager, params: ModelParameters, data: dict):
    """ Extracts results from a solved SciPy model into a DataFrame. """
    if not result.success:
        print("Warning: Solver did not converge successfully.")
        print(f"Message: {result.message}")

    solved_vars = vm.unpack(result.x)

    # Calculate nominal cost for comparison
    nominal_cooling_power = (data["Pt_IT_nom_TEXT"] / params.COP_HVAC)
    total_nominal_power_profile = data["Pt_IT_nom_TEXT"] * (1 + params.nominal_overhead_factor) + nominal_cooling_power
    nominal_cost = sum(
        params.dt_hours * total_nominal_power_profile[t] * (data['electricity_price'][t] / 1000.0)
        for t in params.TEXT_SLOTS_IDX
    )

    results = {
        'Time_Slot_EXT': list(params.TEXT_SLOTS_IDX),
        'Optimized_Cost': result.fun,
        'Nominal_Cost': nominal_cost,
        'P_IT_Total_kW': solved_vars['p_it_total'],
        'P_Grid_IT_kW': solved_vars['p_grid_it'],
        'P_Grid_Cooling_kW': (solved_vars['p_chiller_hvac'] + solved_vars['p_chiller_tes']) / 1000.0,
        'P_Grid_Other_kW': solved_vars['p_grid_od'],
        'P_UPS_Charge_kW': solved_vars['p_ups_ch'],
        'P_UPS_Discharge_kW': solved_vars['p_ups_disch'],
        'E_UPS_kWh': solved_vars['e_ups'],
        'T_IT_Celsius': solved_vars['t_it'],
        'T_Rack_Celsius': solved_vars['t_rack'],
        'T_ColdAisle_Celsius': solved_vars['t_cold_aisle'],
        'T_HotAisle_Celsius': solved_vars['t_hot_aisle'],
        'E_TES_kWh': solved_vars['e_tes'],
        'P_IT_Nominal': data['Pt_IT_nom_TEXT'],
        'Price_GBP_per_MWh': data['electricity_price']
    }
    return pd.DataFrame(results)

def print_summary(results_df: pd.DataFrame):
    """Prints a formatted summary of the optimization results."""
    optimized_cost = results_df['Optimized_Cost'].iloc[0]
    nominal_cost = results_df['Nominal_Cost'].iloc[0]
    cost_saving_abs = nominal_cost - optimized_cost
    cost_saving_rel = (cost_saving_abs / nominal_cost) * 100 if nominal_cost else 0

    print("\n" + "="*50)
    print("--- Non-Linear Optimization Results ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print(f"Baseline (Nominal) Cost: {nominal_cost:,.2f} GBP")
    print("\n--- Savings ---")
    print(f"Absolute Cost Saving: {cost_saving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {cost_saving_rel:.2f} %")
    print("="*50 + "\n")

def create_and_save_charts(params, df: pd.DataFrame):
    """Generates and saves charts based on the results DataFrame."""
    print("6. Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    time_slots_ext = df['Time_Slot_EXT']

    # Figure 1: Power Consumption and Price
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(time_slots_ext, df['P_IT_Nominal'], label='Nominal IT Power', linestyle='--', color='gray', alpha=0.8)
    ax1.plot(time_slots_ext, df['P_IT_Total_kW'], label='Optimized IT Power (Non-Linear)', color='crimson')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Optimized vs. Nominal IT Power Consumption')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(time_slots_ext, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue')
    ax2.set_ylabel('Price (£/MWh)')
    ax2.set_xlabel(f'Time Slot ({params.dt_seconds/60} min intervals)')
    ax2.legend()
    ax2.grid(True)
    fig1.tight_layout()
    fig1.savefig('nlp_power_consumption.png')

    # Figure 2: UPS and TES Energy Storage
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax3.plot(time_slots_ext, df['E_UPS_kWh'], label='UPS Energy (E_UPS)', color='darkorange')
    ax3.axhline(y=params.e_max_kwh, color='r', linestyle='--', label='UPS Max kWh')
    ax3.axhline(y=params.e_min_kwh, color='r', linestyle='--', label='UPS Min kWh')
    ax3.set_ylabel('Energy (kWh)')
    ax3.set_title('Energy Storage Levels')
    ax3.legend()
    ax4.plot(time_slots_ext, df['E_TES_kWh'], label='Thermal Storage (E_TES)', color='deepskyblue')
    ax4.set_ylabel('Energy (kWh)')
    ax4.set_xlabel(f'Time Slot ({params.dt_seconds/60} min intervals)')
    ax4.legend()
    fig2.tight_layout()
    fig2.savefig('nlp_energy_storage.png')

    print("Charts saved as nlp_power_consumption.png and nlp_energy_storage.png")
    plt.show()

# --- Main Orchestrator -------------------------------------------------------
def run_full_optimisation():
    """Main function to orchestrate the NLP optimization process."""
    print("1. Setting up model parameters...")
    # You can adjust dt_seconds here. The script is robust to changes.
    params = ModelParameters(dt_seconds=900)

    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)

    print(f"3. Building NLP model for {params.num_steps_extended} time steps...")
    result, var_manager = build_and_solve_nlp(params, input_data)

    if result.success:
        print("5. Solver finished. Post-processing...")
        results_df = post_process_results(result, var_manager, params, input_data)
        print_summary(results_df)
        create_and_save_charts(params, results_df)

        output_path = "NLP_Results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Results have been successfully exported to '{output_path}'")
        return results_df
    else:
        print(f"*** Solver FAILED. Status: {result.status}, Message: {result.message} ***")
        return None

if __name__ == '__main__':
    run_full_optimisation()