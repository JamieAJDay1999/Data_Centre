import datetime
import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles

# --- Constants and Configuration ---------------------------------------------
_DBGDIR = pathlib.Path("lp_debug")
_DBGDIR.mkdir(exist_ok=True)

# --- Parameter Management Class (unchanged) -----------------------------------
class ModelParameters:
    """A class to hold and manage all model parameters and derived constants."""
    def __init__(self, simulation_minutes=1440, dt_seconds=60, extended_horizon_minutes=180):
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
        self.tranche_max_delay = {1: 30, 2: 60, 3: 120, 4: 180}
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


# --- Data Loading and Processing (unchanged) ---------------------------------
def load_and_prepare_data(params: ModelParameters):
    """Loads external data and preprocesses it for the model."""
    RESAMPLE_FACTOR = int(900 / params.dt_seconds)
    print(f"Resampling factor: {RESAMPLE_FACTOR} (dt_seconds: {params.dt_seconds})")
    inflexible, base_flex, base_flex_t, shiftability, _ = get_load_and_price_profiles(params.TEXT_SLOTS, params.T_SLOTS)
    
    data = {
        'inflexibleLoadProfile_TEXT': np.insert(np.array(np.repeat(inflexible, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0),
        'shiftabilityProfile': resample_shiftability_profile(shiftability, RESAMPLE_FACTOR)
    }
    baseFlexibleLoadProfile_T = np.insert(np.array(np.repeat(base_flex_t, RESAMPLE_FACTOR))[:params.simulation_minutes], 0, 0)
    data['Rt'] = baseFlexibleLoadProfile_T * params.dt_hours
    baseFlexibleLoadProfile_TEXT = np.insert(np.array(np.repeat(base_flex, RESAMPLE_FACTOR))[:params.sim_minutes_ext], 0, 0)
    data['Pt_IT_nom_TEXT'] = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * (data['inflexibleLoadProfile_TEXT'] + baseFlexibleLoadProfile_TEXT)
    data['electricity_price'] = generate_tariff(params.num_steps_extended, params.dt_seconds)
    return data

def resample_shiftability_profile(shiftability_profile, repeats):
    """Resamples the shiftability profile."""
    extended_data = {}
    counter = 1
    for i in range(1, 97):
        for _ in range(repeats):
            for j in range(1, 5):
                extended_data[(counter, j)] = shiftability_profile.get((i, j), 0)
            counter += 1
    return extended_data

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    """Generates a flattened electricity price tariff."""
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)


# --- NEW: Nominal Case Calculation Function ----------------------------------
def calculate_nominal_case(params: ModelParameters, data: dict):
    """
    Calculates the total cost and system state for the nominal case where the
    data centre operates in a steady state without using energy storage.
    """
    results = []
    total_cost = 0.0

    # For the equilibrium case, temperatures are held constant at initial values.
    t_it_eq = params.T_IT_initial_Celsius
    t_rack_eq = params.T_Rack_initial_Celsius
    t_cold_eq = params.T_cAisle_initial
    t_hot_eq = params.T_hAisle_initial
    e_tes_eq = params.TES_initial_charge_kWh
    e_ups_eq = params.e_start_kwh

    for s in params.TEXT_SLOTS:
        # 1. IT power is fixed to the nominal profile.
        p_it_total_kw = data['Pt_IT_nom_TEXT'][s]

        # 2. Overhead power is a fixed factor of IT power.
        p_grid_other_kw = p_it_total_kw * params.nominal_overhead_factor

        # 3. UPS and TES are not used.
        p_ups_charge_kw = 0.0
        p_ups_discharge_kw = 0.0
        q_charge_tes_watts = 0.0
        q_discharge_tes_watts = 0.0
        p_chiller_tes_kw = 0.0
        
        # 4. Grid power for IT equals total IT power as UPS is offline.
        p_grid_it_kw = p_it_total_kw

        # 5. Cooling system calculations for equilibrium.
        # Heat generated by IT equipment (in Watts).
        q_it_heat_watts = p_it_total_kw * 1000.0

        # Cooling provided by the chiller must exactly match the heat generated.
        q_cool_watts = q_it_heat_watts
        
        # Electrical power required by the chiller to provide this cooling.
        p_chiller_hvac_watts = q_cool_watts / params.COP_HVAC
        p_chiller_hvac_kw = p_chiller_hvac_watts / 1000.0
        
        # Total grid power for cooling comes only from the direct chiller component.
        p_grid_cooling_kw = p_chiller_hvac_kw

        # 6. Calculate total power drawn from the grid for this time step.
        p_total_grid_kw = p_grid_it_kw + p_grid_cooling_kw + p_grid_other_kw

        # 7. Calculate the cost for this time step and add to the total.
        price_gbp_per_mwh = data['electricity_price'][s]
        price_gbp_per_kwh = price_gbp_per_mwh / 1000.0
        step_cost = p_total_grid_kw * params.dt_hours * price_gbp_per_kwh

        # Store results for this time step in a dictionary.
        results.append({
            'Time_Slot_EXT': s,
            'P_IT_Total_kW': p_it_total_kw,
            'P_Grid_IT_kW': p_grid_it_kw,
            'P_Chiller_HVAC_kW': p_chiller_hvac_kw,
            'P_Chiller_TES_kW': p_chiller_tes_kw,
            'P_Grid_Cooling_kW': p_grid_cooling_kw,
            'P_Grid_Other_kW': p_grid_other_kw,
            'P_UPS_Charge_kW': p_ups_charge_kw,
            'P_UPS_Discharge_kW': p_ups_discharge_kw,
            'E_UPS_kWh': e_ups_eq,
            'T_IT_Celsius': t_it_eq,
            'T_Rack_Celsius': t_rack_eq,
            'T_ColdAisle_Celsius': t_cold_eq,
            'T_HotAisle_Celsius': t_hot_eq,
            'E_TES_kWh': e_tes_eq,
            'Q_Cool_Total_Watts': q_cool_watts,
            'Q_Charge_TES_Watts': q_charge_tes_watts,
            'Q_Discharge_TES_Watts': q_discharge_tes_watts,
            'P_IT_Nominal': data['Pt_IT_nom_TEXT'][s],
            'Price_GBP_per_MWh': price_gbp_per_mwh,
            'Nominal_Cost': step_cost
        })
        
    df = pd.DataFrame(results)
    
    # Add total cost columns for compatibility with reporting functions.
    df.to_csv("nominal_case_results.csv", index=False, float_format='%.4f')
    return df


# --- Reporting and Charting (unchanged) --------------------------------------
def print_summary(results_df: pd.DataFrame):
    """Prints a formatted summary of the calculation results."""
    calculated_cost = results_df['Nominal_Cost'].sum()
    
    print("\n" + "="*50)
    print("--- Nominal Case Calculation Results ---")
    print(f"Total Calculated Cost: {calculated_cost:,.2f} GBP")
    print("="*50 + "\n")

def create_and_save_charts(params, df: pd.DataFrame):
    """Generates and saves charts based on the results DataFrame."""
    print("Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')

    time_slots_ext = df['Time_Slot_EXT']

    # --- Figure 1: Power Consumption and Energy Price ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(time_slots_ext, df['P_IT_Total_kW'], label='Nominal IT Power', color='crimson')
    ax1.set_ylabel('Total Power Consumption (kW)')
    ax1.set_title('Nominal Power Consumption')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2.plot(time_slots_ext, df['Price_GBP_per_MWh'], label='Energy Price', color='royalblue', alpha=0.8)
    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Energy Price (GBP/MWh)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig1.tight_layout()
    fig1.savefig('nominal_power_consumption.png')
    
    # --- Figure 2: TES Performance (will show zero activity) ---
    fig2, (ax2_tes, ax3_tes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax2_tes.plot(time_slots_ext, df['E_TES_kWh'], label='TES Energy Level', color='mediumblue')
    ax2_tes.set_ylabel('Energy (kWh)')
    ax2_tes.set_title('Thermal Energy Storage (TES) Performance (Nominal Case)')
    ax2_tes.legend()
    ax2_tes.grid(True)
    ax3_tes.plot(time_slots_ext, df['Q_Charge_TES_Watts'], label='Charge Heat Flow', color='green')
    ax3_tes.plot(time_slots_ext, df['Q_Discharge_TES_Watts'], label='Discharge Heat Flow', color='orange')
    ax3_tes.set_xlabel('Time Slot')
    ax3_tes.set_ylabel('Heat Flow (Watts)')
    ax3_tes.legend()
    ax3_tes.grid(True)
    fig2.tight_layout()
    fig2.savefig('nominal_tes_performance.png')
    
    # --- Figure 3: Data Centre Temperatures (will show constant values) ---
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(time_slots_ext, df['T_IT_Celsius'], label='IT Equipment Temp')
    ax4.plot(time_slots_ext, df['T_Rack_Celsius'], label='Rack Temp')
    ax4.plot(time_slots_ext, df['T_HotAisle_Celsius'], label='Hot Aisle Temp')
    ax4.plot(time_slots_ext, df['T_ColdAisle_Celsius'], label='Cold Aisle Temp')
    ax4.set_xlabel('Time Slot')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Data Centre Temperatures (Equilibrium)')
    ax4.legend()
    ax4.grid(True)
    fig3.tight_layout()
    fig3.savefig('nominal_dc_temperatures.png')

    # --- Figure 4: Cooling System Power Components ---
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    df['Total_Grid_Power_kW'] = df['P_Grid_IT_kW'] + df['P_Grid_Cooling_kW'] + df['P_Grid_Other_kW']
    ax5.stackplot(time_slots_ext,
                  df['P_Grid_IT_kW'], df['P_Grid_Cooling_kW'], df['P_Grid_Other_kW'],
                  labels=['IT Power', 'Cooling Power', 'Overhead Power'],
                  colors=['#FF6347', '#4682B4', '#32CD32'])
    ax5.plot(time_slots_ext, df['Total_Grid_Power_kW'], label='Total Grid Power', color='black', linestyle='--')
    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Nominal Power Breakdown')
    ax5.legend(loc='upper left')
    ax5.grid(True)
    fig4.tight_layout()
    fig4.savefig('nominal_power_breakdown.png')

    plt.show()
    print("Charts saved as .png files.")


# --- Main Orchestrator -------------------------------------------------------
def run_nominal_simulation():
    """Main function to orchestrate the nominal case calculation."""
    print("1. Setting up model parameters...")
    params = ModelParameters()
    
    print("2. Loading and preparing input data...")
    input_data = load_and_prepare_data(params)
    
    print(f"3. Calculating nominal case for {params.num_steps_extended} time steps...")
    results_df = calculate_nominal_case(params, input_data)
    
    if not results_df.empty:
        print("4. Calculation complete. Generating summary and charts...")
        print_summary(results_df)
        create_and_save_charts(params, results_df)
        
        output_path = "AllResults_Nominal_Case.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Results have been successfully exported to '{output_path}'")
        return results_df
    else:
        print(f"*** Calculation failed to produce results. ***")
        return None

if __name__ == '__main__':
    run_nominal_simulation()