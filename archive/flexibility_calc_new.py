# main_flexibility_calculator.py

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Import user-provided dependency modules ---
from parameters_optimisation import setup_simulation_parameters
from it_parameters import get_load_and_price_profiles

# --- Constants and Configuration ---------------------------------------------
_DBGDIR = pathlib.Path("lp_debug")
_DBGDIR.mkdir(exist_ok=True)
CYCLE_TES_ENERGY = True

DATA_DIR = pathlib.Path("static/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Parameter Management Class ----------------------------------------------
class ModelParameters:
    """A class to hold and manage all model parameters and derived constants."""
    def __init__(self, simulation_minutes=1440, dt_seconds=900, extended_horizon_minutes=180):
        # --- Time Horizon ---
        self.simulation_minutes = simulation_minutes
        self.dt_seconds = dt_seconds
        self.extended_horizon_minutes = extended_horizon_minutes
        self.dt_hours = self.dt_seconds / 3600.0
        self.sim_minutes_ext = self.simulation_minutes + self.extended_horizon_minutes
        self.num_steps_extended = int(self.sim_minutes_ext * 60 / self.dt_seconds)

        # Time slots (1-based indexing for PuLP readability)
        self.T_SLOTS = range(1, 1 + int(self.simulation_minutes * 60 / self.dt_seconds))
        self.TEXT_SLOTS = range(1, 1 + self.num_steps_extended)
        self.K_TRANCHES = range(1, 5)

        # --- IT Equipment ---
        self.idle_power_kw = 166.7
        self.max_power_kw = 1000.0
        self.max_cpu_usage = 1.0
        self.tranche_max_delay = {1: 2, 2: 4, 3: 8, 4: 12}
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


# --- Flexibility Calculation Functions ---------------------------------------

def calculate_power_flexibility(baseline_df: pd.DataFrame, params: ModelParameters):
    """
    Calculates the upward and downward power flexibility for each timestep.

    Args:
        baseline_df: DataFrame containing the optimised baseline results.
        params: An instance of the ModelParameters class.

    Returns:
        A DataFrame with added columns for flexibility.
    """
    upward_flex = []
    downward_flex = []

    for _, row in baseline_df.iterrows():
        # --- 1. Calculate Upward Flexibility (Power Decrease) ---
        
        # IT Power Reduction: Assume all flexible load is shifted away.
        p_it_min = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * row['Inflexible_Load_CPU']
        
        # Cooling Power Reduction: Cool the minimum IT load, using TES first.
        q_required_watts = p_it_min * 1000
        
        # Available cooling from TES (Watts), limited by max discharge rate and available energy.
        tes_energy_kwh = row['E_TES_kWh']
        max_q_from_tes_by_energy = (tes_energy_kwh - params.E_TES_min_kWh) / params.dt_hours * 1000 * params.TES_discharge_efficiency
        
        q_from_tes = min(q_required_watts, params.TES_w_discharge_max, max_q_from_tes_by_energy)
        q_from_chiller = max(0, q_required_watts - q_from_tes)
        
        # Power for remaining cooling from chiller (kW)
        p_cooling_min = (q_from_chiller / params.COP_HVAC) / 1000.0

        # Other power reduction
        p_other_min = p_it_min * params.nominal_overhead_factor
        
        # Total minimum power consumption
        p_total_min = p_it_min + p_cooling_min + p_other_min
        
        # Baseline total power for comparison
        p_baseline = row['P_IT_Total_kW'] + row['P_Grid_Cooling_kW'] + row['P_Grid_Other_kW']

        # Final Upward Flexibility (kW)
        up_flex = max(0, p_baseline - p_total_min)
        upward_flex.append(up_flex)

        # --- 2. Calculate Downward Flexibility (Power Increase) ---
        
        # IT Power Increase: Set CPU to 100%
        p_it_max = params.max_power_kw
        
        # Cooling Power Increase: Cool max IT load AND charge TES.
        q_cool_demand_watts = p_it_max * 1000
        p_chiller_for_hvac_kw = (q_cool_demand_watts / params.COP_HVAC) / 1000.0
        
        # Remaining chiller capacity for charging TES
        p_chiller_max_kw = params.P_chiller_max / 1000.0
        p_chiller_for_tes_kw = max(0, p_chiller_max_kw - p_chiller_for_hvac_kw)
        
        # Also limited by TES max charge rate
        max_p_chiller_for_tes_by_rate = (params.TES_w_charge_max / params.COP_HVAC) / 1000.0
        
        # And by available capacity in TES
        tes_headroom_kwh = params.TES_capacity_kWh - tes_energy_kwh
        max_p_chiller_for_tes_by_cap = (tes_headroom_kwh / params.dt_hours) / (params.COP_HVAC * params.TES_charge_efficiency)

        p_chiller_for_tes_kw = min(p_chiller_for_tes_kw, max_p_chiller_for_tes_by_rate, max_p_chiller_for_tes_by_cap)
        
        p_cooling_max = p_chiller_for_hvac_kw + p_chiller_for_tes_kw
        
        # Other power increase
        p_other_max = p_it_max * params.nominal_overhead_factor
        
        # Total maximum power
        p_total_max = p_it_max + p_cooling_max + p_other_max
        
        # Final Downward Flexibility (kW)
        down_flex = max(0, p_total_max - p_baseline)
        downward_flex.append(down_flex)

    # Add new columns to the DataFrame
    baseline_df['Upward_Flexibility_kW'] = upward_flex
    baseline_df['Downward_Flexibility_kW'] = downward_flex
    
    return baseline_df


def plot_flexibility(df: pd.DataFrame):
    """Generates and saves a chart of the calculated power flexibility."""
    print("7. Generating and saving flexibility chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    time_slots = df['Time_Slot_EXT']
    baseline_power = df['P_Grid_IT_kW'] + df['P_Grid_Cooling_kW'] + df['P_Grid_Other_kW'] + df['P_UPS_Charge_kW']

    # Plot baseline         p_baseline_2 =  # Nominal power for comparison
    ax.plot(time_slots, baseline_power, label='Optimised Baseline Power', color='black', linewidth=2)
    ax.scatter(time_slots, df['P_Total_kW'],label='Optimised Baseline Power Total', color='black', s=10, alpha=0.5, linestyle = '-')
    
    # Create the flexibility band
    ax.fill_between(
        time_slots,
        baseline_power - df['Upward_Flexibility_kW'],
        baseline_power + df['Downward_Flexibility_kW'],
        color='skyblue',
        alpha=0.5,
        label='Power Flexibility Range'
    )
    
    ax.plot(time_slots, baseline_power - df['Upward_Flexibility_kW'], linestyle='--', color='green', label='Min Power (Max Upward Flex)')
    ax.plot(time_slots, baseline_power + df['Downward_Flexibility_kW'], linestyle='--', color='red', label='Max Power (Max Downward Flex)')

    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Total Power (kW)', fontsize=12)
    ax.set_title('Data Centre Power Flexibility', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('power_flexibility.png')
    print("✅ Power flexibility chart saved as 'power_flexibility.png'")
    plt.show()

# --- Main Orchestrator -------------------------------------------------------

if __name__ == '__main__':
    # Load parameters and baseline from integrated model's output
    params = ModelParameters()
    baseline_path = DATA_DIR / "optimised_baseline.csv"
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df.rename(columns={
        'Inflexible_Load_CPU_Opt': 'Inflexible_Load_CPU',
        'Flexible_Load_CPU_Opt': 'Flexible_Load_CPU'
    })

    if baseline_df is not None:
        # Step 2: Calculate the power flexibility based on the baseline
        print("\n6. Calculating power flexibility...")
        flexibility_df = calculate_power_flexibility(baseline_df, params)
        
        # Step 3: Save the results to a new CSV
        output_path = "results_with_flexibility.csv"
        flexibility_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"✅ Results with flexibility have been exported to '{output_path}'")
        
        # Step 4: Plot the results
        plot_flexibility(flexibility_df)
        
        # Display the first few rows with the new flexibility data
        print("\n--- Results Preview with Flexibility ---")
        print(flexibility_df[['Time_Slot_EXT', 'P_IT_Total_kW', 'Upward_Flexibility_kW', 'Downward_Flexibility_kW']].head().to_string())