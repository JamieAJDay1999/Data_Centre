import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper functions (used internally by setup_simulation_parameters)
# These calculate derived thermal properties based on primary inputs.

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    hourly_prices = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70] # original
    num_hours = (num_steps * dt_seconds) // 3600
    full_price_series = np.tile(hourly_prices, int(np.ceil(num_hours / 24)))
    price_per_step = np.repeat(full_price_series, 3600 // dt_seconds)
    return np.insert(price_per_step[:num_steps], 0, 0)

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

        # Time slots (1-based indexing for readability)
        self.T_SLOTS = range(1, 1 + int(self.simulation_minutes * 60 / self.dt_seconds))
        self.TEXT_SLOTS = range(1, 1 + self.num_steps_extended)
        self.K_TRANCHES = range(1, 5)

        # --- IT Equipment ---
        self.idle_power_kw = 166.7
        self.max_power_kw = 1000.0
        self.max_cpu_usage = 1.0
        self.tranche_max_delay = {1: 2, 2: 4, 3: 8, 4: 12}
        self.nominal_overhead_addition = 53.095 # For other DC loads (lighting, etc.) # 7% of average power consumption in the DC

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

def _calc_cit(it_params):
    """Calculate the total heat capacity of IT equipment."""
    return (it_params['n_racks'] * it_params['n_servers_per_rack'] * it_params['mass_server'] * it_params['c_server'] +
            it_params['n_racks'] * it_params['mass_rack'] * it_params['c_rack'])

def _calc_g_conv(it_params):
    """Calculate the convective heat conductance from IT equipment to rack air."""
    return it_params['n_racks'] * it_params['n_servers_per_rack'] * it_params['mass_server'] * it_params['g_conv_per_k']

def _calc_free_space_in_rack(rack_dims_params):
    """Calculates the free air volume inside a single rack."""
    server_volume = rack_dims_params['server_height'] * rack_dims_params['server_width'] * rack_dims_params['server_depth']
    rack_volume = rack_dims_params['rack_height'] * rack_dims_params['rack_width'] * rack_dims_params['rack_depth']
    total_server_volume = rack_dims_params['n_servers_per_rack'] * server_volume
    return rack_volume - total_server_volume

def _calc_dc_wall_area(dc_dims):
    """Calculates the relevant data center wall area for heat exchange."""
    return (2 * dc_dims['dc_height'] * dc_dims['dc_length'] +
            2 * dc_dims['dc_height'] * dc_dims['dc_width'] +
            dc_dims['dc_length'] * dc_dims['dc_width'])

def calc_c_rack(it_params, params):
    """Calculate the total heat capacity of a rack."""
    free_space_volume = _calc_free_space_in_rack(it_params)
    return (free_space_volume * it_params['n_racks'] * params['rho_air'] * params['c_p_air'] +
            it_params['n_racks'] * it_params['mass_rack'] * it_params['c_rack'] +
            it_params['n_racks'] * it_params['n_servers_per_rack'] * it_params['mass_server'] * it_params['c_server'])

def setup_simulation_parameters(mode="cool_down"):
    """
    Sets up simulation parameters, tailored for the optimisation.py script.
    Currently, this function is configured primarily for "cool_down" mode as used by optimisation.py.
    """
    if mode != "cool_down":
        print(f"Warning: This parameters.py is primarily configured for 'cool_down' mode. Proceeding with 'cool_down' parameters for mode '{mode}'.")

    params = {}

    # === I. Core Physical Properties ===
    params['rho_air'] = 1.16  # Density of air (kg/m^3)
    params['c_p_air'] = 1005.45 # Specific heat capacity of air (J/kg*K)
    params['m_dot_air'] = 100   
    params['T_out_Celsius'] = 22

    # === II. Data Center and IT Equipment Specifications ===
    _dc_dims = {
        'dc_length': 28.0,  # Length of the data center (m)
        'dc_width': 10.0,   # Width of the data center (m)
        'dc_height': 3.0    # Height of the data center (m)
    }
    _it_specs = {
        'n_racks': 100,
        'n_servers_per_rack': 10,
        'mass_server': 20.0,    # Mass of each server (kg)
        'c_server': 600.0,      # Specific heat capacity of server (J/kg*K)
        'mass_rack': 140.0,     # Mass of the rack (kg)
        'c_rack': 420.0,        # Specific heat capacity of rack (J/kg*K)
        'g_conv_per_k': 12000.0 / 2202.0, # Convective heat conductance server to air (W/K/kg)
        # Server and Rack Dimensions (for C_Rack calculation)
        'server_height': 0.0868, # (m)
        'server_width': 0.434,   # (m)
        'server_depth': 0.679,   # (m)
        'rack_height': 2.0,      # (m)
        'rack_width': 0.605,    # (m)
        'rack_depth': 1.2       # (m)
    }
    params['P_IT_heat_source'] = 300000.0  # Total heat generated by IT equipment (Watts)

    # === III. Calculated Thermal Properties (used by optimisation.py) ===
    params['C_IT'] = _calc_cit(_it_specs)  # Total heat capacity of IT equipment (J/K)
    params['G_conv'] = np.round(_calc_g_conv(_it_specs), 3) # Convective heat conductance IT to Rack Air (W/K)

    _v_free_space_per_rack = np.round(_calc_free_space_in_rack(_it_specs), 3)
    params['C_Rack'] = calc_c_rack(_it_specs, params)  # Total heat capacity of a rack (J/K)
    _v_cAisle = 2000.0  # Volume of cold aisle (m^3) - assumption
    params['C_cAisle'] = _v_cAisle * params['rho_air'] * params['c_p_air'] # (J/K)

    _v_hAisle = 1000#96.0    # Volume of hot aisle (m^3) - assumption
    params['C_hAisle'] = _v_hAisle * params['rho_air'] * params['c_p_air'] # (J/K)

    # For G_cold calculation
    _alpha_cAisle = 16.0  # Heat transfer coefficient for cold aisle walls (W/m^2K)
    _a_cAisle = _calc_dc_wall_area(_dc_dims) # Wall area (m^2)
    _r_cold_K_per_W = np.round((0.7 / 1000) / (_a_cAisle / 38.0), 4) if _a_cAisle > 0 else float('inf')

    _term1_Gcold_inv = 1.0 / (_alpha_cAisle * _a_cAisle) if (_alpha_cAisle * _a_cAisle) > 0 else float('inf')
    _denominator_g_cold = _term1_Gcold_inv + _r_cold_K_per_W
    params['G_cold'] = 1.0 / _denominator_g_cold if _denominator_g_cold > 0 else (_alpha_cAisle * _a_cAisle) # (W/K)

    # === IV. Cooling System Parameters ===
    params['COP_HVAC'] = 5
    params['kappa'] = 0.7663  # Air mixing factor / bypass factor for cooling coils
    params['P_chiller_max'] = 400000.0
    params['P_HVAC_ramp'] = 100000.0  # Max power ramp for HVAC (W per time step dt)

    # Thermal Energy Storage (TES)
    params['TES_kwh_cap'] = 1000.0  # Nominal capacity (kWh)
    params['TES_w_discharge_max'] = 300000.0  # Max discharge power (W)
    params['TES_discharge_efficiency'] = 0.9
    params['TES_w_charge_max'] = 300000.0  # Max charge power (W)
    params['TES_charge_efficiency'] = 0.9
    params['E_TES_min_kWh'] = 0.0  # Minimum charge state (kWh)
    params['TES_initial_charge_kWh'] = 0.5 * params['TES_kwh_cap'] # Initial charge (kWh)

    # === V. Initial Conditions & Operational Limits for "cool_down" mode ===
    params['T_IT_initial_Celsius'] = 28.5
    params['T_Rack_initial_Celsius'] = 26
    params['T_cAisle_initial'] = 20
    params['T_hAisle_initial'] = 27
    params['T_target_Air_in_Celsius'] = 20
    
    params['T_cAisle_lower_limit_Celsius'] = 18
    params['T_cAisle_upper_limit_Celsius'] = 22.5

    # Add IT specs and DC dims for printing in __main__
    params['_it_specs'] = _it_specs
    params['_dc_dims'] = _dc_dims

    return params

if __name__ == "__main__":
    params = setup_simulation_parameters("cool_down")
    print(f'C_IT: {params["C_IT"]} J/K')
    print(f'G_conv: {params["G_conv"]} W/K')
    print(f'C_Rack: {params["C_Rack"]} J/K')
    print(f'C_cAisle: {params["C_cAisle"]} J/K')
    print(f'C_hAisle: {params["C_hAisle"]} J/K')
    print(f'G_cold: {params["G_cold"]} W/K')
    