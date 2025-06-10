"""
optimisation.py
===============

Detailed, step-by-step formulation of a small mixed-integer-free**¹**
optimisation model for a data-centre thermal/energy system.

* The physical model is a **first-order RC network** (lumped capacity nodes).
* The optimiser chooses **HVAC power** and **TES charge / discharge** profiles
  to minimise electricity cost over a short horizon.

This script is aimed at **Gurobi beginners** – almost every modelling step is
commented.  Feel free to shorten comments once you become familiar with the API.

-----------
¹ All variables are continuous: we solve a Linear Program (LP).  If you later
  add integer variables (for on/off devices, etc.), Gurobi automatically turns
  it into a Mixed-Integer LP (MILP) without any changes to the Solve call.
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------

import numpy as np                       # numerical helper
import gurobipy as gp                    # Gurobi Python interface
from gurobipy import GRB                 # convenience: GRB.OPTIMAL, GRB.MINIMIZE
import traceback                         # nicer error reporting

from parameters import setup_simulation_parameters
# (parameters.py was supplied by you and is *not* reproduced here)


# ---------------------------------------------------------------------------
# 1. User-changeable switches
# ---------------------------------------------------------------------------

# If True we force the temperatures at the final step to equal their initial
# values – useful for 60-min or 24-h “cyclic” studies.  For a 30-min horizon the
# plant often cannot get back to its start state, so we leave this False.
CYCLE_TEMPERATURES = False

# If True we also force the thermal-energy-storage (TES) state of charge to
# return to its initial value.  For short horizons you can switch this off, but
# *usually* you keep it True so the optimiser cannot “cheat” by fully
# discharging the store without paying the price later.
CYCLE_TES_ENERGY = True


# ---------------------------------------------------------------------------
# 2. Parameter helper – convert the raw “cool_down” parameter set into the one
#    we want for optimisation
# ---------------------------------------------------------------------------

def prepare_optimisation_parameters():
    """
    1. Grab the 'cool_down' parameter set from parameters.py.
    2. Overwrite the horizon length and time step for this optimisation.
    3. Derive a few helpful extra keys (dt in hours, number of steps, etc.).
    4. Check that every parameter we depend on actually exists.
    Returns
    -------
    dict
        A *flat* dictionary `p` – we pass it around instead of using global
        variables.
    """

    # Base dictionary (already contains all geometry & material properties)
    p = setup_simulation_parameters("cool_down")

    # --- horizon and discretisation (edit these two numbers to taste) -----
    p['simulation_time_minutes'] = 30   # horizon length
    p['dt'] = 10                        # time step (seconds)
    # ----------------------------------------------------------------------

    # Convenience: kWh versions of TES parameters
    p['TES_initial_charge_kWh'] = p['TES_initial_charge']
    p['TES_capacity_kWh']       = p['TES_kwh_cap']
    p['E_TES_min_kWh']          = 0.0

    # If 'cool_down' had no cold-aisle upper limit, copy it from warm-up mode
    if p.get('T_cAisle_upper_limit_Celsius') is None:
        warm = setup_simulation_parameters("warm_up")
        p['T_cAisle_upper_limit_Celsius'] = warm['T_cAisle_upper_limit_Celsius']

    p['T_IT_upper_limit_Celsius'] = 75.0   # arbitrary safety limit

    # --- derived time quantities -----------------------------------------
    p['simulation_time_seconds'] = p['simulation_time_minutes'] * 60
    p['num_time_points'] = int(p['simulation_time_seconds'] / p['dt'])
    p['dt_hours'] = p['dt'] / 3600.0

    # --- sanity check: fail early if anything we rely on is missing -------
    required = [
        'C_IT', 'G_conv', 'C_Rack', 'm_dot_air', 'kappa', 'c_p_air',
        'COP_HVAC', 'C_cAisle', 'G_cold', 'C_hAisle', 'P_IT_heat_source',
        'T_out_Celsius', 'P_HVAC_min_watts', 'P_HVAC_max_watts',
        'TES_charge_efficiency', 'TES_discharge_efficiency',
        'TES_w_charge_max', 'TES_w_discharge_max'
    ]
    for key in required:
        if key not in p:
            raise KeyError(f"Parameter '{key}' is missing from parameters.py")
        if p[key] is None:
            raise ValueError(f"Parameter '{key}' is None – please set a value")

    return p


# ---------------------------------------------------------------------------
# 3. Synthetic electricity tariff (smooth daily profile + random noise)
# ---------------------------------------------------------------------------

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:
    """
    Produce a vector of electricity prices (USD/kWh).  Simple sinusoid:
    base 0.10 USD/kWh – peak 0.25 USD/kWh around 18:00 local time.
    """
    hours = np.arange(num_steps) * dt_seconds / 3600.0
    base, peak = 0.10, 0.25
    # Scaled cosine gives nice smooth valley (~6 am) and peak (~18 pm)
    price = base + (peak - base) * 0.5 * (1 - np.cos(2 * np.pi * (hours - 6) / 24))
    # Add a tiny bit of random variability
    price += np.random.normal(loc=0.0, scale=0.01, size=num_steps)
    # Clip to a reasonable minimum
    return np.clip(price, 0.02, None)


# ---------------------------------------------------------------------------
# 4. Main optimisation model
# ---------------------------------------------------------------------------

def optimise(p: dict, tariff: np.ndarray):
    """
    Build and solve the LP with Gurobi.

    Parameters
    ----------
    p : dict
        Prepared parameter dictionary.
    tariff : np.ndarray
        Vector of electricity prices (USD/kWh) for each time step.

    Returns
    -------
    dict | None
        Results (variable trajectories + cost) **or** None if the model is
        infeasible or another error occurs.
    """

    # --- unpack a few constants to keep the code tidy ---------------------
    N      = p['num_time_points']       # number of discrete time points
    dt_s   = p['dt']                    # [s]
    dt_h   = p['dt_hours']              # [h] – for energy & cost
    mcp    = p['m_dot_air'] * p['c_p_air']  # mass-flow × specific heat [W/K]

    try:
        # ------------------------------------------------------------------
        # 4.1  Create a Gurobi Model object – the *container* for everything
        # ------------------------------------------------------------------
        # Names are optional but helpful if you inspect logs or IIS files.
        model = gp.Model("data_centre_cost_min")

        # ------------------------------------------------------------------
        # 4.2  Decision variables
        # ------------------------------------------------------------------
        # We declare **one variable per node / actuator per time step**.
        # All are continuous (default vtype is GRB.CONTINUOUS).
        #
        # addVars(dim, lb=?, ub=?, name="...") returns a dict-like object
        # that we index with integers 0 … N-1.

        # Temperatures (°C)
        T_IT   = model.addVars(N, lb=10, ub=p['T_IT_upper_limit_Celsius'],
                               name="T_IT")        # IT silicon
        T_Rack = model.addVars(N, lb=10, ub=70, name="T_Rack")            # rack air
        T_c    = model.addVars(N, lb=10, ub=p['T_cAisle_upper_limit_Celsius'],
                               name="T_cAisle")     # cold aisle air
        T_h    = model.addVars(N, lb=10, ub=80, name="T_hAisle")          # hot aisle air
        T_in   = model.addVars(N, lb=5,  ub=30, name="T_Air_in")          # HVAC supply air

        # Energy in TES (kWh)
        E_TES  = model.addVars(N, lb=p['E_TES_min_kWh'], ub=p['TES_capacity_kWh'],
                               name="E_TES")

        # Power flows (Watts except where noted)
        P_HVAC = model.addVars(N, lb=p['P_HVAC_min_watts'],
                               ub=p['P_HVAC_max_watts'], name="P_HVAC")
        P_ch   = model.addVars(N, lb=0, ub=p['TES_w_charge_max'], name="P_TES_charge")
        P_dis  = model.addVars(N, lb=0, ub=p['TES_w_discharge_max'], name="P_TES_dis")

        # Algebraic helper: total cooling power (HVAC + TES discharge)
        P_cool = model.addVars(N, lb=0, name="P_Cooling")

        # ------------------------------------------------------------------
        # 4.3  Initial-state constraints
        # ------------------------------------------------------------------
        # addConstr() adds **one** constraint object (linear left == right).
        model.addConstr(T_IT[0]   == p['T_IT_initial_Celsius'],   "init_T_IT")
        model.addConstr(T_Rack[0] == p['T_Rack_initial_Celsius'], "init_T_Rack")
        model.addConstr(T_c[0]    == p['T_cAisle_initial'],       "init_T_c")
        model.addConstr(T_h[0]    == p['T_hAisle_initial'],       "init_T_h")
        model.addConstr(E_TES[0]  == p['TES_initial_charge_kWh'], "init_E_TES")

        # ------------------------------------------------------------------
        # 4.4  Dynamic constraints – loop over time steps
        # ------------------------------------------------------------------
        for t in range(N - 1):
            # ---------- algebraic relations (no time derivative) ----------
            # -> Total cooling = mechanical HVAC power + TES discharge
            model.addConstr(P_cool[t] == P_HVAC[t] + P_dis[t],
                            name=f"P_cool_def_{t}")

            # -> HVAC supply-air temperature from cooling power
            #      T_in  = T_h - (COP * P_cool) / (ṁ * c_p)
            model.addConstr(
                T_in[t] == T_h[t] -
                (P_cool[t] * p['COP_HVAC']) / mcp,
                name=f"T_in_def_{t}"
            )

            # ---------- explicit Euler discretisation for temperatures ----
            # Each node has: C * dT/dt  =  Σ heat flows
            #
            # We update:  T_{t+1} = T_t + (dT/dt)_t * Δt
            #
            # *All coefficients are numeric constants*, so each equation is
            # affine in the decision variables → the whole problem is linear.

            # IT silicon
            dT_IT_dt = (p['P_IT_heat_source']
                        - p['G_conv'] * (T_IT[t] - T_Rack[t])) / p['C_IT']
            model.addConstr(T_IT[t+1] == T_IT[t] + dT_IT_dt * dt_s,
                            name=f"dyn_T_IT_{t}")

            # Rack air
            dT_Rack_dt = (
                p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_c[t] - T_Rack[t])
                + p['G_conv'] * (T_IT[t] - T_Rack[t])
            ) / p['C_Rack']
            model.addConstr(T_Rack[t+1] == T_Rack[t] + dT_Rack_dt * dt_s,
                            name=f"dyn_T_Rack_{t}")

            # Cold aisle
            dT_c_dt = (
                p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_in[t] - T_c[t])
                - p['G_cold'] * (T_c[t] - p['T_out_Celsius'])
            ) / p['C_cAisle']
            model.addConstr(T_c[t+1] == T_c[t] + dT_c_dt * dt_s,
                            name=f"dyn_T_c_{t}")

            # Hot aisle
            dT_h_dt = (
                p['m_dot_air'] * p['kappa'] * p['c_p_air'] * (T_Rack[t] - T_h[t])
            ) / p['C_hAisle']
            model.addConstr(T_h[t+1] == T_h[t] + dT_h_dt * dt_s,
                            name=f"dyn_T_h_{t}")

            # ---------- TES energy balance (kWh) ---------------------------
            # Positive P_ch charges the store (→ efficiency < 1),
            # Positive P_dis discharges the store (divide by efficiency).
            dE_kWh = (
                P_ch[t] * p['TES_charge_efficiency']          # Wh
                - P_dis[t] / p['TES_discharge_efficiency']    # Wh
            ) * dt_h / 1000.0                                 # Wh → kWh
            model.addConstr(E_TES[t+1] == E_TES[t] + dE_kWh,
                            name=f"dyn_E_TES_{t}")

        # ---------- last time step’s algebraic definitions -----------------
        # (repeat supply-air relation so T_in[N-1] is defined)
        model.addConstr(P_cool[N-1] == P_HVAC[N-1] + P_dis[N-1],
                        name="P_cool_def_last")
        model.addConstr(
            T_in[N-1] == T_h[N-1] -
            (P_cool[N-1] * p['COP_HVAC']) / mcp,
            name="T_in_def_last"
        )

        # ------------------------------------------------------------------
        # 4.5  Optional “cyclic” equalities
        # ------------------------------------------------------------------
        if CYCLE_TEMPERATURES:
            model.addConstr(T_IT[N-1]   == T_IT[0],   "cyc_T_IT")
            model.addConstr(T_Rack[N-1] == T_Rack[0], "cyc_T_Rack")
            model.addConstr(T_c[N-1]    == T_c[0],    "cyc_T_c")
            model.addConstr(T_h[N-1]    == T_h[0],    "cyc_T_h")

        if CYCLE_TES_ENERGY:
            model.addConstr(E_TES[N-1] == E_TES[0],   "cyc_E_TES")

        # ------------------------------------------------------------------
        # 4.6  Objective function – minimise total electricity cost
        # ------------------------------------------------------------------
        total_cost = gp.quicksum(
            (P_HVAC[t] + P_ch[t] + p['P_IT_heat_source'])   # total power [W]
            * tariff[t]                                     # $ / kWh
            * dt_h / 1000.0                                 # convert W → kW
            for t in range(N)
        )
        model.setObjective(total_cost, GRB.MINIMIZE)

        # ------------------------------------------------------------------
        # 4.7  Solve the model
        # ------------------------------------------------------------------
        model.optimize()    # **the** call that actually invokes Gurobi

        # Basic check of outcome
        if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print(f"Gurobi ended with status {model.status}.")
            if model.status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write("datacenter_iis.ilp")
                print("IIS written to datacenter_iis.ilp (infeasible set).")
            return None

        # ------------------------------------------------------------------
        # 4.8  Extract results into plain Python lists / NumPy arrays
        # ------------------------------------------------------------------
        res = {
            'time_min': np.arange(N) * dt_s / 60.0,
            'price':    tariff,
            # temperatures
            'T_IT':  [T_IT[t].X   for t in range(N)],
            'T_Rack':[T_Rack[t].X for t in range(N)],
            'T_c':   [T_c[t].X    for t in range(N)],
            'T_h':   [T_h[t].X    for t in range(N)],
            'T_in':  [T_in[t].X   for t in range(N)],
            # storage & power
            'E_TES': [E_TES[t].X  for t in range(N)],
            'P_HVAC':[P_HVAC[t].X for t in range(N)],
            'P_ch':  [P_ch[t].X   for t in range(N)],
            'P_dis': [P_dis[t].X  for t in range(N)],
            # scalar objective value
            'cost': model.ObjVal,
        }
        return res

    # ---- catch & print any Gurobi / Python errors ------------------------
    except gp.GurobiError as e:
        print(f"GurobiError {e.errno}: {e}")
        traceback.print_exc()
        return None
    except Exception as err:
        print("Unexpected error during optimisation:", err)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 5. Tiny helper to print a quick summary (feel free to replace with plots)
# ---------------------------------------------------------------------------

def quick_report(res: dict):
    """Pretty-print a few headline numbers."""
    print(f"\n▸ Optimal cost       : ${res['cost']:.2f}")
    print(f"▸ IT temperature min : {min(res['T_IT']):.1f} °C")
    print(f"▸ IT temperature max : {max(res['T_IT']):.1f} °C")
    print(f"▸ TES level min      : {min(res['E_TES']):.1f} kWh")
    print(f"▸ TES level max      : {max(res['E_TES']):.1f} kWh")


# ---------------------------------------------------------------------------
# 6. “Main” – executed when you run  `python optimisation.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = prepare_optimisation_parameters()
    price_vector = generate_tariff(params['num_time_points'], params['dt'])

    results = optimise(params, price_vector)
    if results:
        quick_report(results)
