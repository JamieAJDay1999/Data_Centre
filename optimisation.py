"""
optimisation.py – data-centre operating-cost optimisation
--------------------------------------------------------

* Two flags let you decide whether end-of-horizon states must equal the
  initial ones (periodic / “cyclic” run):

      CYCLE_TEMPERATURES = False   # True  → enforce T(N-1)=T(0)
      CYCLE_TES_ENERGY   = True    # False → allow different TES level

* Horizon and time-step are freely adjustable via the two lines in
  `setup_optimisation_parameters()`.

The script has been tested with Gurobi 12.0.2.
"""

import traceback
import os
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from parameters import setup_simulation_parameters      # ← your parameters.py


# ---------------------------------------------------------------------------
# user-selectable periodicity
# ---------------------------------------------------------------------------
CYCLE_TEMPERATURES = False     # set True for 60-min / 24-h periodic runs
CYCLE_TES_ENERGY   = False      # keep True in most cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# parameter preparation
# ---------------------------------------------------------------------------

def setup_optimisation_parameters():
    """Clone the 'cool_down' parameter set and adapt for optimisation."""
    base = setup_simulation_parameters("cool_down")
    p    = base.copy()

    # -- horizon & discretisation (edit here as needed) --------------------
    p['simulation_time_minutes'] = 2     # e.g. 30-minute horizon
    p['dt']                       = 1    # 10-second step
    # ----------------------------------------------------------------------

    p['TES_initial_charge_kWh'] = base.get('TES_initial_charge', 10.0)
    p['TES_capacity_kWh']       = base.get('TES_kwh_cap', 2000.0)
    p['E_TES_min_kWh']          = 0.0

    limit = base.get('T_cAisle_upper_limit_Celsius')
    if limit is None:
        limit = setup_simulation_parameters("warm_up")\
                .get('T_cAisle_upper_limit_Celsius', 27.0)
    p['T_cAisle_upper_limit_Celsius'] = limit
    p['T_IT_upper_limit_Celsius']     = 75.0

    p['simulation_time_seconds'] = p['simulation_time_minutes']*60
    p['num_time_points'] = int(p['simulation_time_seconds'] / p['dt'])
    p['dt_hours'] = p['dt']/3600.0

    # sanity check
    required = [
        'C_IT', 'G_conv', 'C_Rack', 'm_dot_air', 'kappa', 'c_p_air', 'COP_HVAC',
        'C_cAisle', 'G_cold', 'C_hAisle', 'P_IT_heat_source', 'T_out_Celsius',
        'P_HVAC_min_watts', 'P_HVAC_max_watts',
        'TES_charge_efficiency', 'TES_discharge_efficiency',
        'TES_w_charge_max', 'TES_w_discharge_max'
    ]
    for k in required:
        if k not in p or (p[k] is None and k != 'T_cAisle_upper_limit_Celsius'):
            raise ValueError(f"parameter '{k}' missing or None")

    return p


# ---------------------------------------------------------------------------
# synthetic electricity tariff
# ---------------------------------------------------------------------------

def make_tariff(n, dt_s):
    hours = np.arange(n)*dt_s/3600.0
    base, peak = 0.10, 0.25
    price = base + (peak-base)*0.5*(1-np.cos(2*np.pi*(hours-6)/24))
    price += np.random.normal(0, 0.01, n)
    return np.clip(price, 0.02, None)


# ---------------------------------------------------------------------------
# optimisation routine
# ---------------------------------------------------------------------------

def run_optimisation(p, tariff):
    N, dt_s, dt_h = p['num_time_points'], p['dt'], p['dt_hours']

    try:
        m = gp.Model("dc_cost_opt")

        # ---------------- decision variables ------------------------------
        T_IT   = m.addVars(N, lb=10, ub=p['T_IT_upper_limit_Celsius'], name="T_IT")
        T_Rack = m.addVars(N, lb=10, ub=70, name="T_Rack")
        T_c    = m.addVars(N, lb=10, ub=p['T_cAisle_upper_limit_Celsius'], name="T_cAisle")
        T_h    = m.addVars(N, lb=10, ub=80, name="T_hAisle")
        E_TES  = m.addVars(N, lb=p['E_TES_min_kWh'], ub=p['TES_capacity_kWh'], name="E_TES")

        P_HVAC = m.addVars(N, lb=p['P_HVAC_min_watts'], ub=p['P_HVAC_max_watts'], name="P_HVAC")
        P_ch   = m.addVars(N, lb=0, ub=p['TES_w_charge_max'],    name="P_TES_ch")
        P_dis  = m.addVars(N, lb=0, ub=p['TES_w_discharge_max'], name="P_TES_dis")

        T_in   = m.addVars(N, lb=5, ub=30, name="T_Air_in")
        P_cool = m.addVars(N, lb=0, name="P_Cool")

        # ---------------- initial conditions ------------------------------
        m.addConstr(T_IT[0]   == p['T_IT_initial_Celsius'],   name="init_T_IT")
        m.addConstr(T_Rack[0] == p['T_Rack_initial_Celsius'], name="init_T_Rack")
        m.addConstr(T_c[0]    == p['T_cAisle_initial'],       name="init_T_cAisle")
        m.addConstr(T_h[0]    == p['T_hAisle_initial'],       name="init_T_hAisle")
        m.addConstr(E_TES[0]  == p['TES_initial_charge_kWh'], name="init_E_TES")

        # ---------------- dynamics ----------------------------------------
        denom_mcp = p['m_dot_air'] * p['c_p_air']

        for t in range(N-1):
            # total cooling power
            m.addConstr(P_cool[t] == P_HVAC[t] + P_dis[t], name=f"P_cool_{t}")

            # supply air temperature
            m.addConstr(T_in[t] == T_h[t] -
                        (P_cool[t]*p['COP_HVAC'])/denom_mcp,
                        name=f"T_in_{t}")

            # explicit Euler updates
            m.addConstr(
                T_IT[t+1] == T_IT[t] + dt_s * (
                    (p['P_IT_heat_source'] -
                     p['G_conv']*(T_IT[t]-T_Rack[t])) / p['C_IT']),
                name=f"T_IT_upd_{t}"
            )

            m.addConstr(
                T_Rack[t+1] == T_Rack[t] + dt_s * (
                    (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_c[t]-T_Rack[t]) +
                     p['G_conv']*(T_IT[t]-T_Rack[t])) / p['C_Rack']),
                name=f"T_Rack_upd_{t}"
            )

            m.addConstr(
                T_c[t+1] == T_c[t] + dt_s * (
                    (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_in[t]-T_c[t]) -
                     p['G_cold']*(T_c[t]-p['T_out_Celsius'])) / p['C_cAisle']),
                name=f"T_cAisle_upd_{t}"
            )

            m.addConstr(
                T_h[t+1] == T_h[t] + dt_s * (
                    (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_Rack[t]-T_h[t]))
                    / p['C_hAisle']),
                name=f"T_hAisle_upd_{t}"
            )

            # TES energy balance (kWh)
            dE = (P_ch[t]*p['TES_charge_efficiency']
                  - P_dis[t]/p['TES_discharge_efficiency']) * dt_h / 1000.0
            m.addConstr(E_TES[t+1] == E_TES[t] + dE, name=f"E_TES_upd_{t}")

        # last-step algebraic
        m.addConstr(P_cool[N-1] == P_HVAC[N-1] + P_dis[N-1], name="P_cool_last")
        m.addConstr(T_in[N-1] == T_h[N-1] -
                    (P_cool[N-1]*p['COP_HVAC'])/denom_mcp,
                    name="T_in_last")

        # ---------------- optional cyclic equalities ----------------------
        if CYCLE_TEMPERATURES:
            m.addConstr(T_IT[N-1]   == T_IT[0],   name="cyc_T_IT")
            m.addConstr(T_Rack[N-1] == T_Rack[0], name="cyc_T_Rack")
            m.addConstr(T_c[N-1]    == T_c[0],    name="cyc_T_cAisle")
            m.addConstr(T_h[N-1]    == T_h[0],    name="cyc_T_hAisle")
        if CYCLE_TES_ENERGY:
            m.addConstr(E_TES[N-1]  == E_TES[0],  name="cyc_E_TES")

        # ---------------- objective ----------------------------------------
        cost = gp.quicksum(
            (P_HVAC[t] + P_ch[t] + p['P_IT_heat_source']) *
            tariff[t] * dt_h for t in range(N)
        )
        m.setObjective(cost, GRB.MINIMIZE)

        # ---------------- solve -------------------------------------------
        m.optimize()

        if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print(f"Model status {m.status} – optimisation failed.")
            if m.status == GRB.INFEASIBLE:
                m.computeIIS()
                m.write("datacenter_iis.ilp")
                print("IIS written to datacenter_iis.ilp")
            return None

        print(f"Optimal cost for {p['simulation_time_minutes']} min: "
              f"${m.ObjVal:,.2f}")

        return {
            'time_min': np.arange(N)*dt_s/60.0,
            'T_IT':  [T_IT[t].X   for t in range(N)],
            'T_Rack':[T_Rack[t].X for t in range(N)],
            'T_c':   [T_c[t].X    for t in range(N)],
            'T_h':   [T_h[t].X    for t in range(N)],
            'T_in':  [T_in[t].X   for t in range(N)],
            'E_TES': [E_TES[t].X  for t in range(N)],
            'P_HVAC':[P_HVAC[t].X for t in range(N)],
            'P_ch':  [P_ch[t].X   for t in range(N)],
            'P_dis': [P_dis[t].X  for t in range(N)],
            'price': tariff,
            'cost':  m.ObjVal,
        }

    except gp.GurobiError as e:
        print(f"Gurobi error {e.errno}: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print("Unexpected error:", e)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# quick textual summary
# ---------------------------------------------------------------------------

def quick_report(res):
    print("\nT_IT range  : {:.1f} … {:.1f} °C".format(min(res['T_IT']),
                                                      max(res['T_IT'])))
    print("TES level   : {:.1f} … {:.1f} kWh".format(min(res['E_TES']),
                                                      max(res['E_TES'])))

def plot_results(res, out_dir="static"):
    os.makedirs(out_dir, exist_ok=True)
    t = res['time_min']
    fig1, ax1 = plt.subplots(figsize=(10,8), nrows=3, sharex=True)
    ax1[0].plot(t, res['T_IT'],  label="T_IT")
    ax1[0].plot(t, res['T_Rack'],label="T_Rack")
    ax1[0].plot(t, res['T_c'],   label="T_cAisle")
    ax1[0].plot(t, res['T_h'],   label="T_hAisle")
    ax1[0].plot(t, res['T_in'],  label="T_Air_in", linestyle='--')
    ax1[0].legend(); ax1[0].set_ylabel("°C"); ax1[0].grid(True)

    ax1[1].plot(t, np.array(res['P_HVAC'])/1000, label="HVAC kW")
    ax1[1].plot(t, (np.array(res['P_HVAC'])+np.array(res['P_dis']))/1000,
                label="Total cooling kW", linestyle='--')
    ax1[1].legend(); ax1[1].set_ylabel("kW"); ax1[1].grid(True)

    ax1[2].plot(t, res['price'], color='green')
    ax1[2].set_ylabel("$ / kWh"); ax1[2].set_xlabel("time [min]")
    ax1[2].grid(True)

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "opt_plot_temps_power.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(t, res['E_TES'], label="E_TES kWh")
    ax2.set_ylabel("kWh"); ax2.set_xlabel("time [min]")
    ax22 = ax2.twinx()
    ax22.plot(t, np.array(res['P_ch'])/1000, label="Charge kW",   linestyle='--')
    ax22.plot(t, np.array(res['P_dis'])/1000,label="Discharge kW",linestyle='-.')
    ax2.legend(loc='upper left'); ax22.legend(loc='upper right')
    ax2.grid(True)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "opt_plot_tes.png"))
    plt.close(fig2)

    print(f"Plots saved to ./{out_dir}/")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = setup_optimisation_parameters()
    tariff = make_tariff(p['num_time_points'], p['dt'])
    result = run_optimisation(p, tariff)
    if result:
        quick_report(result)
        plot_results(result)
