"""
optimisation.py  –  small LP with plotting (PuLP + CBC version)
===============================================================

* Keeps the original model and plots.
* Replaces Gurobi with the free, open-source CBC solver via PuLP.
* Install requirements:  pip install pulp numpy matplotlib
"""

import os, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp                                # << replaces gurobipy
import datetime, pathlib, json          # <- only for debugging artefacts
from parameters_optimisation import setup_simulation_parameters

# --------- user switches --------------------------------------------------
CYCLE_TEMPERATURES = False
CYCLE_TES_ENERGY   = True
# -------------------------------------------------------------------------

_DBGDIR = pathlib.Path("lp_debug")
_DBGDIR.mkdir(exist_ok=True)



def read_IT_optimisation_results():
    df = pd.read_csv("AllResults_PuLP.csv")
    itp = df['Optimized_Total_IT_Power_kW'].iloc[:96]
    return np.repeat(itp.values, 15) * 1000.0  # convert kW to W, repeat for each time step


def setup_optimisation_parameters():
    p = setup_simulation_parameters("cool_down")

    # horizon & discretisation
    p['simulation_time_minutes'] = 1440    # 2-minute simulation
    p['dt'] = 60              # 1-second step
    p['simulation_time_seconds'] = p['simulation_time_minutes'] * 60
    p['num_time_points'] = int(p['simulation_time_seconds'] / p['dt'])
    p['dt_hours'] = p['dt'] / 3600.0

    # airflow
    p['m_dot_air'] = 100                 # kg/s

    p['TES_capacity_kWh']    = p['TES_kwh_cap'] 

    #p['P_IT_heat_source'] = read_IT_optimisation_results()
    #print(p['P_IT_heat_source'])
    #print(len(p['P_IT_heat_source']))
    
    return p

def generate_tariff(num_steps: int, dt_seconds: float) -> np.ndarray:

    tarrif = pd.read_csv("static/uk_electricity_prices_hourly.csv")
    tarrif['time'] = pd.date_range(start='2023-01-01', periods=len(tarrif), freq='h')
    tarrif = tarrif.set_index('time').resample('s').ffill()
    tarrif = tarrif[::dt_seconds]  # downsample to match dt_seconds
    price = np.array(tarrif['Price'].values / 1000.0)[:-1]  # convert to £/kWh
    price = np.array(len(price) * [price.mean()])  # use average price for all time steps
    #price = np.array(1440 * [price.mean()])  # use average price for all time steps
    
    return price

def _stamp(name: str, ext: str) -> pathlib.Path:
    """timestamped file path inside lp_debug/  (e.g. 2025-06-05T12-30-17_model.lp)"""
    ts = datetime.datetime.now().strftime("%Y-m-dT%H-M-S")
    return _DBGDIR / f"{ts}_{name}.{ext}"
# ---------------------------------------------------------------------------


def solve_lp(model, use_highs=False, logfile="solver.log"):
    """
    Solve *model* with CBC (default) or HiGHS.
    • writes the solver log to lp_debug/<timestamp>_solver.log
    • if the model is infeasible or unbounded, dumps:
        – the LP file      (.lp  human-readable)
        – the MPS file     (.mps machine-readable for external tools)
        – a JSON manifest  summarising variables / constraints counts
    """
    log_path = _stamp("solver", "log")
    if use_highs:
        try:
            solver = pulp.HIGHS_CMD(msg=True,
                                    logfile=str(log_path),
                                    timeLimit=None)
        except AttributeError:
            print("HiGHS not available, falling back to CBC")
            use_highs = False   # drop through to CBC
    if not use_highs:
        solver = pulp.PULP_CBC_CMD(msg=True,
                                   logPath=str(log_path),
                                   keepFiles=True)  # keeps .lp/.mps temp copies

    status = model.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "OptimalInfeasible"):  # CBC sometimes reports OptimalInfeasible
        # ------------- dump artefacts for post-mortem -----------------------
        lp_file  = _stamp("model", "lp")
        mps_file = _stamp("model", "mps")
        model.writeLP(str(lp_file))
        model.writeMPS(str(mps_file))
        manifest = {
            "pulp_status": pulp.LpStatus[status],
            "rows": len(model.constraints),
            "cols": len(model.variables()),
            "solver": "HiGHS" if use_highs else "CBC",
            "logfile": str(log_path),
            "lp_file": str(lp_file),
            "mps_file": str(mps_file)
        }
        (_stamp("manifest", "json")).write_text(json.dumps(manifest, indent=2))
        print("\n*** MODEL NOT OPTIMAL:", pulp.LpStatus[status],
              "***\nDiagnostic files written to", _DBGDIR.resolve(), "\n")
    else:
        # optional: delete log if you only want it when things go wrong
        pass
    return status
# -------------------------------------------------------------------------
# optimisation with PuLP + CBC
# -------------------------------------------------------------------------


def run_optimisation(p, price):
    N, dt_s, dt_h = p['num_time_points'], p['dt'], p['dt_hours']
    m = pulp.LpProblem("dc_cost_opt", pulp.LpMinimize)

    # ----------- variables ------------------------------------------------
    T_IT   = pulp.LpVariable.dicts("T_IT",   range(N), lowBound=14, upBound=75)
    T_Rack = pulp.LpVariable.dicts("T_Rack", range(N), lowBound=14, upBound=70)
    T_c    = pulp.LpVariable.dicts("T_c",    range(N),
        lowBound=14, upBound=p['T_cAisle_upper_limit_Celsius'])
    T_h    = pulp.LpVariable.dicts("T_h",    range(N), lowBound=14, upBound=80)
    E_TES  = pulp.LpVariable.dicts("E_TES",  range(N),
                                   lowBound=p['E_TES_min_kWh'],
                                   upBound=p['TES_capacity_kWh'])

    T_in   = pulp.LpVariable.dicts("T_in",   range(N), lowBound=14, upBound=30)
    P_chiller_HVAC = pulp.LpVariable.dicts("P_HVAC", range(N), lowBound=0)
    P_chiller_TES = pulp.LpVariable.dicts("P_cool", range(N), lowBound=0)
    q_cool = pulp.LpVariable.dicts("q_cool", range(N), lowBound=0)
    q_chiller_hvac = pulp.LpVariable.dicts("q_HVAC", range(N), lowBound=0)
    q_dis  = pulp.LpVariable.dicts("q_dis",  range(N), lowBound=0, upBound=p['TES_w_discharge_max'])
    q_ch = pulp.LpVariable.dicts("q_ch", range(N), lowBound=0, upBound=p['TES_w_charge_max'])
    # ----------- initial conditions --------------------------------------
    m += T_IT[0]   >= p['T_IT_initial_Celsius']
    m += T_Rack[0] >= p['T_Rack_initial_Celsius']
    m += T_c[0]    >= p['T_cAisle_initial']
    m += T_h[0]    >= p['T_hAisle_initial']
    m += E_TES[0]  >= p['TES_initial_charge_kWh']

    mcp = p['m_dot_air'] * p['c_p_air']

    # ----------- dynamics -------------------------------------------------
    for t in range(N-1):
        m += q_cool[t] == q_chiller_hvac[t] + q_dis[t]
        m += q_chiller_hvac[t] == P_chiller_HVAC[t] * p['COP_HVAC']
        m += q_ch[t] == P_chiller_TES[t] * p['COP_HVAC']

        m += T_in[t] == T_h[t] - q_cool[t] / mcp
        q_max_drop = (T_h[t] - p['T_cAisle_lower_limit_Celsius']) * mcp
        m += q_cool[t] <= q_max_drop

        m += T_IT[t+1] == T_IT[t] + dt_s * (
             (p['P_IT_heat_source']
              - p['G_conv'] * (T_IT[t] - T_Rack[t])) / p['C_IT'])

        m += T_Rack[t+1] == T_Rack[t] + dt_s * (
             (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_c[t]-T_Rack[t])
              + p['G_conv']*(T_IT[t]-T_Rack[t])) / p['C_Rack'])

        m += T_c[t+1] == T_c[t] + dt_s * (
             (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_in[t]-T_c[t])
              - p['G_cold']*(T_c[t]-p['T_out_Celsius'])) / p['C_cAisle'])

        m += T_h[t+1] == T_h[t] + dt_s * (
             (p['m_dot_air']*p['kappa']*p['c_p_air']*(T_Rack[t]-T_h[t]))
             / p['C_hAisle'])

        dE = (q_ch[t]*p['TES_charge_efficiency']
              - q_dis[t]/p['TES_discharge_efficiency']) * dt_h / 1000.0
        m += E_TES[t+1] == E_TES[t] + dE
        m += q_dis[t+1] - q_dis[t] <= p['TES_p_dis_ramp']  
        m += q_ch[t+1] - q_ch[t] <= p['TES_p_ch_ramp']
        m += q_cool[t+1] - q_cool[t] <= p['P_HVAC_ramp']  # ramping HVAC
        m += q_cool[t+1] <= p['P_HVAC_max_watts']

    m += q_cool[N-1] == q_chiller_hvac[N-1] + q_dis[N-1]
    m += T_in[N-1] == T_h[N-1] - q_cool[N-1]*p['COP_HVAC']/mcp
    Tmax_drop_N_minus_1 = (T_h[N-1] - 14) * mcp / p['COP_HVAC']
    m += q_cool[N-1] <= Tmax_drop_N_minus_1

    # --- cyclic-end conditions (PuLP) -----------------------------------------
    if CYCLE_TEMPERATURES:
        #m += T_IT[N-1]   == T_IT[0],   "cyc_T_IT"
        #m += T_Rack[N-1] == T_Rack[0], "cyc_T_Rack"
        #m += T_c[N-1]    == T_c[0],    "cyc_T_cAisle"
        #m += T_h[N-1]    == T_h[0],    "cyc_T_hAisle"
        pass
    if CYCLE_TES_ENERGY:
        m += E_TES[N-1]  == E_TES[0],  "cyc_E_TES"


    # ----------- objective ------------------------------------------------
    cost_expr = pulp.lpSum(
        (P_chiller_HVAC[t] + P_chiller_TES[t] + p['P_IT_heat_source']) * price[t] * dt_h
        for t in range(N))
    m += cost_expr

    # ----------- solve ----------------------------------------------------
    status = solve_lp(m, use_highs=False)
    if pulp.LpStatus[status] != "Optimal":
        print("Opt failed, status:", pulp.LpStatus[status])
        return None

    # ----------- collect results -----------------------------------------
    val = pulp.value
    res = {
        't':      np.arange(N) * dt_s / 60,  # minutes
        'T_IT':   [val(T_IT[t])   for t in range(N)],
        'T_Rack': [val(T_Rack[t]) for t in range(N)],
        'T_c':    [val(T_c[t])    for t in range(N)],
        'T_h':    [val(T_h[t])    for t in range(N)],
        'T_in':   [val(T_in[t])   for t in range(N)],
        'E_TES':  [val(E_TES[t])  for t in range(N)],
        'P_chiller_HVAC': [val(P_chiller_HVAC[t]) for t in range(N)],
        'P_chiller_TES': [val(P_chiller_TES[t]) for t in range(N)],
        'P_Total': [val(P_chiller_HVAC[t] + P_chiller_TES[t]) for t in range(N)],
        'q_chiller_hvac': [val(q_chiller_hvac[t])   for t in range(N)],
        'q_ch':   [val(q_ch[t])   for t in range(N)],
        'q_dis':  [val(q_dis[t])  for t in range(N)],
        'q_cool':  [val(q_cool[t])  for t in range(N)],
        'price':  price,
        'cost':   val(cost_expr),
    }
    return res


   


# -------------------------------------------------------------------------
# P L O T T I N G (unchanged)
# -------------------------------------------------------------------------
def plot_results(res, out_dir="static"):
    os.makedirs(out_dir, exist_ok=True)
    res = pd.DataFrame(res)
    #res = res.iloc[30:-30,:].reset_index(drop=True)
    #res['t'] = res['t'] - 30
    t = res['t']

    fig1, ax1 = plt.subplots(figsize=(10, 8), nrows=2, sharex=True)
    ax1[0].plot(t, res['T_IT'],  label="T_IT")
    ax1[0].plot(t, res['T_Rack'],label="T_Rack")
    ax1[0].plot(t, res['T_c'],   label="T_cAisle")
    ax1[0].plot(t, res['T_h'],   label="T_hAisle")
    ax1[0].plot(t, res['T_in'],  label="T_Air_in", linestyle='--')
    ax1[0].legend(loc='upper right'); ax1[0].set_ylabel("°C", fontsize=14); ax1[0].grid(True)


    ax1[1].plot(t, res['price'], color='green', label="Price (£/kWh)")
    ax1[1].set_ylabel("£ / kWh", fontsize=14)
    ax1[1].grid(True)
    ax1[1].set_xlabel("time [min]", fontsize=14)

    fig2, ax2 = plt.subplots(figsize=(10, 8), nrows=2, sharex=True)

    ax2[0].plot(t, np.array(res['P_Total'])/1000, label="Total Electrical Power kW", color='red')
    ax2[0].plot(t, np.array(res['P_chiller_HVAC'])/1000, label="Chiller,HVAC Electrical Power kW", linestyle='--', color='blue')
    ax2[0].plot(t, np.array(res['P_chiller_TES'])/1000, label="Chiller,TES Electrical Power kW", linestyle='--', color='green')

    ax2[0].legend(loc='upper right'); ax1[1].set_ylabel("kW", fontsize=14); ax1[1].grid(True)
    ax2[0].set_ylim(0, 300)
    ax2[0].legend(loc='upper left')
    ax2[0].grid(True)

    ax2[1].plot(t, np.array(res['q_cool'])/1000, label="Total Cooling Energy kW", color='red', linewidth=2)
    ax2[1].plot(t, np.array(res['q_dis'])/1000, label='TES Discharge (Thermal) kW', linestyle='--', color='blue')
    ax2[1].plot(t, np.array(res['q_chiller_hvac'])/1000, label="Chiller,HVAC (Thermal) kW", linestyle='--', color='green')

    ax2[1].legend(loc='upper right'); ax1[1].set_ylabel("kW", fontsize=14); ax1[1].grid(True)
    ax2[1].set_ylim(0, 1000)
    ax2[1].legend(loc='upper left')
    ax2[1].grid(True)

    """ax1_2 = ax1[2].twinx()
    ax1_2.plot(t, p['P_IT_heat_source']/1000, label="IT Power kW", linestyle='--', color='orange')
    ax1_2.set_ylabel("IT Power (kW)", fontsize=14)
    ax1_2.legend(loc='upper right')
    ax1[2].set_ylabel("$ / kWh", fontsize=14); ax1[2].set_xlabel("time [min]", fontsize=14)
    ax1[2].grid(True)"""

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "opt_plot_temps_power.png"))
    #plt.close(fig1)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t, res['E_TES'], label="E_TES kWh")
    ax3.set_ylim(0, 350)
    ax3.set_ylabel("kWh", fontsize=14); ax3.set_xlabel("time [min]", fontsize=14)
    ax33 = ax3.twinx()
    ax33.plot(t, np.array(res['q_ch'])/1000, label="TES Charge (Thermal) kW",   linestyle='--')
    ax33.plot(t, np.array(res['q_dis'])/1000,label="TES Discharge (Thermal) kW",linestyle='-.')
    #ax22.plot(t, (np.array(res['q_cool']))/1000, color='green', label="Total Cooling (Thermal) kW", linestyle=':')
    ax33.set_ylim(0, 600)
    ax33.set_ylabel("kW", fontsize=14)
    ax3.legend(loc='upper left'); ax33.legend(loc='upper right')
    ax3.grid(True)

    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "opt_plot_tes.png"))
    #fig1.show()
    #fig2.show()
    plt.show()
    #plt.close(fig2)

    print(f"Plots saved to ./{out_dir}/")


    res.to_csv(os.path.join(out_dir, "optimisation_results.csv"), index=False)


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
if __name__ == "__main__":

    p = setup_optimisation_parameters()
    tariff = generate_tariff(p['num_time_points'], p['dt'])
    results = run_optimisation(p, tariff)
    if results:
        print(f"Optimal cost: ${results['cost']:.2f}")
        plot_results(results)
