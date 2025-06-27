import pandas as pd
import pulp
import matplotlib.pyplot as plt  # Added for charting
from it_parameters import get_load_and_price_profiles

def create_and_save_charts(results_df: pd.DataFrame):
    """
    Generates and saves charts based on the results DataFrame.
    """
    print("Generating and saving charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    time_slots = results_df['Time_Slot']

    # --- Figure 1: Power Consumption and Energy Price ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Power Consumption
    ax1.plot(time_slots, results_df['Total_Nominal_Power_kW'], label='Nominal Total Power', linestyle='--', color='gray', alpha=0.8)
    ax1.plot(time_slots, results_df['Total_Optimized_Power_kW'], label='Optimized Total Power', color='crimson')
    ax1.set_ylabel('Total Power from Grid (kW)')
    ax1.set_title('Optimized vs. Nominal Power Consumption')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Energy Price
    ax2.plot(time_slots, results_df['Spot_Price_GBP_per_MWh'], label='Energy Price', color='royalblue')
    ax2.set_xlabel(f"Time Slot ({results_df['Time_Slot'].iloc[1] - results_df['Time_Slot'].iloc[0]} min intervals)")
    ax2.set_ylabel('Price (£/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    fig1.tight_layout()
    fig1.savefig('power_and_price.png')

    # --- Figure 2: UPS (Battery) Performance ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: UPS Energy Level
    ax3.plot(time_slots, results_df['E_UPS_kWh'], label='UPS Energy Level (SoC)', color='purple')
    ax3.set_ylabel('Energy Stored (kWh)')
    ax3.set_title('UPS (Battery) Performance')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Subplot 2: UPS Charge/Discharge Power
    # We plot discharge as negative power for better visualization of energy flow
    ax4.bar(time_slots, results_df['P_UPS_Charge_kW'], width=1.0, label='Charge Power', color='mediumseagreen')
    ax4.bar(time_slots, -results_df['P_UPS_Discharge_kW'], width=1.0, label='Discharge Power', color='coral')
    ax4.axhline(0, color='black', linewidth=0.5) # Zero line
    ax4.set_xlabel(f"Time Slot ({results_df['Time_Slot'].iloc[1] - results_df['Time_Slot'].iloc[0]} min intervals)")
    ax4.set_ylabel('Charge/Discharge Power (kW)')
    ax4.legend(loc='upper left')
    ax4.grid(True)

    fig2.tight_layout()
    fig2.savefig('ups_performance.png')
    
    print("Charts saved as 'power_and_price.png' and 'ups_performance.png'.")
    # plt.show() # Uncomment to display charts interactively after running

def solve_datacenter_optimization_pulp():
    # ==============================================================================
    # DATA AND PARAMETERS
    # ==============================================================================
    T_SLOTS = range(1, 97)
    TEXT_SLOTS = range(1, 109)
    K_TRANCHES = range(1, 5)

    slotDurationHours = 0.25
    maxCpuUsage = 1.0
    idlePower_kW = 166.7
    maxPower_kW = 1000.0

    eta_ch = 0.82
    eta_disch = 0.92
    E_nom_kWh = 600.0
    SoC_min = 0.5
    SoC_max = 1.0
    E_start_kWh = 600.0
    P_max_ch_kW = 270.0
    P_max_disch_kW = 2700.0
    P_min_ch_kW = 40.0
    P_min_disch_kW = 100.0

    E_min_kWh = SoC_min * E_nom_kWh
    E_max_kWh = SoC_max * E_nom_kWh

    trancheMaxDelay = {1: 2, 2: 4, 3: 8, 4: 12}

    inflexibleLoadProfile_TEXT, baseFlexibleLoadProfile_TEXT, baseFlexibleLoadProfile_T,\
    shiftabilityProfile, spotPrice_data =  get_load_and_price_profiles(TEXT_SLOTS, T_SLOTS)

    spotPrice = pd.Series(spotPrice_data, index=TEXT_SLOTS)

    Rt = baseFlexibleLoadProfile_T * slotDurationHours
    Pt_IT_nom_TEXT = idlePower_kW + (maxPower_kW - idlePower_kW) * (inflexibleLoadProfile_TEXT + baseFlexibleLoadProfile_TEXT)

    # ==============================================================================
    # MODEL DEFINITION (PuLP)
    # ==============================================================================
    model = pulp.LpProblem("Data_Center_Cost_Optimization", pulp.LpMinimize)

    # --- Decision variables ---
    totalCpuUsage = pulp.LpVariable.dicts("totalCpuUsage", TEXT_SLOTS, lowBound=0, cat='Continuous')
    Pgrid_IT = pulp.LpVariable.dicts("Pgrid_IT", TEXT_SLOTS, lowBound=0, cat='Continuous')
    Pgrid_cooling = pulp.LpVariable.dicts("Pgrid_cooling", TEXT_SLOTS, lowBound=0, cat='Continuous')
    Pgrid_OD = pulp.LpVariable.dicts("Pgrid_OD", TEXT_SLOTS, lowBound=0, cat='Continuous')
    PUPS_ch = pulp.LpVariable.dicts("PUPS_ch", TEXT_SLOTS, lowBound=0, cat='Continuous')
    PUPS_disch = pulp.LpVariable.dicts("PUPS_disch", TEXT_SLOTS, lowBound=0, cat='Continuous')
    EUPS = pulp.LpVariable.dicts("EUPS", TEXT_SLOTS, lowBound=E_min_kWh, upBound=E_max_kWh, cat='Continuous')
    Pt_IT_act = pulp.LpVariable.dicts("Pt_IT_act", TEXT_SLOTS, lowBound=0, cat='Continuous')
    Pt_IT_act_ext = pulp.LpVariable.dicts("Pt_IT_act_ext", TEXT_SLOTS, lowBound=None, cat='Continuous')
    Pt_IT_total = pulp.LpVariable.dicts("Pt_IT_total", TEXT_SLOTS, lowBound=0, cat='Continuous')
    zt_ch = pulp.LpVariable.dicts("zt_ch", TEXT_SLOTS, cat='Binary')
    zt_disch = pulp.LpVariable.dicts("zt_disch", TEXT_SLOTS, cat='Binary')
    ut_ks_idx = [(t, k, s) for t in T_SLOTS for k in K_TRANCHES for s in TEXT_SLOTS if s >= t and s <= t + trancheMaxDelay[k]]
    ut_ks = pulp.LpVariable.dicts("ut_ks", ut_ks_idx, lowBound=0, cat='Continuous')

    # --- Objective function ---
    model += pulp.lpSum(
        slotDurationHours * (Pgrid_IT[s] + Pgrid_cooling[s] + Pgrid_OD[s] + PUPS_ch[s]) * (spotPrice[s] / 1000)
        for s in TEXT_SLOTS
    ), "Total_Energy_Cost"

    # --- Constraints ---
    # Job Completion Constraint
    for t in T_SLOTS:
        for k in K_TRANCHES:
            model += pulp.lpSum(ut_ks[(t, k, s)] * slotDurationHours for s in TEXT_SLOTS if (t, k, s) in ut_ks_idx) == Rt[t] * shiftabilityProfile.get((t, k), 0), f"JobCompletion_CON_{t}_{k}"

    # Total CPU Usage and Capacity
    for s in TEXT_SLOTS:
        flexible_usage = pulp.lpSum(ut_ks[(t, k, s_idx)] for t, k, s_idx in ut_ks_idx if s_idx == s)
        model += totalCpuUsage[s] == inflexibleLoadProfile_TEXT[s] + flexible_usage, f"Total_CPU_Usage_EQ_{s}"
        model += totalCpuUsage[s] <= maxCpuUsage, f"CpuCapacity_CON_{s}"

    # IT Power Definitions
    for s in TEXT_SLOTS:
        if s in T_SLOTS:
            model += Pt_IT_act[s] == idlePower_kW + (maxPower_kW - idlePower_kW) * totalCpuUsage[s], f"Define_IT_act_EQ_{s}"
            model += Pt_IT_act_ext[s] == 0, f"Fix_IT_act_ext_in_T_{s}"
        else:
            model += Pt_IT_act[s] == 0, f"Fix_IT_act_in_Ext_{s}"
            nominal_power_in_ext = Pt_IT_nom_TEXT[s]
            model += Pt_IT_act_ext[s] >= (idlePower_kW + (maxPower_kW - idlePower_kW) * totalCpuUsage[s]) - nominal_power_in_ext, f"Define_IT_act_ext_EQ_{s}"
        model += Pt_IT_total[s] == Pt_IT_act[s] + Pt_IT_act_ext[s], f"Define_IT_total_EQ_{s}"

    # Power Balance Equations
    for s in TEXT_SLOTS:
        model += Pt_IT_total[s] == Pgrid_IT[s] + PUPS_disch[s], f"PowerBalance_IT_EQ_{s}"
        model += Pgrid_cooling[s] == Pt_IT_total[s] * 0.4, f"PowerBalance_Cooling_EQ_{s}"
        model += Pgrid_OD[s] == Pt_IT_total[s] * 0.1, f"PowerBalance_OD_EQ_{s}"

    # UPS Equations
    for s in TEXT_SLOTS:
        prev_energy = E_start_kWh if s == 1 else EUPS[s-1]
        charge = eta_ch * PUPS_ch[s] * slotDurationHours
        discharge = (PUPS_disch[s] / eta_disch) * slotDurationHours
        model += EUPS[s] == prev_energy + charge - discharge, f"UPS_EnergyBalance_EQ_{s}"
        model += PUPS_ch[s] <= zt_ch[s] * P_max_ch_kW, f"UPS_Charge_Limit_EQ_{s}"
        model += PUPS_ch[s] >= zt_ch[s] * P_min_ch_kW, f"UPS_Min_Charge_Limit_EQ_{s}"
        model += PUPS_disch[s] <= zt_disch[s] * P_max_disch_kW, f"UPS_Discharge_Limit_EQ_{s}"
        model += PUPS_disch[s] >= zt_disch[s] * P_min_disch_kW, f"UPS_Min_Discharge_Limit_EQ_{s}"
        model += zt_ch[s] + zt_disch[s] <= 1, f"UPS_Simultaneous_Op_CON_{s}"

    # Final condition for UPS
    model += EUPS[max(TEXT_SLOTS)] == E_start_kWh, "Final_UPS_Energy_Level"

    # ==============================================================================
    # MODEL AND SOLVE
    # ==============================================================================
    print("Solver starting with PuLP...")
    solver = pulp.PULP_CBC_CMD(msg=True, gapRel=0.01)
    model.solve(solver)

    if pulp.LpStatus[model.status] == 'Optimal':
        print("Optimization successful!")
    else:
        print(f"Solver did not find an optimal solution. Status: {pulp.LpStatus[model.status]}")
        return None # Return None if not solved

    # ==============================================================================
    # POST-PROCESSING AND DISPLAY RESULTS
    # ==============================================================================
    TotalNominalPowerProfile = Pt_IT_nom_TEXT * (1 + 0.4 + 0.1)
    totalCost_nominal = sum(
        slotDurationHours * Pt_IT_nom_TEXT[t] * (1 + 0.4 + 0.1) * (spotPrice[t] / 1000) 
        for t in T_SLOTS
    )
    totalNominalPower = sum(
        slotDurationHours * Pt_IT_nom_TEXT[t] * (1 + 0.4 + 0.1)
        for t in T_SLOTS
    )
    optimized_cost = pulp.value(model.objective)
    costSaving_abs = totalCost_nominal - optimized_cost
    costReductionRate = (costSaving_abs / totalCost_nominal) * 100 if totalCost_nominal else 0

    print("\n" + "="*50)
    print("--- Optimization Results (PuLP) ---")
    print(f"Optimized Total Cost: {optimized_cost:,.2f} GBP")
    print("\n--- Baseline (No Opt.) ---")
    print(f"Nominal Total Cost: {totalCost_nominal:,.2f} GBP")
    print("\n--- Savings ---")
    print(f"Absolute Cost Saving: {costSaving_abs:,.2f} GBP")
    print(f"Relative Cost Saving: {costReductionRate:.2f} %")
    print(f"Total Nominal Energy (primary horizon): {totalNominalPower:,.2f} kWh")
    print("="*50 + "\n")

    # ==============================================================================
    # --- EXPORT RESULTS TO CSV ---
    # ==============================================================================
    TotalOptimizedPowerProfile = {s: (Pgrid_IT[s].value() + Pgrid_cooling[s].value() + Pgrid_OD[s].value() + PUPS_ch[s].value()) for s in TEXT_SLOTS}
    
    results_data = {
        'Time_Slot': list(TEXT_SLOTS),
        'Inflexible_Load': [inflexibleLoadProfile_TEXT[s] for s in TEXT_SLOTS],
        'Base_Flexible_Load': [baseFlexibleLoadProfile_TEXT[s] for s in TEXT_SLOTS],
        'Optimized_Total_CPU': [totalCpuUsage[s].value() for s in TEXT_SLOTS],
        'Nominal_IT_Power_kW': [Pt_IT_nom_TEXT[s] for s in TEXT_SLOTS],
        'Optimized_Total_IT_Power_kW': [Pt_IT_total[s].value() for s in TEXT_SLOTS],
        'P_Grid_IT_kW': [Pgrid_IT[s].value() for s in TEXT_SLOTS],
        'P_Grid_Cooling_kW': [Pgrid_cooling[s].value() for s in TEXT_SLOTS],
        'P_Grid_OD_kW': [Pgrid_OD[s].value() for s in TEXT_SLOTS],
        'P_UPS_Charge_kW': [PUPS_ch[s].value() for s in TEXT_SLOTS],
        'P_UPS_Discharge_kW': [PUPS_disch[s].value() for s in TEXT_SLOTS],
        'E_UPS_kWh': [EUPS[s].value() for s in TEXT_SLOTS],
        'Total_Nominal_Power_kW': [TotalNominalPowerProfile[s] for s in TEXT_SLOTS],
        'Total_Optimized_Power_kW': [TotalOptimizedPowerProfile[s] for s in TEXT_SLOTS],
        'Spot_Price_GBP_per_MWh': [spotPrice[s] for s in TEXT_SLOTS]
    }

    results_df = pd.DataFrame(results_data)
    results_df.to_csv('AllResults_PuLP.csv', index=False, float_format='%.4f')
    print("Results have been successfully exported to 'AllResults_PuLP.csv'")
    
    return results_df # Return the DataFrame for charting

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    results_df = solve_datacenter_optimization_pulp()
    if results_df is not None:
        create_and_save_charts(results_df)