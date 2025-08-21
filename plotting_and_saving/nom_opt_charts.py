import matplotlib.pyplot as plt 

def gen_charts(df, time_slots_ext, IMAGE_DIR):
    # --- Figure 2: TES Performance ---
    fig2, (ax2_tes, ax3_tes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax2_tes.plot(time_slots_ext, df['E_TES_kWh'], label='TES Energy Level', color='mediumblue')
    ax2_tes.set_ylabel('Energy (kWh)')
    ax2_tes.set_title('Thermal Energy Storage (TES) Performance')
    ax2_tes.legend()
    ax2_tes.grid(True)
    ax3_tes.plot(time_slots_ext, df['Q_Charge_TES_Watts'], label='Charge Heat Flow', color='green')
    ax3_tes.plot(time_slots_ext, df['Q_Discharge_TES_Watts'], label='Discharge Heat Flow', color='orange')
    ax3_tes.set_xlabel('Time Slot')
    ax3_tes.set_ylabel('Heat Flow (Watts)')
    ax3_tes.legend()
    ax3_tes.grid(True)
    fig2.tight_layout()
    fig2.savefig(IMAGE_DIR / 'tes_performance.png')
    print("✅ TES performance chart saved.")

    # --- Figure 3: Data Centre Temperatures ---
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(time_slots_ext, df['T_IT_Celsius'], label='IT Equipment Temp')
    ax4.plot(time_slots_ext, df['T_Rack_Celsius'], label='Rack Temp')
    ax4.plot(time_slots_ext, df['T_HotAisle_Celsius'], label='Hot Aisle Temp')
    ax4.plot(time_slots_ext, df['T_ColdAisle_Celsius'], label='Cold Aisle Temp')
    ax4.set_xlabel('Time Slot')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Data Centre Temperatures')
    ax4.legend()
    ax4.grid(True)
    fig3.tight_layout()
    fig3.savefig(IMAGE_DIR / 'dc_temperatures.png')
    print("✅ Data centre temperatures chart saved.")

    # --- Figure 4: Cooling System Power Components ---
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(time_slots_ext, df['P_Grid_Cooling_kW'], label='Total Cooling Power (kW)', color='blue')
    ax5.plot(time_slots_ext, df['P_Chiller_HVAC_kW'], label='Chiller HVAC Power (kW)', color='red')
    ax5.plot(time_slots_ext, df['P_Chiller_TES_kW'], label='Chiller TES Power (kW)', color='green')
    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Cooling System Power Components')
    ax5.legend()
    ax5.grid(True)
    fig4.tight_layout()
    fig4.savefig(IMAGE_DIR / 'cooling_power_components.png')
    print("✅ Cooling system power components chart saved.")

    # --- Figure 5: Thermal Cooling Power (q) ---
    fig5, ax6 = plt.subplots(figsize=(12, 6))
    df['Q_Chiller_Direct_Watts'] = df['Q_Cool_Total_Watts'] - df['Q_Discharge_TES_Watts']
    ax6.stackplot(time_slots_ext, df['Q_Chiller_Direct_Watts'], df['Q_Discharge_TES_Watts'],
                  labels=['Cooling from Chiller (Direct)', 'Cooling from TES'],
                  colors=['green', 'blue'])
    ax6.plot(time_slots_ext, df['P_IT_Total_kW'] * 1000, label='Total Cooling Demand (Heat from IT)', color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Time Slot')
    ax6.set_ylabel('Thermal Power (Watts)')
    ax6.set_title('Cooling Power (q) by Source')
    ax6.legend(loc='upper left')
    ax6.grid(True)
    fig5.tight_layout()
    fig5.savefig(IMAGE_DIR / 'thermal_cooling_power.png')
    print("✅ Thermal cooling power chart saved.")

    # --- Figure 6: Cumulative Cost ---
    fig6, ax7 = plt.subplots(figsize=(12, 7))
    ax7.plot(time_slots_ext, df['Nominal_Cost'].cumsum(), label='Cumulative Nominal Cost', color='crimson', linewidth=2)
    ax7.set_xlabel('Time Slot')
    ax7.set_ylabel('Cumulative Cost (£)')
    ax7.set_title('Cumulative Nominal Energy Cost')
    ax7.legend()
    ax7.grid(True)
    fig6.tight_layout()
    fig6.savefig(IMAGE_DIR / 'cumulative_nominal_cost.png')
    print("✅ Cumulative cost chart saved.")