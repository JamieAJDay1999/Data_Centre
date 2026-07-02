import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DC Flex: Advanced Methodology Visualiser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "PRO" LOOK ---
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric Cards - Glassmorphism */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 24px !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #00ff9d !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    .highlight-text {
        color: #00d4ff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA GENERATION & SIMULATION ENGINE ---

def generate_simulation_data(it_cap, ups_cap, tes_cap, volatility):
    """
    Generates the 24h profile based on the paper's methodology.
    Mimics the inputs from Table I and IV.
    """
    hours = np.arange(24)
    
    # 1. Price Profile (Table IV)
    # Base prices typical of UK day-ahead market
    base_prices = np.array([60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70])
    # Apply volatility multiplier
    prices = base_prices * volatility
    
    # 2. Base IT Load (Figure 2)
    # Diurnal pattern: Low at night, ramps up morning, high evening
    base_it_pct = np.array([0.6, 0.55, 0.5, 0.45, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.85, 0.85, 0.8, 0.8, 0.75, 0.75, 0.7, 0.65, 0.6, 0.65, 0.75, 0.8, 0.7, 0.65])
    base_it_load = base_it_pct * it_cap
    
    # 3. Cooling Load (Thermodynamic approximation)
    # Base Cooling is roughly 30-40% of IT load (PUE ~1.3-1.4)
    base_cooling_load = base_it_load * 0.35 
    
    # 4. Aux Load (Constant)
    aux_load = 53.0 
    
    # 5. Optimization Logic (Heuristic based on Scenario 2)
    # Identify cheapest 4 hours for charging, most expensive 3 hours for discharging
    sorted_indices = np.argsort(prices)
    cheapest_hours = sorted_indices[:4]
    expensive_hours = sorted_indices[-3:]
    
    # -- UPS Dispatch --
    ups_power = np.zeros(24)
    ups_soc = np.zeros(24)
    current_ups = ups_cap * 0.5 # Start at 50%
    
    # Heuristic: Charge at cheapest, Discharge at most expensive
    charge_rate = 270 # kW
    discharge_rate = 270 # kW
    
    for h in range(24):
        if h in cheapest_hours and current_ups < ups_cap:
            actual_charge = min(charge_rate, ups_cap - current_ups)
            ups_power[h] = actual_charge # Positive = Load
            current_ups += actual_charge
        elif h in expensive_hours and current_ups > (ups_cap * 0.2): # Min SoC 20%
            actual_discharge = min(discharge_rate, current_ups - (ups_cap*0.2))
            ups_power[h] = -actual_discharge # Negative = Generation
            current_ups -= actual_discharge
        ups_soc[h] = current_ups

    # -- TES Dispatch --
    # Similar logic: Pre-cool (charge) at low price, stop chillers (discharge) at high price
    tes_thermal_power = np.zeros(24)
    tes_elec_impact = np.zeros(24)
    current_tes = 0
    cop = 5.0 #
    
    for h in range(24):
        if h in cheapest_hours and current_tes < tes_cap:
            # Charging TES means running chillers HARDER -> Increase Load
            thermal_charge = 300 # kW thermal
            tes_thermal_power[h] = thermal_charge
            current_tes += thermal_charge
        elif h in expensive_hours and current_tes > 0:
            # Discharging TES means turning chillers OFF -> Reduce Load
            thermal_discharge = min(300, current_tes)
            tes_thermal_power[h] = -thermal_discharge
            current_tes -= thermal_discharge
            
        # Electric impact: Charging adds load, Discharging removes cooling load
        tes_elec_impact[h] = tes_thermal_power[h] / cop

    # -- IT Shifting --
    # Move 15% of load from expensive hours to cheapest hours
    shifted_it_load = base_it_load.copy()
    shiftable_volume = 0
    
    for h in expensive_hours:
        amount = shifted_it_load[h] * 0.15 # 15% shiftable
        shifted_it_load[h] -= amount
        shiftable_volume += amount
        
    # Distribute to cheapest
    per_hour_add = shiftable_volume / len(cheapest_hours)
    for h in cheapest_hours:
        shifted_it_load[h] += per_hour_add

    # Total Optimised Load
    # Base Components (IT, Cool, Aux) + Changes (UPS, TES, Shifting)
    # Note: Cooling load changes based on IT load changes too, simplified here.
    opt_core_load = shifted_it_load + (shifted_it_load * 0.35) + aux_load
    total_opt_load = opt_core_load + ups_power + tes_elec_impact
    
    base_total_load = base_it_load + base_cooling_load + aux_load

    return {
        "hours": hours,
        "prices": prices,
        "base_load": base_total_load,
        "opt_load": total_opt_load,
        "opt_core_load": opt_core_load,
        "ups_power": ups_power,
        "tes_elec": tes_elec_impact,
        "ups_soc": ups_soc,
        "tes_soc": current_tes # End state
    }

# --- SIDEBAR ---
st.sidebar.title("🛠️ Simulation Controls")
st.sidebar.info("Adjust parameters from Table I.")

it_capacity = st.sidebar.slider("IT Capacity (kW)", 500, 2000, 1000, 100)
ups_capacity = st.sidebar.slider("UPS Battery (kWh)", 100, 2000, 600, 100)
tes_capacity = st.sidebar.slider("TES Capacity (kWh-th)", 100, 2000, 1000, 100)
volatility = st.sidebar.slider("Market Price Volatility", 0.5, 2.5, 1.0, help="Simulate higher renewable penetration (more negative/positive spikes).")

# Run Simulation
data = generate_simulation_data(it_capacity, ups_capacity, tes_capacity, volatility)

# --- MAIN CONTENT ---

st.title("⚡ Data Centre Flexibility Visualiser")
st.markdown("""
**Based on:** *Characterisation and Quantification of Data Centre Flexibility for Power System Support* by Takci et al.
This tool visualises how a DC optimizes costs and calculates its flexibility duration envelope ($τ$).
""")

# --- KPI ROW ---
col1, col2, col3, col4 = st.columns(4)
base_cost = np.sum(data['base_load'] * data['prices'] / 1000)
opt_cost = np.sum(data['opt_load'] * data['prices'] / 1000)
savings = base_cost - opt_cost
savings_pct = (savings / base_cost) * 100
peak_red = np.max(data['base_load']) - np.max(data['opt_load'])

with col1:
    st.metric("Baseline Cost (Day)", f"£{base_cost:,.2f}")
with col2:
    st.metric("Optimised Cost (Day)", f"£{opt_cost:,.2f}")
with col3:
    st.metric("Net Savings", f"£{savings:,.2f}", f"{savings_pct:.1f}%")
with col4:
    st.metric("Peak Load Reduction", f"{peak_red:.0f} kW", help="Reduction in maximum power draw from grid")

st.markdown("---")

# --- TABS ---
tab_opt, tab_flex, tab_sim = st.tabs(["📈 Operational Optimisation", "🔥 Flexibility Heatmap", "🎮 Grid Request Sim"])

with tab_opt:
    st.subheader("Scenario 2: Cost Minimisation Results")
    st.markdown("The chart below replicates **Figure 5**. It shows how the DC shifts load away from price spikes.")
    
    # Create Figure 5 Replication
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Baseline Trace (Dotted)
    fig.add_trace(go.Scatter(
        x=data['hours'], y=data['base_load'],
        name="Baseline Load (Passive)",
        line=dict(color='gray', width=2, dash='dot')
    ), secondary_y=False)

    # 2. Optimised Core Load (Filled Area - Blue)
    fig.add_trace(go.Scatter(
        x=data['hours'], y=data['opt_core_load'],
        name="Core DC Load (IT+Cool)",
        stackgroup='one',
        line=dict(width=0),
        fillcolor='rgba(0, 119, 255, 0.6)'
    ), secondary_y=False)

    # 3. Charging Loads (Stacked on top - Green/Orange)
    ups_charge = np.maximum(data['ups_power'], 0)
    tes_charge = np.maximum(data['tes_elec'], 0)
    
    fig.add_trace(go.Scatter(
        x=data['hours'], y=ups_charge,
        name="UPS Charging",
        stackgroup='one',
        line=dict(width=0),
        fillcolor='rgba(0, 255, 157, 0.5)'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=data['hours'], y=tes_charge,
        name="TES Charging",
        stackgroup='one',
        line=dict(width=0),
        fillcolor='rgba(255, 166, 0, 0.6)'
    ), secondary_y=False)

    # 4. Discharging (Negative bars below axis)
    ups_discharge = np.minimum(data['ups_power'], 0)
    tes_discharge = np.minimum(data['tes_elec'], 0)
    
    # Combine discharge for cleaner look or keep separate? Let's stack them visually in negative
    fig.add_trace(go.Bar(
        x=data['hours'], y=ups_discharge,
        name="UPS Discharge",
        marker_color='rgba(0, 255, 157, 0.9)',
        base=0
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        x=data['hours'], y=tes_discharge,
        name="TES Discharge",
        marker_color='rgba(255, 166, 0, 0.9)',
        base=ups_discharge # Stack visually below UPS discharge
    ), secondary_y=False)

    # 5. Price Line (Secondary Axis)
    fig.add_trace(go.Scatter(
        x=data['hours'], y=data['prices'],
        name="Grid Price (£/MWh)",
        line=dict(color='#ff4b4b', width=3),
        mode='lines'
    ), secondary_y=True)

    # Layout Polish
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Time of Day", 
            tickmode='linear', 
            tick0=0, dtick=2, 
            showgrid=False,
            color='white'
        ),
        yaxis=dict(
            title="Power (kW)", 
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis2=dict(
            title="Price (£/MWh)", 
            overlaying='y', 
            side='right',
            color='#ff4b4b'
        ),
        legend=dict(orientation="h", y=1.1, font=dict(color='white')),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Annotations for insight
    max_price_idx = np.argmax(data['prices'])
    fig.add_annotation(
        x=max_price_idx, y=data['prices'][max_price_idx],
        xref="x", yref="y2",
        text="Peak Price: Discharge Assets",
        showarrow=True, arrowhead=2,
        ax=0, ay=-40,
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Observation:** Notice the **Asymmetry** mentioned in the paper. 
    The system consumes extra power (Green/Orange areas) during the cheap night hours to charge buffers. 
    During the price spike (Red line peak), it discharges (bars below zero) to drastically reduce grid import.
    """)

with tab_flex:
    st.subheader("Scenario 3: Duration-Aware Flexibility Assessment")
    st.markdown("""
    This heatmap replicates **Figure 6**. It answers: 
    *"If the grid asks for X kW of flexibility at hour T, how long can we sustain it?"*
    """)
    
    # Generate Heatmap Data (Replicating the binary search logic roughly)
    flex_mags = np.arange(-500, 550, 50) # kW
    heatmap_z = np.zeros((len(flex_mags), 24))
    
    # Constants for flex calculation
    max_ups_dis = 270
    max_ups_chg = 270
    max_tes_chg = 200
    
    for t_idx in range(24):
        # State at start of flexibility event
        base_load_t = data['opt_load'][t_idx]
        soc_ups_t = data['ups_soc'][t_idx]
        
        for m_idx, mag in enumerate(flex_mags):
            duration = 0
            
            # UPWARD FLEX (Reducing Load, Negative Mag)
            if mag < 0:
                req_cut = abs(mag)
                # Cap: Can't cut more than current load + max battery discharge
                # Paper says we can cut IT load by shifting, plus discharge battery
                feasible_cut = base_load_t * 0.4 + max_ups_dis # Approx 40% shiftable + batt
                
                if req_cut <= feasible_cut:
                    # Duration limited by Battery Energy
                    energy_avail = soc_ups_t
                    power_from_batt = max(0, req_cut - (base_load_t * 0.2)) # Assume 20% comes from IT shift first
                    
                    if power_from_batt <= 0:
                        duration = 4.0 # Limited only by IT shift window
                    else:
                        duration = min(4.0, energy_avail / power_from_batt) if power_from_batt > 0 else 4.0
            
            #DOWNWARD FLEX (Increasing Load, Positive Mag)
            elif mag > 0:
                req_add = mag
                # Cap: Charger limits
                feasible_add = max_ups_chg + max_tes_chg
                
                if req_add <= feasible_add:
                    # Duration limited by "Headroom" in storage
                    space_ups = ups_capacity - soc_ups_t
                    cop = 5#
                    space_tes = (tes_capacity - 0) / cop # Simplified TES state
                    
                    # Heuristic: fill UPS first, then TES
                    duration = min(4.0, (space_ups + space_tes) / req_add)
            
            heatmap_z[m_idx, t_idx] = duration

    # Plotting Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_z,
        x=data['hours'],
        y=flex_mags,
        colorscale='Viridis',
        colorbar=dict(title=dict(text="Max Duration (hrs)", font=dict(color='white')), tickfont=dict(color='white')),
        hovertemplate="Hour: %{x}:00<br>Flex: %{y} kW<br>Duration: %{z:.1f} hrs<extra></extra>"
    ))
    
    fig_heat.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Start Time (Hour)", color='white', dtick=1),
        yaxis=dict(title="Flexibility Magnitude (kW) <br>(-ve = Reduce Load / +ve = Increase Load)", color='white'),
        title=dict(text="Flexibility Duration Envelope (Fig. 6)", font=dict(color='white'))
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.warning("""
    **Key Insight (Asymmetry):** Look at the evening peak (17:00-19:00). 
    * **Upward Flex (Top half)** is weak/short because we are already discharging batteries to save money (Base Case).
    * **Downward Flex (Bottom half)** is strong/long because batteries are empty and ready to absorb grid surplus.
    """)

with tab_sim:
    st.subheader("🎮 Interactive Grid Request Simulator")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 1. Grid Signal")
        sim_hour = st.selectbox("Time of Event", range(24), index=18, format_func=lambda x: f"{x}:00")
        sim_req = st.number_input("Requested Power Deviation (kW)", min_value=-1000, max_value=1000, value=-250, step=50)
        
        st.markdown("### 2. System State")
        curr_load = data['opt_load'][sim_hour]
        curr_soc = data['ups_soc'][sim_hour]
        st.metric("Current Grid Draw", f"{curr_load:.0f} kW")
        st.metric("Battery SoC", f"{(curr_soc/ups_capacity)*100:.0f}%")
        
    with c2:
        st.markdown("### 3. Feasibility & Stack")
        
        # Determine response logic (Visualizing Fig 7/8)
        response_breakdown = {}
        
        remaining_req = abs(sim_req)
        feasible = True
        msg = ""
        
        if sim_req < 0: # UPWARD (Cut Load)
            # 1. Shift IT (First line of defense)
            it_shift_cap = 150 # kW
            it_act = min(remaining_req, it_shift_cap)
            response_breakdown["Defer IT Workload"] = it_act
            remaining_req -= it_act
            
            # 2. Discharge Battery
            if remaining_req > 0:
                batt_power_cap = 270
                batt_energy_cap = curr_soc # kWh available
                # Assuming 1 hour duration request for simplicity of stack viz
                batt_act = min(remaining_req, batt_power_cap)
                if batt_energy_cap < batt_act: 
                    batt_act = batt_energy_cap # Energy limited
                    feasible = False
                    msg = "Limited by Battery Energy"
                
                response_breakdown["Discharge UPS"] = batt_act
                remaining_req -= batt_act
                
            # 3. Reduce Cooling (TES)
            if remaining_req > 0:
                cool_act = min(remaining_req, 100) # Residual thermal inertia
                response_breakdown["Utilise Thermal Inertia"] = cool_act
                remaining_req -= cool_act

        else: # DOWNWARD (Absorb Load)
            # 1. Charge Battery
            space = ups_capacity - curr_soc
            batt_act = min(remaining_req, 270)
            if space < batt_act: batt_act = space
            response_breakdown["Charge UPS"] = batt_act
            remaining_req -= batt_act
            
            # 2. Charge TES
            if remaining_req > 0:
                tes_act = min(remaining_req, 200)
                response_breakdown["Charge TES"] = tes_act
                remaining_req -= tes_act

        # Check feasibility
        if remaining_req > 10: # Tolerance
            st.error(f"❌ **Infeasible**: Shortfall of {remaining_req:.0f} kW")
        else:
            st.success(f"✅ **Feasible**: Request Met")

        # Visualize Stack
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            x=list(response_breakdown.keys()),
            y=list(response_breakdown.values()),
            marker_color=['#00d4ff', '#00ff9d', '#ffaa00'],
            text=list(response_breakdown.values()),
            textposition='auto'
        ))
        fig_stack.update_layout(
            title="Flexibility Asset Stack",
            yaxis_title="Power Contribution (kW)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_stack, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Simulation Methodology derived from: Takci, M. T., Day, J., & Qadrdan, M. 'Characterisation and Quantification of Data Centre Flexibility for Power System Support'. Journal of Latex Class Files, Oct 2025.")