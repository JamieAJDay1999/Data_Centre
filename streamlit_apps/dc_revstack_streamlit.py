import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="GB Data Centre Revenue Stacking | Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔋"
)

# Custom CSS for "Data Centre Cyberpunk" aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00f2ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00f2ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stAlert {
        background-color: #1f2937;
        color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: EXECUTIVE SUMMARY ---
with st.sidebar:
    st.title("🔋 Revenue Stacking")
    st.markdown("### Based on Day & Qadrdan (2025)")
    st.info("This tool simulates the techno-economic analysis of BESS-equipped data centres in the GB market.")
    
    st.markdown("---")
    st.markdown("**Key Paper Concepts:**")
    st.markdown("1. **Transition:** GB Grid moving to Net-Zero & Low Inertia.")
    st.markdown("2. **Opportunity:** Data Centres (DCs) have flexible IT loads & Batteries[cite: 12].")
    st.markdown("3. **Solution:** Stacking revenue from Wholesale Arbitrage + Frequency Services (Dynamic Containment)[cite: 16, 51].")
    
    st.markdown("---")
    st.caption("Simulation created by Gemini")

# --- DATA GENERATION ENGINE ---
def generate_market_data():
    """Generates synthetic 24h profiles for GB Market prices and Grid Frequency."""
    hours = np.arange(24)
    # Wholesale Price (£/MWh) - Duck curve shape
    wholesale_price = 50 + 40 * np.sin((hours - 6) * np.pi / 12)**2 + np.random.normal(0, 5, 24)
    # Dynamic Containment Price (£/MWh) - High value service
    dc_price = np.full(24, 15) + np.random.normal(0, 2, 24) 
    dc_price[16:20] = dc_price[16:20] * 2.5 # Peak time premium
    # Grid Frequency (Hz) - Volatility
    frequency = 50 + np.random.normal(0, 0.05, 24)
    
    return pd.DataFrame({
        "Hour": hours,
        "Wholesale_Price": wholesale_price,
        "DC_Price": dc_price,
        "Grid_Frequency": frequency
    })

def optimize_operation(df, bess_capacity, bess_power, it_flexibility):
    """
    Simplified heuristic of the MILP optimization described in Casella et al..
    Prioritizes the highest revenue stream per hour while respecting constraints.
    """
    revenue = []
    action = []
    soc = [50] # Start at 50% State of Charge
    
    for i in range(len(df)):
        price_arb = df['Wholesale_Price'].iloc[i]
        price_dc = df['DC_Price'].iloc[i] * 1.5 # Premium multiplier for availability
        
        current_soc = soc[-1]
        
        # Logic: Compare Wholesale Arbitrage vs Dynamic Containment
        if price_dc > price_arb and 20 < current_soc < 80:
            # Commit to Dynamic Containment (Holding capacity)
            rev = price_dc * (bess_power * 0.8) # Paid for availability
            act = "Dynamic Containment (DC)"
            next_soc = current_soc # DC mostly energy neutral in short term
        elif price_arb < 40 and current_soc < 90:
            # Charge from Grid (Arbitrage - Buy Low)
            rev = -price_arb * bess_power
            act = "Charge (Wholesale)"
            next_soc = min(100, current_soc + (bess_power/bess_capacity)*100)
        elif price_arb > 80 and current_soc > 10:
            # Discharge to Grid (Arbitrage - Sell High)
            rev = price_arb * bess_power
            act = "Discharge (Wholesale)"
            next_soc = max(0, current_soc - (bess_power/bess_capacity)*100)
        else:
            # Idle / IT Load Support
            rev = 0
            act = "Idle / IT Support"
            next_soc = current_soc
            
        revenue.append(rev)
        action.append(act)
        if i < len(df)-1:
            soc.append(next_soc)
            
    df['Revenue'] = revenue
    df['Action'] = action
    df['SoC'] = soc
    return df

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["🌐 The Strategic Context", "🎛️ Digital Twin Simulation", "🧠 The Algorithm", "📊 Key Insights"])

# --- TAB 1: CONTEXT ---
with tab1:
    st.header("Why Data Centres Must Evolve")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### The Problem: System Volatility
        The GB power system is undergoing a transformation to Net Zero[cite: 4].
        * **Loss of Inertia:** As synchronous thermal plants retire, they are replaced by wind/solar. This reduces system inertia, making frequency control harder.
        * **Rising Demand:** Data centres are growing rapidly, increasing demand on the grid[cite: 11].
        
        ### The Opportunity: Flexibility
        Modern Data Centres are not just loads; they are **Virtual Power Plants**.
        * **BESS:** Battery Energy Storage Systems.
        * **TES:** Thermal Energy Storage (Cooling tanks).
        * **Shiftable Load:** Moving IT jobs to cheaper times[cite: 12].
        """)
        
        st.info("The NESO (National Energy System Operator) has launched 'Dynamic' services (DC, DM, DR) specifically for fast-acting assets like batteries[cite: 8, 9].")

    with col2:
        st.markdown("### Visualizing the Shift")
        # Sankey-style concept
        fig_context = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = ["Wind/Solar Generation", "Grid Volatility", "Data Centre Load", "Flexibility Services", "Grid Stability"],
              color = ["green", "red", "blue", "purple", "gold"]
            ),
            link = dict(
              source = [0, 1, 2, 3], 
              target = [1, 2, 3, 4],
              value = [8, 6, 5, 8]
          ))])
        fig_context.update_layout(title_text="The Flexibility Value Chain", font_size=12, height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_context, use_container_width=True)

# --- TAB 2: SIMULATION ---
with tab2:
    st.header("🎛️ Co-Optimization Simulator")
    st.markdown("Adjust the Data Centre's assets below to see how the revenue stack changes.")
    
    # Controls
    col_con1, col_con2, col_con3 = st.columns(3)
    with col_con1:
        bess_cap = st.slider("BESS Capacity (MWh)", 1, 50, 10, help="Size of the battery.")
    with col_con2:
        bess_pwr = st.slider("BESS Power (MW)", 1, 20, 5, help="Max charge/discharge rate.")
    with col_con3:
        it_flex = st.select_slider("IT Load Flexibility", options=["Rigid", "Moderate", "High"], value="Moderate")

    # Run Simulation
    data = generate_market_data()
    results = optimize_operation(data, bess_cap, bess_pwr, it_flex)
    
    # Metrics
    total_rev = results['Revenue'].sum()
    dc_rev = results[results['Action'].str.contains("DC")]['Revenue'].sum()
    arb_rev = results[results['Action'].str.contains("Wholesale")]['Revenue'].sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Daily Revenue", f"£{total_rev:,.2f}", delta="Optimized")
    m2.metric("Dynamic Containment (DC)", f"£{dc_rev:,.2f}", help="Revenue from frequency stability services [cite: 8]")
    m3.metric("Wholesale Arbitrage", f"£{arb_rev:,.2f}", help="Buying low, selling high [cite: 50]")

    # Charts
    st.subheader("24-Hour Operational Strategy")
    
    # Layer 1: Prices
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=results['Hour'], y=results['Wholesale_Price'], name="Wholesale Price (£)", line=dict(color='#00f2ff')), secondary_y=False)
    fig.add_trace(go.Scatter(x=results['Hour'], y=results['DC_Price'], name="DC Service Value (£)", line=dict(color='#ff00ff', dash='dot')), secondary_y=False)
    
    # Layer 2: Battery SoC
    fig.add_trace(go.Scatter(x=results['Hour'], y=results['SoC'], name="Battery SoC (%)", line=dict(color='#ffe600', width=3), fill='tozeroy', fillcolor='rgba(255, 230, 0, 0.1)'), secondary_y=True)
    
    fig.update_layout(title="Market Signals vs. Battery Response", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue Stack Visual
    st.subheader("Revenue Stack Composition")
    fig_stack = px.bar(results, x="Hour", y="Revenue", color="Action", title="Revenue Stacking by Hour [cite: 16]", color_discrete_map={
        "Dynamic Containment (DC)": "#ff00ff",
        "Discharge (Wholesale)": "#00ff00",
        "Charge (Wholesale)": "#ff3333",
        "Idle / IT Support": "#808080"
    })
    fig_stack.update_layout(template="plotly_dark")
    st.plotly_chart(fig_stack, use_container_width=True)

# --- TAB 3: METHODOLOGY ---
with tab3:
    st.header("🧠 The 4-Stage Optimization Process")
    st.markdown("The simulation uses the hierarchical framework described in the paper (Casella et al.)[cite: 59, 83].")
    
    st.markdown("""
    #### How the Mathematics Works:
    The paper proposes a **Mixed Integer Linear Programming (MILP)** formulation.
    
    $$
    \text{Maximize } J = \sum_{t} (R_{DAM, t} + R_{IDA, t} + R_{DFR, t} + R_{BM, t} - C_{deg, t})
    $$
    
    Where:
    * $R_{DAM}$: Revenue from Day-Ahead Market.
    * $R_{DFR}$: Revenue from Dynamic Frequency Response (The most lucrative stack).
    * $C_{deg}$: Battery degradation cost.
    """)
    
    st.markdown("### Sequential Decision Flow")
    st.graphviz_chart("""
    digraph {
        bgcolor="#0e1117"
        node [style=filled, color="#00f2ff", fontcolor="black", fontname="Segoe UI"]
        edge [color="white"]
        
        Start [label="Start Day"]
        Stage1 [label="Stage 1: Day-Ahead\n& DFR Allocation"]
        Stage2 [label="Stage 2: Intraday\nAdjustments"]
        Stage3 [label="Stage 3: Real-Time\nFrequency Calc"]
        Stage4 [label="Stage 4: Balancing\nMechanism"]
        
        Start -> Stage1 [label=" Forecast Prices"]
        Stage1 -> Stage2 [label=" Update IT Load"]
        Stage2 -> Stage3 [label=" Grid Frequency Data"]
        Stage3 -> Stage4 [label=" Spare Capacity"]
    }
    """)
    st.caption("The optimization flow ensures primary IT mission continuity while maximizing profit.")

# --- TAB 4: INSIGHTS ---
with tab4:
    st.header("📊 Critical Insights from the Literature")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("### 1. The 'Spare Capacity' Reality")
        st.write("""
        Most data centres are built with **15 minutes of backup power** for 100% utilization. 
        However, utilization is rarely 100%. This creates **surplus capacity** that is currently wasted but could be monetized[cite: 44, 76].
        """)
        
        st.markdown("### 2. Dynamic Containment is King")
        st.write("""
        Studies show that **Dynamic Containment (DC)** offers the highest Net Present Value (NPV). 
        However, it requires strict State of Energy (SoE) management[cite: 51, 61].
        """)

    with col_i2:
        st.markdown("### 3. The Degradation Trade-off")
        st.warning("Aggressive revenue stacking accelerates battery aging.")
        
        # Degradation visual
        deg_data = pd.DataFrame({
            "Strategy": ["Idle Backup", "Arbitrage Only", "Stacked (DC + Arb)"],
            "Lifetime_Years": [15, 10, 12], # Stacked is often better than Arb only due to shallow cycling in DC
            "NPV_Millions": [0, 2.5, 4.2]
        })
        
        fig_deg = px.scatter(deg_data, x="Lifetime_Years", y="NPV_Millions", text="Strategy", size="NPV_Millions", 
                             color="Strategy", title="Trade-off: Project Value vs. Battery Life ")
        fig_deg.update_traces(textposition='top center')
        fig_deg.update_layout(template="plotly_dark")
        st.plotly_chart(fig_deg, use_container_width=True)
        
    st.markdown("---")
    st.markdown("**Next Steps for Operators:**")
    st.success("Would you like to generate a PDF report of this simulation to present to your stakeholders? (Feature coming soon)")