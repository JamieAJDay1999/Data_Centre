
import pyomo.environ as pyo

def add_it_and_job_constraints(m, params, data):
    # --- IT Job Scheduling Constraints ---
    m.JobCompletion = pyo.ConstraintList()
    for t in m.T_SLOTS:
        for k in m.K_TRANCHES:
            expr = sum(m.ut_ks[(t, k, s)] * params.dt_hours for s in m.TEXT_SLOTS if (t,k,s) in m.ut_ks_idx)
            m.JobCompletion.add(expr == data['Rt'][t] * data['shiftabilityProfile'].get((t, k), 0))

    # --- Manual Piecewise Linearization using SOS2 Constraints ---

    # 1. Define the (x,y) points for the approximation of y = x**1.32
    num_pw_points = 11
    pw_x = [i / (num_pw_points - 1) for i in range(num_pw_points)] # x-coordinates (CPU usage)
    pw_y = [x**1.32 for x in pw_x] # y-coordinates (power factor)
    m.PW_POINTS = pyo.RangeSet(0, num_pw_points - 1)

    # 2. Create weighting variables for each time slot 's' and approximation point 'i'.
    m.w = pyo.Var(m.TEXT_SLOTS, m.PW_POINTS, within=pyo.NonNegativeReals)
    
    # 3. Add constraints for the weights.
    m.WeightSum = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # The weights for each time slot must sum to 1.
        m.WeightSum.add(sum(m.w[s, i] for i in m.PW_POINTS) == 1)

    # FIX: Correctly declare the indexed SOS2 constraint
    # We define a rule that, for each time slot 's', returns the list of variables for the SOS2 set.
    def sos_rule(model, s):
        return [model.w[s, i] for i in model.PW_POINTS]
    m.CPU_SOS2 = pyo.SOSConstraint(m.TEXT_SLOTS, rule=sos_rule, sos=2)

    # 4. Define total_cpu and the power factor based on these weights.
    m.CPUandPower = pyo.ConstraintList()
    m.PowerFactorDef = pyo.ConstraintList()
    m.cpu_power_factor = pyo.Var(m.TEXT_SLOTS, within=pyo.NonNegativeReals)
    
    for s in m.TEXT_SLOTS:
        flexible_usage = sum(m.ut_ks[idx] for idx in m.ut_ks_idx if idx[2] == s)
        base_cpu = data['inflexibleLoadProfile_TEXT'][s] + flexible_usage
        m.CPUandPower.add(m.total_cpu[s] == base_cpu)
        m.CPUandPower.add(m.total_cpu[s] == sum(pw_x[i] * m.w[s, i] for i in m.PW_POINTS))
        m.PowerFactorDef.add(m.cpu_power_factor[s] == sum(pw_y[i] * m.w[s, i] for i in m.PW_POINTS))

        # p_it_total_kw is in kW
        power_expr = params.idle_power_kw + (params.max_power_kw - params.idle_power_kw) * m.cpu_power_factor[s]
        m.CPUandPower.add(m.p_it_total_kw[s] == power_expr)
              

def add_ups_constraints(m, params):
    m.UPS_Constraints = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # e_ups_kwh is in kWh, p_ups... variables are in kW
        prev_energy = params.e_start_kwh if s == m.TEXT_SLOTS.first() else m.e_ups_kwh[s-1]
        charge = params.eta_ch * m.p_ups_ch_kw[s] * params.dt_hours
        discharge = (m.p_ups_disch_kw[s] / params.eta_disch) * params.dt_hours
        m.UPS_Constraints.add(m.e_ups_kwh[s] == prev_energy + charge - discharge)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] <= m.z_ch[s] * params.p_max_ch_kw)
        m.UPS_Constraints.add(m.p_ups_ch_kw[s] >= m.z_ch[s] * params.p_min_ch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] <= m.z_disch[s] * params.p_max_disch_kw)
        m.UPS_Constraints.add(m.p_ups_disch_kw[s] >= m.z_disch[s] * params.p_min_disch_kw)
        m.UPS_Constraints.add(m.z_ch[s] + m.z_disch[s] <= 1)
    m.UPS_Constraints.add(m.e_ups_kwh[m.TEXT_SLOTS.last()] == params.e_start_kwh)

def add_power_balance_constraints(m, params):
    m.PowerBalance = pyo.ConstraintList()
    for s in m.TEXT_SLOTS:
        # All power variables in this balance are in kW
        m.PowerBalance.add(m.p_it_total_kw[s] == m.p_grid_it_kw[s] + m.p_ups_disch_kw[s])
        m.PowerBalance.add(m.p_grid_od_kw[s] == m.p_it_total_kw[s] * params.nominal_overhead_factor)

def add_cooling_constraints(m, params, CYCLE_TES_ENERGY):
    m.CoolingConstraints = pyo.ConstraintList()
    m.CoolingConstraints.add(m.t_it[1] >= params.T_IT_initial_Celsius)
    m.CoolingConstraints.add(m.t_rack[1] >= params.T_Rack_initial_Celsius)
    m.CoolingConstraints.add(m.t_cold_aisle[1] >= params.T_cAisle_initial)
    m.CoolingConstraints.add(m.t_hot_aisle[1] >= params.T_hAisle_initial)
    m.CoolingConstraints.add(m.e_tes_kwh[1] >= params.TES_initial_charge_kWh)
    mcp = params.m_dot_air * params.c_p_air
    
    # This constraint can cause issues if the list is empty
    if len(m.TEXT_SLOTS) > 1:
        avg_hvac_w = sum(m.p_chiller_hvac_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) - 1)
        avg_tes_w = sum(m.p_chiller_tes_w[k] for k in m.TEXT_SLOTS if k > 1) / (len(m.TEXT_SLOTS) - 1)
        m.CoolingConstraints.add(m.p_chiller_hvac_w[1] == avg_hvac_w)
        m.CoolingConstraints.add(m.p_chiller_tes_w[1] == avg_tes_w)
    
    for t in m.TEXT_SLOTS:
        if t > 1:
            # Thermal power (q) is in Watts, Electrical power (p_chiller) is in Watts
            m.CoolingConstraints.add(m.q_cool_w[t] == (m.p_chiller_hvac_w[t] * params.COP_HVAC) + m.q_dis_tes_w[t])
            m.CoolingConstraints.add(m.q_ch_tes_w[t] == m.p_chiller_tes_w[t] * params.COP_HVAC)
            m.CoolingConstraints.add(m.t_in[t] == m.t_hot_aisle[t] - m.q_cool_w[t] / mcp)
            m.CoolingConstraints.add(m.q_cool_w[t] <= (m.t_hot_aisle[t] - params.T_cAisle_lower_limit_Celsius) * mcp)
            
            # Convert IT power from kW to Watts for thermal calculation
            it_heat_watts = m.p_it_total_kw[t] * 1000.0
            
            m.CoolingConstraints.add(m.t_it[t] == m.t_it[t-1] + params.dt_seconds * ((it_heat_watts - params.G_conv * (m.t_it[t-1] - m.t_rack[t])) / params.C_IT))
            m.CoolingConstraints.add(m.t_rack[t] == m.t_rack[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_cold_aisle[t]-m.t_rack[t-1]) + params.G_conv*(m.t_it[t-1]-m.t_rack[t-1])) / params.C_Rack))
            m.CoolingConstraints.add(m.t_cold_aisle[t] == m.t_cold_aisle[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_in[t]-m.t_cold_aisle[t-1]) - params.G_cold*(m.t_cold_aisle[t-1]-params.T_out_Celsius)) / params.C_cAisle))
            m.CoolingConstraints.add(m.t_hot_aisle[t] == m.t_hot_aisle[t-1] + params.dt_seconds * ((params.m_dot_air*params.kappa*params.c_p_air*(m.t_rack[t]-m.t_hot_aisle[t-1])) / params.C_hAisle))
            
            # dE_tes is in kWh, so q_..._w (in W) must be divided by 1000
            dE_tes_kwh = (m.q_ch_tes_w[t]*params.TES_charge_efficiency - m.q_dis_tes_w[t]/params.TES_discharge_efficiency) * params.dt_hours / 1000.0
            m.CoolingConstraints.add(m.e_tes_kwh[t] == m.e_tes_kwh[t-1] + dE_tes_kwh)
            
            m.CoolingConstraints.add(m.q_dis_tes_w[t] - m.q_dis_tes_w[t-1] <= params.TES_p_dis_ramp)
            m.CoolingConstraints.add(m.q_ch_tes_w[t] - m.q_ch_tes_w[t-1] <= params.TES_p_ch_ramp)
            m.CoolingConstraints.add(m.p_chiller_tes_w[t] + m.p_chiller_hvac_w[t] <= params.P_chiller_max)
            m.CoolingConstraints.add(m.q_cool_w[t] >= it_heat_watts)

    if CYCLE_TES_ENERGY:
        m.CoolingConstraints.add(m.e_tes_kwh[m.TEXT_SLOTS.last()] == m.e_tes_kwh[1])
