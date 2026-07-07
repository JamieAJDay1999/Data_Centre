"""Market layer: commitments, headrooms, reservations, objective.

Implements Sections 4.2-4.6 of the paper (rev_stack/paper/main.tex) on top of
the facility model built by facility.build_facility_model:

  - commitment variables r[j,w] at each product's native granularity and
    per-asset allocations r_alloc[j,w,a]                          (Eq. 15)
  - per-asset headroom constraints                                 (Eq. 16-22)
  - shared (aggregate) stacking constraints + connection cap       (Eq. 23-26)
  - stored-energy reservations for UPS and TES                     (Eq. 28-30)
  - net-cost objective                                             (Eq. 31)

Decision staging (paper Section 4, decision-gate paragraph): the commitment
variables r belong to the day-ahead gate, while the delivered-energy
variables (BM offers/bids, DFS deliveries) are within-day RECOURSE
decisions. A single solve of this model co-optimises both layers under
perfect foresight and is therefore the full-information upper bound (B4);
gate-sequential operation is evaluated by benchmarks.py (B5) and the
first-stage/recourse split under uncertainty by run_scenarios.py.

Conservative linearisations (documented in the paper):
  * IT turn-down headroom uses a single utilisation floor at the largest
    IT-eligible duration TAU_IT_MAX_H; the floor power is evaluated through
    a second piecewise map sharing the breakpoints of the facility's
    CPU->power curve. Because u^1.32 is convex and the floor power is
    minimised over the weight simplex, adjacent-breakpoint behaviour is
    automatic and no SOS2 restriction is required.
  * IT turn-up headroom is set to zero (Paper 1's asymmetry result).
  * UPS minimum-power thresholds are ignored in headroom and checked in
    certification.
"""
from dataclasses import dataclass, field
import time

import pyomo.environ as pyo

from . import config
from .market_parameters import (PRODUCTS, ASSETS, DOWN_PRODUCTS, UP_PRODUCTS,
                                TAU_IT_MAX_H, n_windows)
from .market_data import MarketDay


@dataclass
class StackOptions:
    """Switches used by the benchmark ladder (Section 4.9 / doc 04)."""
    allowed_products: tuple = tuple(PRODUCTS)   # B0 passes ()
    enable_bm: bool = True
    enable_dfs: bool = True
    enable_dso: bool = True
    cm_kw: float = config.CM_DEFAULT_KW
    p_conn_kw: float = config.P_CONN_KW
    bm_acceptance: float = config.BM_ACCEPTANCE
    allow_it_for_qr: bool = False               # speed-class sensitivity
    excluded_assets: tuple = ()                 # leave-one-out attribution
    fixed_commitments: dict = None              # {(j, w): kW} for greedy/cert runs


def clean_commitment_value(value, tol=1e-8):
    """Clean solver residue before reusing commitments as fixed data."""
    if value is None:
        return 0.0
    value = float(value)
    if -tol <= value < 0.0:
        return 0.0
    return value


def commitment_values(m, tol=1e-8):
    """Return solved commitments safe to pass into StackOptions."""
    return {(j, w): clean_commitment_value(pyo.value(m.r[j, w]), tol)
            for (j, w) in m.R_IDX}


def _window_of(product, t):
    if product.granularity == "efa":
        return config.slot_to_efa(t)
    if product.granularity == "sp":
        return config.slot_to_sp(t)
    return None


def add_market_layer(m, params, mkt: MarketDay, opts: StackOptions):
    """Attach the market layer to a facility model m (modified in place)."""
    prods = {name: PRODUCTS[name] for name in opts.allowed_products}
    if opts.allow_it_for_qr and "PQR" in prods:
        p = prods["PQR"]
        prods["PQR"] = type(p)(p.name, p.direction, p.tau_h, p.granularity,
                               ("UPS", "IT"), p.price_file, p.util_priced,
                               p.family)
    if opts.excluded_assets:
        pruned = {}
        for name, p in prods.items():
            assets = tuple(a for a in p.assets if a not in opts.excluded_assets)
            if assets:
                pruned[name] = type(p)(p.name, p.direction, p.tau_h,
                                       p.granularity, assets, p.price_file,
                                       p.util_priced, p.family)
        prods = pruned
    m._products = prods
    m._opts = opts
    m._mkt = mkt
    T = list(range(1, config.N_SLOTS_DAY + 1))

    # ------------------------------------------------------------------
    # Commitment variables at native granularity, with per-asset allocation
    # ------------------------------------------------------------------
    r_idx, alloc_idx = [], []
    for name, p in prods.items():
        if p.granularity == "window":
            # data-driven windows (DSO): one commitment per tendered window
            for w, (slots, ap, up) in enumerate(mkt.dso_windows, start=1):
                r_idx.append((name, w))
                alloc_idx += [(name, w, a) for a in p.assets]
        else:
            for w in range(1, n_windows(p) + 1):
                if mkt.avail_price.get(name, {}).get(w) is None:
                    continue
                r_idx.append((name, w))
                alloc_idx += [(name, w, a) for a in p.assets]
    m.R_IDX = pyo.Set(initialize=r_idx, dimen=2)
    m.ALLOC_IDX = pyo.Set(initialize=alloc_idx, dimen=3)
    m.r = pyo.Var(m.R_IDX, within=pyo.NonNegativeReals, initialize=0)
    m.r_alloc = pyo.Var(m.ALLOC_IDX, within=pyo.NonNegativeReals, initialize=0)

    m.AllocSum = pyo.ConstraintList()
    for (j, w) in m.R_IDX:
        m.AllocSum.add(sum(m.r_alloc[j, w, a] for a in prods[j].assets)
                       == m.r[j, w])

    if opts.fixed_commitments is not None:
        for (j, w) in m.R_IDX:
            m.r[j, w].fix(clean_commitment_value(
                opts.fixed_commitments.get((j, w), 0.0)))

    # helper: commitments of a direction allocated to asset a, active at slot t
    def alloc_at(t, direction, asset):
        total = 0
        for (j, w) in m.R_IDX:
            p = prods[j]
            if p.direction != direction or asset not in p.assets:
                continue
            if p.granularity == "window":
                slots = mkt.dso_windows[w - 1][0]
                if t in slots:
                    total += m.r_alloc[j, w, asset]
            elif _window_of(p, t) == w:
                total += m.r_alloc[j, w, asset]
        return total

    def committed_at(t, direction, tau_min_h=0.0):
        total = 0
        for (j, w) in m.R_IDX:
            p = prods[j]
            if p.direction != direction or p.tau_h < tau_min_h:
                continue
            if p.granularity == "window":
                if t in mkt.dso_windows[w - 1][0]:
                    total += m.r[j, w]
            elif _window_of(p, t) == w:
                total += m.r[j, w]
        return total
    m._alloc_at, m._committed_at = alloc_at, committed_at

    # ------------------------------------------------------------------
    # Delivered-energy positions: BM, DFS (within-day recourse variables;
    # co-optimised with the day-ahead layer only in the full-information
    # upper bound, see module docstring)
    # ------------------------------------------------------------------
    m.x_bm_off = pyo.Var(T, within=pyo.NonNegativeReals, initialize=0)
    m.x_bm_bid = pyo.Var(T, within=pyo.NonNegativeReals, initialize=0)
    if not opts.enable_bm:
        for t in T:
            m.x_bm_off[t].fix(0)
            m.x_bm_bid[t].fix(0)

    dfs_dn_slots = sorted({t for slots, d, _ in mkt.dfs_events
                           for t in slots if d == "down"})
    dfs_up_slots = sorted({t for slots, d, _ in mkt.dfs_events
                           for t in slots if d == "up"})
    m.DFS_DN = pyo.Set(initialize=dfs_dn_slots)
    m.DFS_UP = pyo.Set(initialize=dfs_up_slots)
    m.x_dfs_dn = pyo.Var(m.DFS_DN, within=pyo.NonNegativeReals, initialize=0)
    m.x_dfs_up = pyo.Var(m.DFS_UP, within=pyo.NonNegativeReals, initialize=0)
    if not opts.enable_dfs:
        for t in dfs_dn_slots:
            m.x_dfs_dn[t].fix(0)
        for t in dfs_up_slots:
            m.x_dfs_up[t].fix(0)

    # ------------------------------------------------------------------
    # IT utilisation floor and duration-aware IT headroom (Eq. 16-18)
    # ------------------------------------------------------------------
    tau_slots = int(round(TAU_IT_MAX_H / params.dt_hours))
    num_pw = 11
    pw_x = [i / (num_pw - 1) for i in range(num_pw)]
    pw_y = [x ** 1.32 for x in pw_x]
    m.FLR_PTS = pyo.RangeSet(0, num_pw - 1)
    m.w_flr = pyo.Var(T, m.FLR_PTS, within=pyo.NonNegativeReals, initialize=0)
    m.u_flr = pyo.Var(T, bounds=(0, params.max_cpu_usage), initialize=0)
    m.p_flr = pyo.Var(T, within=pyo.NonNegativeReals, initialize=0)
    m.FloorDef = pyo.ConstraintList()
    for t in T:
        # deferrable executions at t with >= tau_slots of remaining slack
        d_t = sum(m.ut_ks[(t2, k, s)] for (t2, k, s) in m.ut_ks_idx
                  if s == t and (t2 + params.tranche_max_delay[k]) - t >= tau_slots)
        m.FloorDef.add(m.u_flr[t] == m.total_cpu[t] - d_t)
        m.FloorDef.add(sum(m.w_flr[t, i] for i in m.FLR_PTS) == 1)
        m.FloorDef.add(m.u_flr[t] == sum(pw_x[i] * m.w_flr[t, i] for i in m.FLR_PTS))
        # Floor power: minimised over the simplex; convexity of u^1.32 makes
        # the piecewise interpolant the minimum, so no SOS2 is needed and the
        # resulting headroom P_IT - p_flr is a conservative (never over-)
        # estimate of the achievable sustained reduction.
        m.FloorDef.add(m.p_flr[t] == params.idle_power_kw
                       + (params.max_power_kw - params.idle_power_kw)
                       * sum(pw_y[i] * m.w_flr[t, i] for i in m.FLR_PTS))

    def h_it_dn(mod, t):
        return mod.p_it_total_kw[t] - mod.p_flr[t]
    m.H_it_dn = pyo.Expression(T, rule=h_it_dn)

    # ------------------------------------------------------------------
    # UPS and cooling headrooms (Eq. 19-22)
    # ------------------------------------------------------------------
    def h_ups_dn(mod, t):
        return (params.p_max_disch_kw - mod.p_ups_disch_kw[t]
                + mod.p_ups_ch_kw[t])
    def h_ups_up(mod, t):
        return (params.p_max_ch_kw - mod.p_ups_ch_kw[t]
                + mod.p_ups_disch_kw[t])
    def h_cl_up(mod, t):
        return (params.P_chiller_max - mod.p_chiller_hvac_w[t]
                - mod.p_chiller_tes_w[t]) / 1000.0
    m.H_ups_dn = pyo.Expression(T, rule=h_ups_dn)
    m.H_ups_up = pyo.Expression(T, rule=h_ups_up)
    m.H_cl_up = pyo.Expression(T, rule=h_cl_up)

    # ------------------------------------------------------------------
    # Per-asset capacity constraints (Eq. 23-24)
    # ------------------------------------------------------------------
    m.AssetHeadroom = pyo.ConstraintList()
    for t in T:
        # IT: duration-aware floor headroom; realised on the grid side
        m.AssetHeadroom.add(alloc_at(t, "down", "IT") <= m.H_it_dn[t])
        m.AssetHeadroom.add(alloc_at(t, "down", "IT") <= m.p_grid_it_kw[t])
        # UPS: converter headroom; discharge only displaces grid-served IT
        m.AssetHeadroom.add(alloc_at(t, "down", "UPS") <= m.H_ups_dn[t])
        m.AssetHeadroom.add(alloc_at(t, "down", "UPS")
                            <= m.p_grid_it_kw[t] + m.p_ups_ch_kw[t])
        m.AssetHeadroom.add(alloc_at(t, "up", "UPS") <= m.H_ups_up[t])
        # Cooling turn-down: curtail chiller, replace via TES discharge
        m.AssetHeadroom.add(alloc_at(t, "down", "CL")
                            <= (m.p_chiller_hvac_w[t] + m.p_chiller_tes_w[t]) / 1000.0)
        m.AssetHeadroom.add(alloc_at(t, "down", "CL")
                            <= m.p_chiller_tes_w[t] / 1000.0
                            + (params.TES_w_discharge_max - m.q_dis_tes_w[t])
                            / (1000.0 * params.COP_HVAC))
        # Cooling turn-up: spare chiller capacity AND spare TES charge rate
        m.AssetHeadroom.add(alloc_at(t, "up", "CL") <= m.H_cl_up[t])
        m.AssetHeadroom.add(alloc_at(t, "up", "CL")
                            <= (params.TES_w_charge_max - m.q_ch_tes_w[t])
                            / (1000.0 * params.COP_HVAC))

    # ------------------------------------------------------------------
    # Aggregate stacking constraints + connection cap (Eq. 25-26)
    # ------------------------------------------------------------------
    m.Stacking = pyo.ConstraintList()
    cm_kw = opts.cm_kw * (mkt.cm_derating if opts.cm_kw else 0.0)
    for t in T:
        h_dn_total = (m.H_it_dn[t] + m.H_ups_dn[t]
                      + (m.p_chiller_hvac_w[t] + m.p_chiller_tes_w[t]) / 1000.0)
        dfs_dn = m.x_dfs_dn[t] if t in dfs_dn_slots else 0
        dfs_up = m.x_dfs_up[t] if t in dfs_up_slots else 0
        cm_term = cm_kw if t in config.CM_WINDOW_SLOTS else 0.0
        m.Stacking.add(committed_at(t, "down") + m.x_bm_off[t] + dfs_dn
                       + cm_term <= h_dn_total)
        h_up_total = m.H_ups_up[t] + m.H_cl_up[t]
        m.Stacking.add(committed_at(t, "up") + m.x_bm_bid[t] + dfs_up
                       <= h_up_total)
        m.Stacking.add(m.P_grid[t] + committed_at(t, "up") + m.x_bm_bid[t]
                       + dfs_up <= opts.p_conn_kw)
        if config.DFS_PRIMACY_EXCLUSIVE and (t in dfs_dn_slots):
            # During DFS events, DFS delivery and committed products may not
            # both claim the same headroom - already enforced by the shared
            # constraint above; additionally bar new BM activity (primacy).
            m.Stacking.add(m.x_bm_off[t] <= 0)

    # ------------------------------------------------------------------
    # Stored-energy reservations (Eq. 28-30)
    # ------------------------------------------------------------------
    m.Reservations = pyo.ConstraintList()
    for t in T:
        ups_dn_res = sum(m.r_alloc[j, w, "UPS"] * prods[j].tau_h
                         / params.eta_disch
                         for (j, w) in m.R_IDX
                         if "UPS" in prods[j].assets
                         and prods[j].direction == "down"
                         and _active(prods[j], w, t, mkt))
        ups_up_res = sum(m.r_alloc[j, w, "UPS"] * prods[j].tau_h
                         * params.eta_ch
                         for (j, w) in m.R_IDX
                         if "UPS" in prods[j].assets
                         and prods[j].direction == "up"
                         and _active(prods[j], w, t, mkt))
        m.Reservations.add(m.e_ups_kwh[t] >= params.e_min_kwh + ups_dn_res)
        m.Reservations.add(m.e_ups_kwh[t] <= params.e_max_kwh - ups_up_res)
        cl_dn_res = sum(m.r_alloc[j, w, "CL"] * prods[j].tau_h
                        * params.COP_HVAC / params.TES_discharge_efficiency
                        for (j, w) in m.R_IDX
                        if "CL" in prods[j].assets
                        and prods[j].direction == "down"
                        and _active(prods[j], w, t, mkt))
        cl_up_res = sum(m.r_alloc[j, w, "CL"] * prods[j].tau_h
                        * params.COP_HVAC * params.TES_charge_efficiency
                        for (j, w) in m.R_IDX
                        if "CL" in prods[j].assets
                        and prods[j].direction == "up"
                        and _active(prods[j], w, t, mkt))
        m.Reservations.add(m.e_tes_kwh[t] >= params.E_TES_min_kWh + cl_dn_res)
        m.Reservations.add(m.e_tes_kwh[t] <= params.TES_capacity_kWh - cl_up_res)

    # ------------------------------------------------------------------
    # Objective (Eq. 31): energy + network cost - market revenues
    # ------------------------------------------------------------------
    dt = params.dt_hours
    m.cost_energy = pyo.Expression(expr=sum(
        dt * (mkt.da_price[s] + mkt.duos[s]) * m.P_grid[s] / 1000.0
        for s in m.TEXT_SLOTS))

    def _avail_rev():
        total = 0
        for (j, w) in m.R_IDX:
            p = prods[j]
            if p.granularity == "window":
                slots, ap, up = mkt.dso_windows[w - 1]
                hours = len(slots) * dt
                total += ap * m.r[j, w] * hours / 1000.0
            else:
                hours = (config.SLOTS_PER_EFA if p.granularity == "efa"
                         else config.SLOTS_PER_SP) * dt
                total += mkt.avail_price[j][w] * m.r[j, w] * hours / 1000.0
        return total
    m.rev_availability = pyo.Expression(expr=_avail_rev())

    def _util_rev():
        total = 0
        for (j, w) in m.R_IDX:
            p = prods[j]
            phi = mkt.phi.get(j, 0.0)
            if p.granularity == "window":
                slots, ap, up = mkt.dso_windows[w - 1]
                hours, price = len(slots) * dt, up
            else:
                hours = (config.SLOTS_PER_EFA if p.granularity == "efa"
                         else config.SLOTS_PER_SP) * dt
                price = mkt.util_price.get(j, 0.0)
            total += phi * price * m.r[j, w] * hours / 1000.0
        return total
    m.rev_utilisation = pyo.Expression(expr=_util_rev())

    # BM revenue is earned on the SPREAD against the day-ahead price, not the
    # gross accepted price: an accepted offer is paid the offer price for
    # energy already bought day-ahead (and later re-consumed in the rebound),
    # so the facility's net gain per accepted MWh is (offer - DA); an accepted
    # bid buys extra energy at the bid price instead of the DA price, gaining
    # (DA - bid). Acceptance probability kappa scales expected volume.
    T96 = list(range(1, config.N_SLOTS_DAY + 1))
    m.rev_bm = pyo.Expression(expr=sum(
        dt * opts.bm_acceptance
        * ((mkt.bm_offer[t] - mkt.da_price[t]) * m.x_bm_off[t]
           + (mkt.da_price[t] - mkt.bm_bid[t]) * m.x_bm_bid[t]) / 1000.0
        for t in T96))

    dfs_price = {t: pr for slots, d, pr in mkt.dfs_events for t in slots}
    m.rev_dfs = pyo.Expression(expr=(
        sum(dt * dfs_price[t] * m.x_dfs_dn[t] / 1000.0 for t in dfs_dn_slots)
        + sum(dt * dfs_price[t] * m.x_dfs_up[t] / 1000.0 for t in dfs_up_slots)))

    m.rev_cm = pyo.Expression(expr=(mkt.cm_price * mkt.cm_derating
                                    * opts.cm_kw / 365.0))

    m.objective = pyo.Objective(
        expr=(m.cost_energy - m.rev_availability - m.rev_utilisation
              - m.rev_bm - m.rev_dfs - m.rev_cm),
        sense=pyo.minimize)
    return m


def _active(product, w, t, mkt):
    """True if window w of a product covers slot t."""
    if product.granularity == "window":
        return t in mkt.dso_windows[w - 1][0]
    return _window_of(product, t) == w


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------
def get_solver(preference=None):
    preference = preference or config.SOLVER_PREFERENCE
    for name in preference:
        try:
            solver = pyo.SolverFactory(name)
            try:
                ok = solver.available(False)
            except TypeError:
                ok = solver.available()
            if ok:
                return name, solver
        except Exception:
            continue
    raise RuntimeError(f"No solver available from {preference}")


def prepare_for_solver(m, solver_name: str):
    """Drop SOS2 components for solvers that do not support them (HiGHS).

    Safe here because the piecewise curve u^1.32 is convex and the weighted
    power factor is always under downward pressure (it only ever increases
    cost or tightens a <= constraint), so the simplex minimum already lies on
    the piecewise interpolant. SCIP keeps the SOS2 restriction as belt and
    braces.
    """
    if "highs" in solver_name.lower():
        if hasattr(m, "CPU_SOS2"):
            m.del_component(m.CPU_SOS2)
            if hasattr(m, "CPU_SOS2_index"):
                m.del_component(m.CPU_SOS2_index)
    return m


def solve(m, solver_name=None, solver=None, tee=False, time_limit=None,
          label=None, verbose=False):
    """Solve in place. Returns (solver_name, ok) where ok means a usable
    (optimal or time-limited feasible) solution has been loaded into m.
    Infeasibility is reported as ok=False, never raised, so the certification
    loop can probe delivery feasibility."""
    if solver is None:
        solver_name, solver = get_solver(
            (solver_name,) if solver_name else None)
    if time_limit is None:
        time_limit = config.SOLVER_TIME_LIMIT_SECONDS
    prepare_for_solver(m, solver_name)

    kwargs = {"tee": tee, "load_solutions": False}
    old_time_limit = None
    if solver_name == "scip":
        kwargs["options"] = {"limits/time": time_limit}
    elif "highs" in solver_name.lower() and hasattr(solver, "config"):
        try:
            old_time_limit = solver.config.time_limit
            solver.config.time_limit = time_limit
        except (AttributeError, ValueError):
            old_time_limit = None
    if verbose and label:
        print(f"  [solve] start {label} ({solver_name}, limit={time_limit}s)",
              flush=True)
    t0 = time.time()
    try:
        results = solver.solve(m, **kwargs)
    finally:
        if old_time_limit is not None:
            solver.config.time_limit = old_time_limit
    elapsed = time.time() - t0
    tc = str(results.solver.termination_condition)
    ok = tc in ("optimal", "maxTimeLimit", "feasible", "locallyOptimal")
    if tc == "other" and results.problem.lower_bound != results.problem.upper_bound:
        ok = True  # SCIP quirk handled as in optimisation.py
    if ok:
        try:
            m.solutions.load_from(results)
        except Exception:
            ok = False
    if verbose and label:
        print(f"  [solve] done  {label}: {tc}, ok={ok}, {elapsed:.1f}s",
              flush=True)
    return solver_name, ok
