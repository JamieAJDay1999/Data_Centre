"""Duration certification loop (Section 4.5 of the paper / doc 03).

For each EFA block carrying turn-down commitments, a worst-case simultaneous
call is simulated: every committed product is called at the block start and
must deliver for its own sustain duration tau_j, producing a staircase
deviation profile. A delivery model (the same facility + market model with
commitments FIXED and the staircase imposed on the grid-power trajectory) is
solved for feasibility, with all workload-completion, storage and thermal
constraints active - Paper 1's Scenario 3 logic, generalised to a portfolio.

Infeasible calls are resolved by binary search on a scale factor and returned
to the stacking problem as a linear cut on the block's total commitment.
The revenue difference between the first (uncertified) and final (certified)
portfolios is the phantom-flexibility metric reported in the paper.
"""
import copy

import pyomo.environ as pyo

from . import config
from .facility import build_facility_model
from .stack_model import add_market_layer, solve, StackOptions
from .postprocess import revenue_summary


def _staircase(m_solved, block: int, t0: int):
    """Per-slot worst-case deviation (kW) from calling every turn-down
    commitment active in `block` at slot t0, each for its own tau_j."""
    prods, mkt = m_solved._products, m_solved._mkt
    calls = []  # (kW, n_slots)
    for (j, w) in m_solved.R_IDX:
        p = prods[j]
        if p.direction != "down":
            continue
        active = False
        if p.granularity == "efa":
            active = (w == block)
        elif p.granularity == "sp":
            active = (config.slot_to_efa((w - 1) * config.SLOTS_PER_SP + 1)
                      == block and t0 in config.sp_slots(w))
        else:  # window products
            active = t0 in mkt.dso_windows[w - 1][0]
        if active:
            r = pyo.value(m_solved.r[j, w])
            if r > 1e-3:
                calls.append((r, int(round(p.tau_h / 0.25))))
    if not calls:
        return {}
    horizon = max(n for _, n in calls)
    dev = {}
    for i in range(horizon):
        t = t0 + i
        if t > config.N_SLOTS_DAY:
            break
        dev[t] = sum(r for r, n in calls if i < n)
    return dev


def _delivery_feasible(params, data, mkt, opts, fixed_r, p_base, dev, scale,
                       solver_name):
    """Build and solve the delivery model: fixed commitments + staircase."""
    m = build_facility_model(params, data)
    d_opts = copy.copy(opts)
    d_opts.fixed_commitments = fixed_r
    add_market_layer(m, params, mkt, d_opts)
    m.Delivery = pyo.ConstraintList()
    for t, d in dev.items():
        m.Delivery.add(m.P_grid[t] <= p_base[t] - scale * d + config.CERT_TOL_KW)
    _, ok = solve(m, solver_name=solver_name)
    return ok


def certify(params, data, mkt, opts: StackOptions, m_solved, solver_name,
            verbose=True):
    """Run the certification loop starting from a solved stacking model.

    Returns (m_certified, report) where report records per-iteration revenue,
    the cuts applied and the phantom-flexibility gap.
    """
    uncertified = revenue_summary(m_solved)
    cuts = {}          # {(block, t0): scale}
    report = {"iterations": 0, "checks": 0, "failures": 0, "cuts": {},
              "uncertified_net_cost": uncertified["net_cost"]}
    m = m_solved

    for it in range(config.CERT_MAX_OUTER_ITERS):
        report["iterations"] = it + 1
        p_base = {t: pyo.value(m.P_grid[t]) for t in m.TEXT_SLOTS}
        fixed_r = {(j, w): pyo.value(m.r[j, w]) for (j, w) in m.R_IDX}
        new_cuts = False
        for block in range(1, config.N_EFA + 1):
            for off in config.CERT_START_OFFSETS:
                t0 = (block - 1) * config.SLOTS_PER_EFA + 1 + off
                dev = _staircase(m, block, t0)
                if not dev:
                    continue
                report["checks"] += 1
                if _delivery_feasible(params, data, mkt, opts, fixed_r,
                                      p_base, dev, 1.0, solver_name):
                    continue
                report["failures"] += 1
                lo, hi = 0.0, 1.0
                for _ in range(config.CERT_BISECTION_STEPS):
                    mid = 0.5 * (lo + hi)
                    if _delivery_feasible(params, data, mkt, opts, fixed_r,
                                          p_base, dev, mid, solver_name):
                        lo = mid
                    else:
                        hi = mid
                key = (block, t0)
                if lo < cuts.get(key, 1.0):
                    cuts[key] = lo
                    new_cuts = True
                if verbose:
                    print(f"  [cert] block {block} t0={t0}: worst-case call "
                          f"infeasible; certified scale = {lo:.3f}")
        if not new_cuts:
            break
        # Re-solve the stacking problem with the accumulated cuts.
        cand_totals = {}
        for (block, t0), scale in cuts.items():
            total = sum(pyo.value(m.r[j, w]) for (j, w) in m.R_IDX
                        if m._products[j].direction == "down"
                        and _in_block(m._products[j], w, block, m._mkt))
            cand_totals[(block, t0)] = scale * total
        m = build_facility_model(params, data)
        add_market_layer(m, params, mkt, opts)
        m.EnvelopeCuts = pyo.ConstraintList()
        for (block, t0), cap in cand_totals.items():
            expr = sum(m.r[j, w] for (j, w) in m.R_IDX
                       if m._products[j].direction == "down"
                       and _in_block(m._products[j], w, block, mkt))
            m.EnvelopeCuts.add(expr <= cap + config.CERT_TOL_KW)
        _, ok = solve(m, solver_name=solver_name)
        if not ok:
            raise RuntimeError("Stacking model infeasible after envelope cuts")

    certified = revenue_summary(m)
    report["certified_net_cost"] = certified["net_cost"]
    report["phantom_gbp"] = (report["uncertified_net_cost"]
                             - certified["net_cost"]) * -1.0
    report["cuts"] = {f"block{b}_t{t}": s for (b, t), s in cuts.items()}
    return m, report


def _in_block(product, w, block, mkt):
    if product.granularity == "efa":
        return w == block
    if product.granularity == "sp":
        return config.slot_to_efa((w - 1) * config.SLOTS_PER_SP + 1) == block
    return any(config.slot_to_efa(t) == block for t in mkt.dso_windows[w - 1][0])
