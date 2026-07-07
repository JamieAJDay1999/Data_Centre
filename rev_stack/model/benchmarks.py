"""Benchmark ladder B0-B5 (Section 4.9 of the paper): the value of
co-optimisation, certification and market-sequential operation.

  B0  energy-only (Paper 1 cost optimisation; no market participation)
  B1  dedicated assets (UPS -> Dynamic Containment only; no other products)
  B2  greedy sequential stacking (products committed one at a time by
      standalone value, each taking the residual capability)
  B3  full co-optimisation, uncertified
  B4  full co-optimisation, certified (the full-information upper bound)
  B5  market-sequential certified: decisions taken in gate order. The
      day-ahead stage co-optimises the energy profile and all availability
      commitments with BM/DFS unavailable and is certified; the schedule and
      commitments are then frozen and BM/DFS positions are filled over the
      residual aggregate headroom only (per-slot LP, solved in closed form).
      The B4-B5 gap prices the within-day foresight that the one-pass
      co-optimisation assumes.

Usage (repo root, venv active):
    python -m rev_stack.model.benchmarks --date 2025-01-15
    python -m rev_stack.model.benchmarks --date 2025-01-15 --solver scip

Output: rev_stack/results/benchmarks_<date>.csv
"""
import argparse

import pandas as pd
import pyomo.environ as pyo

from . import config
from .facility import scaled_parameters, load_facility_data, build_facility_model
from .market_data import load_market_day
from .market_parameters import PRODUCTS
from .stack_model import (StackOptions, add_market_layer, solve, get_solver,
                          clean_commitment_value)
from .certification import certify
from .postprocess import revenue_summary


def _solve_case(params, data, mkt, opts, solver_name):
    m = build_facility_model(params, data)
    add_market_layer(m, params, mkt, opts)
    _, ok = solve(m, solver_name=solver_name)
    if not ok:
        raise RuntimeError("case infeasible")
    return m


def _sequential_recourse(m, mkt, opts):
    """Stage-2 (within-day) BM/DFS fill over the residual headroom of a
    solved day-ahead model whose commitments and schedule are taken as fixed.

    Mirrors the aggregate stacking constraints of stack_model (Eq. 25-26)
    with all stage-1 quantities at their solved values. Because the
    objective is linear and, per slot, all candidate positions in a
    direction draw on a single shared residual pool, the per-slot LP is
    solved exactly by assigning the pool to the highest-value positive
    candidate. DFS primacy is applied as in stack_model: BM offers are
    barred during DFS turn-down events.

    Returns (rev_bm_gbp, rev_dfs_gbp).
    """
    dt = 0.25
    kappa = opts.bm_acceptance
    cm_kw = opts.cm_kw * (mkt.cm_derating if opts.cm_kw else 0.0)
    dfs_price = {t: pr for slots, d, pr in mkt.dfs_events for t in slots}
    dfs_dn_slots = {t for slots, d, _ in mkt.dfs_events
                    for t in slots if d == "down"}
    dfs_up_slots = {t for slots, d, _ in mkt.dfs_events
                    for t in slots if d == "up"}
    rev_bm = rev_dfs = 0.0
    for t in range(1, config.N_SLOTS_DAY + 1):
        # residual turn-down pool
        h_dn = (pyo.value(m.H_it_dn[t]) + pyo.value(m.H_ups_dn[t])
                + (pyo.value(m.p_chiller_hvac_w[t])
                   + pyo.value(m.p_chiller_tes_w[t])) / 1000.0)
        cm_term = cm_kw if t in config.CM_WINDOW_SLOTS else 0.0
        pool_dn = max(0.0, h_dn - pyo.value(m._committed_at(t, "down"))
                      - cm_term)
        candidates = []
        if opts.enable_bm and not (config.DFS_PRIMACY_EXCLUSIVE
                                   and t in dfs_dn_slots):
            candidates.append(("bm", kappa * (mkt.bm_offer[t]
                                              - mkt.da_price[t])))
        if opts.enable_dfs and t in dfs_dn_slots:
            candidates.append(("dfs", dfs_price[t]))
        if candidates and pool_dn > 0:
            kind, val = max(candidates, key=lambda c: c[1])
            if val > 0:
                gbp = dt * val * pool_dn / 1000.0
                if kind == "bm":
                    rev_bm += gbp
                else:
                    rev_dfs += gbp
        # residual turn-up pool (headroom AND connection cap)
        h_up = pyo.value(m.H_ups_up[t]) + pyo.value(m.H_cl_up[t])
        committed_up = pyo.value(m._committed_at(t, "up"))
        pool_up = max(0.0, min(h_up - committed_up,
                               opts.p_conn_kw - pyo.value(m.P_grid[t])
                               - committed_up))
        candidates = []
        if opts.enable_bm:
            candidates.append(("bm", kappa * (mkt.da_price[t]
                                              - mkt.bm_bid[t])))
        if opts.enable_dfs and t in dfs_up_slots:
            candidates.append(("dfs", dfs_price[t]))
        if candidates and pool_up > 0:
            kind, val = max(candidates, key=lambda c: c[1])
            if val > 0:
                gbp = dt * val * pool_up / 1000.0
                if kind == "bm":
                    rev_bm += gbp
                else:
                    rev_dfs += gbp
    return rev_bm, rev_dfs


def _standalone_value_order(params, data, mkt, solver_name):
    """Rank products by their standalone daily value (solo participation)."""
    values = {}
    for name in PRODUCTS:
        opts = StackOptions(allowed_products=(name,), enable_bm=False,
                            enable_dfs=False, enable_dso=False)
        try:
            m = _solve_case(params, data, mkt, opts, solver_name)
            s = revenue_summary(m)
            values[name] = s["rev_availability"] + s["rev_utilisation"]
        except RuntimeError:
            values[name] = 0.0
    return sorted(values, key=values.get, reverse=True), values


def run_benchmarks(date, solver_name=None):
    params = scaled_parameters()
    mkt = load_market_day(date)
    data = load_facility_data(params, mkt.da_price)
    if solver_name is None:
        solver_name, _ = get_solver()
    rows = []

    # B0: energy-only ---------------------------------------------------------
    opts = StackOptions(allowed_products=(), enable_bm=False,
                        enable_dfs=False, enable_dso=False)
    m0 = _solve_case(params, data, mkt, opts, solver_name)
    s0 = revenue_summary(m0)
    rows.append({"case": "B0_energy_only", **s0})
    print(f"B0 energy-only        net cost {s0['net_cost']:9.2f} GBP")

    # B1: dedicated assets (UPS -> DC only) -----------------------------------
    opts = StackOptions(allowed_products=("DCL", "DCH"), enable_bm=False,
                        enable_dfs=False, enable_dso=False)
    m1 = _solve_case(params, data, mkt, opts, solver_name)
    s1 = revenue_summary(m1)
    rows.append({"case": "B1_dedicated", **s1})
    print(f"B1 dedicated (DC)     net cost {s1['net_cost']:9.2f} GBP")

    # B2: greedy sequential ---------------------------------------------------
    order, standalone = _standalone_value_order(params, data, mkt, solver_name)
    fixed = {}
    for i, name in enumerate(order):
        allowed = tuple(order[:i + 1])
        opts = StackOptions(allowed_products=allowed, enable_bm=False,
                            enable_dfs=False, enable_dso=False,
                            fixed_commitments=None)
        m_g = build_facility_model(params, data)
        add_market_layer(m_g, params, mkt, opts)
        # fix earlier products at their greedy volumes; only the newest is free
        for (j, w) in m_g.R_IDX:
            if j != name:
                m_g.r[j, w].fix(clean_commitment_value(
                    fixed.get((j, w), 0.0)))
        _, ok = solve(m_g, solver_name=solver_name)
        if not ok:
            continue
        for (j, w) in m_g.R_IDX:
            if j == name:
                fixed[(j, w)] = clean_commitment_value(pyo.value(m_g.r[j, w]))
    opts = StackOptions(fixed_commitments=fixed)
    m2 = _solve_case(params, data, mkt, opts, solver_name)
    s2 = revenue_summary(m2)
    rows.append({"case": "B2_greedy", **s2})
    print(f"B2 greedy sequential  net cost {s2['net_cost']:9.2f} GBP")

    # B3: full co-optimisation, uncertified ------------------------------------
    opts = StackOptions()
    m3 = _solve_case(params, data, mkt, opts, solver_name)
    s3 = revenue_summary(m3)
    rows.append({"case": "B3_uncertified", **s3})
    print(f"B3 co-opt uncertified net cost {s3['net_cost']:9.2f} GBP")

    # B4: full co-optimisation, certified ---------------------------------------
    m4, report = certify(params, data, mkt, opts, m3, solver_name)
    s4 = revenue_summary(m4)
    s4["cert_checks"] = report["checks"]
    s4["cert_failures"] = report["failures"]
    s4["phantom_gbp"] = report["phantom_gbp"]
    rows.append({"case": "B4_certified", **s4})
    print(f"B4 co-opt certified   net cost {s4['net_cost']:9.2f} GBP "
          f"(phantom {report['phantom_gbp']:.2f} GBP, "
          f"{report['failures']}/{report['checks']} checks failed)")

    # B5: market-sequential certified (gate order: day-ahead, then BM/DFS) ----
    # Stage 1 - the day-ahead gate: energy profile + availability commitments
    # (incl. seasonal DSO windows, committed before the day), with the
    # within-day products unavailable; certified as in B4.
    opts_da = StackOptions(enable_bm=False, enable_dfs=False)
    m5 = _solve_case(params, data, mkt, opts_da, solver_name)
    m5c, report5 = certify(params, data, mkt, opts_da, m5, solver_name)
    # Stage 2 - within-day recourse: BM/DFS fill the residual headroom of the
    # frozen stage-1 schedule and commitments.
    rev_bm5, rev_dfs5 = _sequential_recourse(m5c, mkt, StackOptions())
    s5 = revenue_summary(m5c)
    s5["rev_bm"] = rev_bm5
    s5["rev_dfs"] = rev_dfs5
    s5["net_cost"] = s5["net_cost"] - rev_bm5 - rev_dfs5
    s5["cert_checks"] = report5["checks"]
    s5["cert_failures"] = report5["failures"]
    s5["phantom_gbp"] = report5["phantom_gbp"]
    rows.append({"case": "B5_sequential", **s5})
    print(f"B5 market-sequential  net cost {s5['net_cost']:9.2f} GBP "
          f"(recourse BM {rev_bm5:.2f} + DFS {rev_dfs5:.2f} GBP; "
          f"sequencing gap vs B4 {s5['net_cost'] - s4['net_cost']:.2f} GBP)")

    df = pd.DataFrame(rows)
    df["value_vs_B0_gbp"] = s0["net_cost"] - df["net_cost"]
    out = config.RESULTS_DIR / f"benchmarks_{pd.Timestamp(date).date()}.csv"
    df.to_csv(out, index=False)
    print(f"\nBenchmark ladder written to {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()
    config.ensure_dirs()
    run_benchmarks(args.date, solver_name=args.solver)


if __name__ == "__main__":
    main()
