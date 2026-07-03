"""Benchmark ladder B0-B4 (Section 4.9 of the paper): the value of
co-optimisation and certification.

  B0  energy-only (Paper 1 cost optimisation; no market participation)
  B1  dedicated assets (UPS -> Dynamic Containment only; no other products)
  B2  greedy sequential stacking (products committed one at a time by
      standalone value, each taking the residual capability)
  B3  full co-optimisation, uncertified
  B4  full co-optimisation, certified (the paper's framework)

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
