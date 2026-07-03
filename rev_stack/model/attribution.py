"""Per-asset attribution: leave-one-out marginal values and synergies (RQ2).

For a given day, solves the full stack, then re-solves with each asset
excluded from market participation (leave-one-out) and with each asset alone
(standalone). The difference between the sum of standalone values and the
portfolio value measures cross-asset synergy.

Usage (repo root, venv active):
    python -m rev_stack.model.attribution --date 2025-01-15

Output: rev_stack/results/attribution_<date>.csv
"""
import argparse

import pandas as pd

from . import config
from .facility import scaled_parameters, load_facility_data, build_facility_model
from .market_data import load_market_day
from .stack_model import StackOptions, add_market_layer, solve, get_solver
from .postprocess import revenue_summary

ASSETS = ("IT", "UPS", "CL")


def _net_cost(params, data, mkt, opts, solver_name):
    m = build_facility_model(params, data)
    add_market_layer(m, params, mkt, opts)
    _, ok = solve(m, solver_name=solver_name)
    if not ok:
        raise RuntimeError("attribution case infeasible")
    return revenue_summary(m)["net_cost"]


def run_attribution(date, solver_name=None):
    params = scaled_parameters()
    mkt = load_market_day(date)
    data = load_facility_data(params, mkt.da_price)
    if solver_name is None:
        solver_name, _ = get_solver()

    full = _net_cost(params, data, mkt, StackOptions(), solver_name)
    none = _net_cost(params, data, mkt,
                     StackOptions(excluded_assets=ASSETS, enable_bm=False,
                                  enable_dfs=False, enable_dso=False),
                     solver_name)
    portfolio_value = none - full

    rows = []
    standalone_sum = 0.0
    for a in ASSETS:
        others = tuple(x for x in ASSETS if x != a)
        loo = _net_cost(params, data, mkt,
                        StackOptions(excluded_assets=(a,)), solver_name)
        solo = _net_cost(params, data, mkt,
                         StackOptions(excluded_assets=others), solver_name)
        marginal = loo - full          # value the asset adds to the stack
        standalone = none - solo       # value the asset earns alone
        standalone_sum += standalone
        rows.append({"asset": a, "marginal_value_gbp": round(marginal, 2),
                     "standalone_value_gbp": round(standalone, 2)})
        print(f"  {a:4s} marginal {marginal:8.2f}  standalone {standalone:8.2f} GBP")

    synergy = portfolio_value - standalone_sum
    rows.append({"asset": "PORTFOLIO", "marginal_value_gbp": round(portfolio_value, 2),
                 "standalone_value_gbp": round(standalone_sum, 2)})
    print(f"  portfolio value {portfolio_value:.2f} GBP; "
          f"sum of standalone {standalone_sum:.2f} GBP; "
          f"synergy {synergy:.2f} GBP")

    df = pd.DataFrame(rows)
    df["synergy_gbp"] = round(synergy, 2)
    out = config.RESULTS_DIR / f"attribution_{pd.Timestamp(date).date()}.csv"
    df.to_csv(out, index=False)
    print(f"Attribution written to {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()
    config.ensure_dirs()
    run_attribution(args.date, solver_name=args.solver)


if __name__ == "__main__":
    main()
