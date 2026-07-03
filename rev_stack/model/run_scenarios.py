"""Out-of-sample risk evaluation (Stage 3, light version).

Takes the deterministic first-stage commitments for a day and re-evaluates
them under price scenarios (mean-preserving spread scalings and level shifts
of the day-ahead curve, and utilisation shocks), with the physical dispatch
re-optimised per scenario. Reports the expected net cost and CVaR_q of the
outcome distribution - the foresight premium and risk metrics of Section 4.7
of the paper without the full two-stage model (noted there as Stage-3 work).

Usage (repo root, venv active):
    python -m rev_stack.model.run_scenarios --date 2025-01-15 --n 12

Output: rev_stack/results/scenarios_<date>.csv
"""
import argparse
import copy

import numpy as np
import pandas as pd

from . import config
from .facility import scaled_parameters, load_facility_data, build_facility_model
from .market_data import load_market_day
from .stack_model import (StackOptions, add_market_layer, solve, get_solver,
                          commitment_values)
from .postprocess import revenue_summary

Q = 0.95  # CVaR confidence level


def _perturbed_market(mkt, rng):
    m2 = copy.deepcopy(mkt)
    spread = rng.uniform(0.6, 1.6)      # mean-preserving spread scaling
    shift = rng.normal(0, 8.0)          # level shift (GBP/MWh)
    mean = m2.da_price[1:].mean()
    m2.da_price = np.concatenate(
        [[0.0], mean + spread * (m2.da_price[1:] - mean) + shift])
    m2.bm_offer = m2.bm_offer * spread + shift
    m2.bm_bid = m2.bm_bid * spread + shift
    util_shock = rng.uniform(0.5, 3.0)  # utilisation-frequency shock
    m2.phi = {k: min(1.0, v * util_shock) for k, v in m2.phi.items()}
    return m2


def run_scenarios(date, n=12, solver_name=None, seed=1):
    params = scaled_parameters()
    mkt = load_market_day(date)
    data = load_facility_data(params, mkt.da_price)
    if solver_name is None:
        solver_name, _ = get_solver()
    rng = np.random.default_rng(seed)

    # First stage: deterministic commitments under the central forecast.
    m = build_facility_model(params, data)
    add_market_layer(m, params, mkt, StackOptions())
    _, ok = solve(m, solver_name=solver_name)
    if not ok:
        raise RuntimeError("first-stage model infeasible")
    fixed = commitment_values(m)
    det = revenue_summary(m)["net_cost"]
    print(f"Deterministic (perfect foresight) net cost: {det:.2f} GBP")

    # Second stage: re-dispatch under each scenario with commitments fixed.
    rows = []
    for i in range(n):
        mkt_w = _perturbed_market(mkt, rng)
        data_w = load_facility_data(params, mkt_w.da_price)
        m_w = build_facility_model(params, data_w)
        add_market_layer(m_w, params, mkt_w,
                         StackOptions(fixed_commitments=fixed))
        _, ok = solve(m_w, solver_name=solver_name)
        if not ok:
            print(f"  scenario {i}: infeasible under fixed commitments "
                  f"(counts toward delivery risk)")
            rows.append({"scenario": i, "net_cost": np.nan, "feasible": False})
            continue
        c = revenue_summary(m_w)["net_cost"]
        rows.append({"scenario": i, "net_cost": c, "feasible": True})
        print(f"  scenario {i}: net cost {c:.2f} GBP")

    df = pd.DataFrame(rows)
    feas = df[df["feasible"]]["net_cost"]
    exp_cost = feas.mean()
    var_q = feas.quantile(Q)
    cvar = feas[feas >= var_q].mean()
    print(f"\nExpected net cost {exp_cost:.2f} GBP | "
          f"CVaR_{Q:.2f} {cvar:.2f} GBP | "
          f"foresight premium {exp_cost - det:.2f} GBP | "
          f"infeasible scenarios {int((~df['feasible']).sum())}/{n}")

    df["deterministic_net_cost"] = det
    df["expected_net_cost"] = exp_cost
    df["cvar"] = cvar
    out = config.RESULTS_DIR / f"scenarios_{pd.Timestamp(date).date()}.csv"
    df.to_csv(out, index=False)
    print(f"Scenario evaluation written to {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    config.ensure_dirs()
    run_scenarios(args.date, n=args.n, solver_name=args.solver, seed=args.seed)


if __name__ == "__main__":
    main()
