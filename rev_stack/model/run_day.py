"""Solve the revenue-stacking co-optimisation for a single trading day.

Usage (from the repo root, venv active):
    python -m rev_stack.model.run_day --date 2025-01-15
    python -m rev_stack.model.run_day --date 2025-07-10 --no-certify
    python -m rev_stack.model.run_day --date 2025-01-15 --cm-kw 2000 --solver scip

Outputs to rev_stack/results/day_<date>/:
    summary.csv, commitments.csv, revenue_by_product.csv, dispatch.csv,
    certification.csv (if certification ran)
"""
import argparse
import json
import time

import pandas as pd

from . import config
from .facility import scaled_parameters, load_facility_data, build_facility_model
from .market_data import load_market_day
from .stack_model import StackOptions, add_market_layer, solve, get_solver
from .certification import certify
from .postprocess import save_day_results


def run_one_day(date, opts: StackOptions = None, do_certify=True,
                solver_name=None, soc_floor=None, scale=None, verbose=True):
    """Build, solve and (optionally) certify one day. Returns (summary, m)."""
    opts = opts or StackOptions()
    params = scaled_parameters(scale=scale or config.SCALE, soc_floor=soc_floor)
    mkt = load_market_day(date)
    data = load_facility_data(params, mkt.da_price)
    if solver_name is None:
        solver_name, _ = get_solver()

    t0 = time.time()
    m = build_facility_model(params, data)
    add_market_layer(m, params, mkt, opts)
    _, ok = solve(m, solver_name=solver_name)
    if not ok:
        raise RuntimeError(f"Stacking model infeasible/unsolved for {date}")
    build_s = time.time() - t0

    report = None
    if do_certify:
        m, report = certify(params, data, mkt, opts, m, solver_name,
                            verbose=verbose)

    extra = {"date": str(pd.Timestamp(date).date()), "solver": solver_name,
             "solve_seconds": round(time.time() - t0, 1)}
    if report:
        extra.update({"cert_checks": report["checks"],
                      "cert_failures": report["failures"],
                      "phantom_gbp": round(report["phantom_gbp"], 2)})
    outdir = config.RESULTS_DIR / f"day_{pd.Timestamp(date).date()}"
    summary = save_day_results(outdir, m, params, summary_extra=extra)
    if report:
        with open(outdir / "certification.json", "w") as f:
            json.dump(report, f, indent=2)

    if verbose:
        print(f"\n=== {date} (solver: {solver_name}, "
              f"{extra['solve_seconds']} s) ===")
        for k in ("cost_energy", "rev_availability", "rev_utilisation",
                  "rev_bm", "rev_dfs", "rev_cm", "net_cost"):
            print(f"  {k:20s} {summary[k]:10.2f} GBP")
        if report:
            print(f"  certification: {report['checks']} checks, "
                  f"{report['failures']} failures, "
                  f"phantom = {report['phantom_gbp']:.2f} GBP")
        print(f"  results -> {outdir}")
    return summary, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--no-certify", action="store_true")
    ap.add_argument("--cm-kw", type=float, default=config.CM_DEFAULT_KW)
    ap.add_argument("--solver", default=None,
                    help="scip | appsi_highs (default: auto)")
    ap.add_argument("--soc-floor", type=float, default=None,
                    help="UPS resilience floor, e.g. 0.3 / 0.5 / 0.7")
    ap.add_argument("--scale", type=float, default=None,
                    help="facility scale in MW (default 10)")
    args = ap.parse_args()

    config.ensure_dirs()
    opts = StackOptions(cm_kw=args.cm_kw)
    run_one_day(args.date, opts=opts, do_certify=not args.no_certify,
                solver_name=args.solver, soc_floor=args.soc_floor,
                scale=args.scale)


if __name__ == "__main__":
    main()
