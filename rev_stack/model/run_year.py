"""Annual (multi-day) revenue-stacking simulation - Stage 2 of the analysis.

Each day is an independent solve (the facility model enforces cyclic UPS/TES
end states, matching Paper 1), so days parallelise across processes.

Usage (repo root, venv active):
    python -m rev_stack.model.run_year --days 30                 # sample run
    python -m rev_stack.model.run_year --all --no-certify        # full year, fast
    python -m rev_stack.model.run_year --days 60 --jobs 4        # parallel
    python -m rev_stack.model.run_year --days 30 --cm-sweep 0 2000 5000

Outputs:
    rev_stack/results/annual_summary.csv          one row per day
    rev_stack/results/annual_by_product.csv       product revenue per day
    rev_stack/results/cm_sweep.csv                if --cm-sweep given
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from . import config
from .market_data import available_dates
from .stack_model import StackOptions
from .run_day import run_one_day


def _worker(date, cm_kw, do_certify, solver_name):
    from .stack_model import StackOptions  # re-import inside process
    try:
        opts = StackOptions(cm_kw=cm_kw)
        summary, m = run_one_day(date, opts=opts, do_certify=do_certify,
                                 solver_name=solver_name, verbose=False)
        from .postprocess import availability_revenue_by_product
        by_prod = availability_revenue_by_product(m)
        by_prod["date"] = str(date)
        return summary, by_prod, None
    except Exception as e:  # keep the sweep alive; report failures
        return {"date": str(date), "error": str(e)}, None, str(e)


def run_days(dates, cm_kw=0.0, do_certify=True, solver_name=None, jobs=1,
             tag=""):
    summaries, by_products = [], []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, d, cm_kw, do_certify, solver_name): d
                    for d in dates}
            for fut in as_completed(futs):
                s, bp, err = fut.result()
                summaries.append(s)
                if bp is not None:
                    by_products.append(bp)
                print(f"  done {s.get('date')}"
                      + (f"  [ERROR {err}]" if err else ""))
    else:
        for d in dates:
            s, bp, err = _worker(d, cm_kw, do_certify, solver_name)
            summaries.append(s)
            if bp is not None:
                by_products.append(bp)
            print(f"  done {s.get('date')}"
                  + (f"  [ERROR {err}]" if err else ""))

    df = pd.DataFrame(summaries)
    suffix = f"_{tag}" if tag else ""
    df.to_csv(config.RESULTS_DIR / f"annual_summary{suffix}.csv", index=False)
    if by_products:
        pd.concat(by_products, ignore_index=True).to_csv(
            config.RESULTS_DIR / f"annual_by_product{suffix}.csv", index=False)
    return df


def print_annual(df, cm_kw, scale_mw=config.SCALE):
    ok = df[df.get("error").isna()] if "error" in df else df
    n = len(ok)
    if n == 0:
        print("No successful days.")
        return
    days_per_year = 365.0
    f = days_per_year / n / 1000.0  # scale sample to kGBP/yr
    print(f"\n=== Annual estimate from {n} days (CM = {cm_kw:.0f} kW) ===")
    total_rev = 0.0
    for k, label in [("rev_availability", "Availability"),
                     ("rev_utilisation", "Expected utilisation"),
                     ("rev_bm", "Balancing Mechanism"),
                     ("rev_dfs", "DFS"), ("rev_cm", "Capacity Market")]:
        v = ok[k].sum() * f
        total_rev += v
        print(f"  {label:22s} {v:8.1f} kGBP/yr "
              f"({v / scale_mw:6.2f} kGBP/MW/yr)")
    cost = ok["cost_energy"].sum() * f
    net = ok["net_cost"].sum() * f
    print(f"  {'Energy + network cost':22s} {cost:8.1f} kGBP/yr")
    print(f"  {'Total market revenue':22s} {total_rev:8.1f} kGBP/yr "
          f"({total_rev / scale_mw:6.2f} kGBP/MW/yr)")
    print(f"  {'Net cost':22s} {net:8.1f} kGBP/yr")
    if "phantom_gbp" in ok:
        print(f"  {'Phantom flexibility':22s} "
              f"{ok['phantom_gbp'].sum() * f:8.1f} kGBP/yr")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30,
                    help="number of days, evenly sampled across the year")
    ap.add_argument("--all", action="store_true", help="run every day")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-certify", action="store_true")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--cm-kw", type=float, default=config.CM_DEFAULT_KW)
    ap.add_argument("--cm-sweep", type=float, nargs="+", default=None,
                    help="run the CM outer sweep over these kW values")
    args = ap.parse_args()

    config.ensure_dirs()
    dates = available_dates()
    if not args.all:
        step = max(1, len(dates) // args.days)
        dates = dates[::step][:args.days]

    if args.cm_sweep:
        rows = []
        for cm in args.cm_sweep:
            print(f"\n--- CM sweep: {cm:.0f} kW ---")
            df = run_days(dates, cm_kw=cm, do_certify=not args.no_certify,
                          solver_name=args.solver, jobs=args.jobs,
                          tag=f"cm{int(cm)}")
            ok = df[df.get("error").isna()] if "error" in df else df
            rows.append({"cm_kw": cm, "n_days": len(ok),
                         "mean_net_cost_gbp": ok["net_cost"].mean(),
                         "mean_cm_rev_gbp": ok["rev_cm"].mean()})
            print_annual(df, cm)
        pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "cm_sweep.csv",
                                  index=False)
    else:
        df = run_days(dates, cm_kw=args.cm_kw,
                      do_certify=not args.no_certify,
                      solver_name=args.solver, jobs=args.jobs)
        print_annual(df, args.cm_kw)


if __name__ == "__main__":
    main()
