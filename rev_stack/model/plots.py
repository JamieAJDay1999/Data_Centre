"""Figures from saved results (Section 7 of doc 04 / the paper's figure list).

Usage (repo root, venv active; run the analysis scripts first):
    python -m rev_stack.model.plots                     # all available figures
    python -m rev_stack.model.plots --date 2025-01-15   # day-specific figures

Reads rev_stack/results/, writes PNGs to rev_stack/results/figures/.
"""
import argparse
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.style.use("seaborn-v0_8-whitegrid")
LABEL, TICK, LEGEND = 14, 12, 11


def _save(fig, name):
    out = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def plot_revenue_waterfall():
    """Annual revenue stack from annual_summary.csv (the 'money chart')."""
    path = config.RESULTS_DIR / "annual_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "error" in df:
        df = df[df["error"].isna()]
    n = len(df)
    if n == 0:
        return
    f = 365.0 / n / 1000.0  # sample -> kGBP/yr
    components = [("Availability", df["rev_availability"].sum() * f),
                  ("Utilisation", df["rev_utilisation"].sum() * f),
                  ("Balancing Mech.", df["rev_bm"].sum() * f),
                  ("DFS", df["rev_dfs"].sum() * f),
                  ("Capacity Market", df["rev_cm"].sum() * f)]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [c[0] for c in components] + ["Total"]
    vals = [c[1] for c in components]
    cum = np.concatenate([[0], np.cumsum(vals)])
    for i, v in enumerate(vals):
        ax.bar(i, v, bottom=cum[i], color="#2ca02c" if v >= 0 else "#d62728")
    ax.bar(len(vals), cum[-1], color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, fontsize=TICK)
    ax.set_ylabel("Revenue (kGBP/yr)", fontsize=LABEL)
    ax.set_title(f"Annual market revenue stack ({n}-day sample, scaled)",
                 fontsize=LABEL)
    _save(fig, "revenue_waterfall.png")


def plot_stack_composition():
    """Committed MW by product x EFA block (average across days)."""
    path = config.RESULTS_DIR / "annual_by_product.csv"
    day_files = sorted(glob.glob(str(config.RESULTS_DIR / "day_*" /
                                     "commitments.csv")))
    frames = [pd.read_csv(fp) for fp in day_files]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    efa = df[df["granularity"] == "efa"]
    if efa.empty:
        return
    pivot = efa.pivot_table(index="product", columns="window",
                            values="committed_kw", aggfunc="mean").fillna(0) / 1000.0
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"EFA {c}" for c in pivot.columns], fontsize=TICK)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=TICK)
    for (i, j), v in np.ndenumerate(pivot.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v < pivot.values.max() * 0.6 else "black",
                fontsize=9)
    fig.colorbar(im, label="Mean committed capacity (MW)")
    ax.set_title("Average stack composition by product and EFA block",
                 fontsize=LABEL)
    _save(fig, "stack_composition_heatmap.png")


def plot_asset_allocation():
    """Aggregate committed capacity by backing asset."""
    day_files = sorted(glob.glob(str(config.RESULTS_DIR / "day_*" /
                                     "commitments.csv")))
    frames = [pd.read_csv(fp) for fp in day_files]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby("product")[["alloc_it_kw", "alloc_ups_kw",
                                 "alloc_cl_kw"]].mean() / 1000.0
    fig, ax = plt.subplots(figsize=(9, 5))
    agg.plot(kind="bar", stacked=True, ax=ax,
             color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_ylabel("Mean committed capacity (MW)", fontsize=LABEL)
    ax.set_xlabel("")
    ax.legend(["IT workload", "UPS", "Cooling/TES"], fontsize=LEGEND)
    ax.set_title("Which assets back which products", fontsize=LABEL)
    ax.tick_params(axis="x", rotation=30, labelsize=TICK)
    _save(fig, "asset_allocation.png")


def plot_day_dispatch(date):
    """Slot-level dispatch and market positions for one day."""
    ddir = config.RESULTS_DIR / f"day_{pd.Timestamp(date).date()}"
    path = ddir / "dispatch.csv"
    if not path.exists():
        print(f"  no dispatch results for {date} - run run_day first")
        return
    df = pd.read_csv(path)
    hours = (df["slot"] - 1) / 4.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(hours, df["P_grid_kw"] / 1000.0, color="crimson",
             label="Grid draw")
    ax1.fill_between(hours,
                     (df["P_grid_kw"] - df["committed_down_kw"]) / 1000.0,
                     (df["P_grid_kw"] + df["committed_up_kw"]) / 1000.0,
                     alpha=0.25, color="grey", label="Committed band")
    ax1.set_ylabel("Power (MW)", fontsize=LABEL)
    ax1b = ax1.twinx()
    ax1b.plot(hours, df["da_price"], color="royalblue", alpha=0.7,
              label="DA price")
    ax1b.set_ylabel("Price (GBP/MWh)", color="royalblue", fontsize=LABEL)
    ax1.legend(loc="upper left", fontsize=LEGEND)
    ax2.plot(hours, df["E_ups_kwh"] / 1000.0, label="UPS (MWh)")
    ax2.plot(hours, df["E_tes_kwh"] / 1000.0, label="TES (MWh, thermal)")
    ax2.set_ylabel("Stored energy (MWh)", fontsize=LABEL)
    ax2.set_xlabel("Hour of day", fontsize=LABEL)
    ax2.legend(fontsize=LEGEND)
    fig.suptitle(f"Dispatch and committed flexibility band - {date}",
                 fontsize=LABEL)
    _save(fig, f"dispatch_{pd.Timestamp(date).date()}.png")


def plot_benchmark_ladder(date):
    path = config.RESULTS_DIR / f"benchmarks_{pd.Timestamp(date).date()}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["case"], df["value_vs_B0_gbp"], color="#1f77b4")
    ax.set_ylabel("Daily value vs energy-only (GBP)", fontsize=LABEL)
    ax.tick_params(axis="x", rotation=20, labelsize=TICK)
    ax.set_title(f"Value of co-optimisation and certification - {date}",
                 fontsize=LABEL)
    _save(fig, f"benchmark_ladder_{pd.Timestamp(date).date()}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    config.ensure_dirs()
    print("Generating figures...")
    plot_revenue_waterfall()
    plot_stack_composition()
    plot_asset_allocation()
    if args.date:
        plot_day_dispatch(args.date)
        plot_benchmark_ladder(args.date)
    else:
        for d in sorted(glob.glob(str(config.RESULTS_DIR / "day_*"))):
            date = pathlib.Path(d).name.replace("day_", "")
            plot_day_dispatch(date)
            plot_benchmark_ladder(date)


if __name__ == "__main__":
    main()
