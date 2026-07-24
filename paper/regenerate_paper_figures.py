"""
regenerate_paper_figures.py — formatting-only regeneration of the paper's
result figures in response to the reviewer comments on figure readability.

No optimisation is re-run: every figure is rebuilt from CSVs already produced
by the pipeline, so the underlying numbers are unchanged. Only presentation is
improved:

  * Figure 5 — hatch the discharge (negative) bars so UPS/TES charge and
      discharge are distinguishable despite sharing a colour (R5-8).
  * Figure 6 — reversed colour map (darker = longer duration), a thick divider
      between downward (positive dP) and upward (negative dP) rows, larger cell
      annotations, and clock-time axis labels (R2-C019/C020/C021).
  * Figures 7 & 8 — (a)-(f) panel labels and larger fonts (R2-2, R2-C018).

Outputs overwrite paper/images/Figure_5.png, Figure_6.png, Figure_7.png,
Figure_8.png.

Usage (from the repo root):
    python paper/regenerate_paper_figures.py
"""
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "static" / "data"
FLEX_DIR = DATA / "flexibility_outputs"
OUT = ROOT / "paper" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def clock(hours: float) -> str:
    """Decimal hours (may exceed 24) -> HH:MM on a 24-h wrap."""
    total_min = int(round(hours * 60))
    return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"


# --------------------------------------------------------------------------- #
# Figure 5 — optimised component power, with hatched discharge bars           #
# --------------------------------------------------------------------------- #
def figure5():
    df = pd.read_csv(DATA / "optimisation_outputs" / "optimised_baseline.csv")
    df = df.reset_index(drop=True)
    x_hours = (df["Time_Slot_EXT"] - 1) / 4.0
    bar_width = 1.0 / 4.0

    plt.style.use("seaborn-v0_8-whitegrid")
    LABEL_FS, LEGEND_FS, TICK_FS = 16, 12.5, 14
    fig, ax = plt.subplots(figsize=(14, 8))

    consume = {
        "IT Power": "P_Grid_IT_kW",
        "CRAC Power": "P_Chiller_HVAC_kW",
        "UPS Charging Power": "P_UPS_Charge_kW",
        "TES Charging Power": "P_Chiller_TES_kW",
        "Other DC Power": "P_Grid_Other_kW",
    }
    colors_consume = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]

    bottom = pd.Series(0.0, index=df.index)
    for (label, col), color in zip(consume.items(), colors_consume):
        ax.bar(x_hours, df[col], width=bar_width, bottom=bottom, color=color,
               label=label, alpha=0.85, edgecolor="none")
        bottom += df[col]

    # Discharge (negative) bars: same colours as their charge counterparts but
    # hatched, so positive/negative components are no longer ambiguous.
    ups_disch = -df["P_UPS_Discharge_kW"]
    tes_disch = -(df["Q_Discharge_TES_Watts"] / (1000.0 * 6))
    neg_bottom = pd.Series(0.0, index=df.index)
    for series, color, label in (
        (ups_disch, colors_consume[2], "UPS Discharge"),
        (tes_disch, colors_consume[3], "TES Discharge"),
    ):
        ax.bar(x_hours, series, width=bar_width, bottom=neg_bottom, color=color,
               label=label, alpha=0.85, hatch="////", edgecolor="white", linewidth=0.0)
        neg_bottom += series

    ax.plot(x_hours, df["P_Total_kW_Nominal"], label="Base DC Power",
            linestyle=":", color="black", linewidth=2.5)

    ax.set_ylabel("Power (kW)", fontsize=LABEL_FS)
    ax.set_xlabel("Time (Hours)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.set_xlim(x_hours.min() - bar_width, x_hours.max() + bar_width)
    ax.set_ylim(bottom=neg_bottom.min() * 1.1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.axhline(0, color="black", linewidth=0.8)

    # Two-group legend: solid = grid draw / charge, hatched = discharge.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=LEGEND_FS, ncol=2,
              framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "Figure_5.png", dpi=200)
    plt.close(fig)
    print("  -> Figure_5.png")


# --------------------------------------------------------------------------- #
# Figure 6 — flexibility duration heatmap                                     #
# --------------------------------------------------------------------------- #
def figure6():
    df = pd.read_csv(FLEX_DIR / "flex_duration_results_full.csv")
    heat = df.pivot(index="Flex_Magnitude_kW", columns="Timestep_Hours",
                    values="Max_Duration_Hours")
    heat = heat.reindex(sorted(heat.columns), axis=1)
    heat = heat.reindex(sorted(heat.index), axis=0)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        heat, cmap="mako_r", linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Max Duration (hours)"}, square=False,
        annot=True, fmt=".2f", annot_kws={"size": 11}, ax=ax,
    )
    ax.invert_yaxis()

    # Thick divider between upward (negative dP) and downward (positive dP).
    n_upward = int((heat.index < 0).sum())
    ax.axhline(n_upward, color="black", linewidth=3.5)

    # Clock-time x labels so the temporal sampling is unambiguous.
    ax.set_xticklabels([clock(float(t.get_text())) for t in ax.get_xticklabels()],
                       rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    ax.set_xlabel("Start Time $t_0$ (HH:MM)", fontsize=18)
    ax.set_ylabel(r"Flexibility Magnitude $\Delta P$ (kW)", fontsize=18)
    ax.figure.axes[-1].yaxis.label.set_size(16)  # colourbar label

    fig.tight_layout()
    fig.savefig(OUT / "Figure_6.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> Figure_6.png")


# --------------------------------------------------------------------------- #
# Figures 7 & 8 — component contribution grids                                #
# --------------------------------------------------------------------------- #
COL_MAP = {
    "P_IT_Total_kW_diff": "IT", "P_Chiller_HVAC_kW_diff": "Chiller CRAC",
    "P_Chiller_TES_kW_diff": "Chiller TES", "P_UPS_NET_kw_Diff": "UPS charge",
}
PALETTE = {"IT": "#0072B2", "Chiller CRAC": "#E69F00",
           "Chiller TES": "#009E73", "UPS charge": "#56B4E9"}


def _load_detail(ts, mag):
    tag = f"flexneg{-mag}" if mag < 0 else f"flex{mag}"
    path = FLEX_DIR / f"flex_duration_detailed_results_ts{ts}_{tag}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path).set_index("Time_Slot_EXT")
    return d


def flex_grid(timesteps, magnitudes, out_name):
    sorted_mags = sorted(magnitudes, reverse=True)
    n_rows, n_cols = len(sorted_mags), len(timesteps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                             sharey=True, squeeze=False)
    series_order = list(COL_MAP.values())

    # Collect windows first to fix a shared x-extent.
    panels, global_max = {}, 0
    for ci, ts in enumerate(timesteps):
        for ri, mag in enumerate(sorted_mags):
            d = _load_detail(ts, mag)
            if d is None:
                continue
            dur_steps = max(0, len(d) - 12)
            global_max = max(global_max, dur_steps)
            panels[(ri, ci)] = (d, ts, mag, dur_steps)

    LETTERS = "abcdefghijkl"
    for (ri, ci), (d, ts, mag, dur_steps) in panels.items():
        ax = axes[ri, ci]
        window = [t for t in d.index if ts <= t < ts + dur_steps]
        cols = [c for c in COL_MAP if c in d.columns]
        diff = d.loc[window, cols].rename(columns=COL_MAP) if window else pd.DataFrame()

        if not diff.empty:
            x = np.arange(len(diff))
            pos_b = np.zeros(len(diff)); neg_b = np.zeros(len(diff))
            for name in series_order:
                if name not in diff.columns:
                    continue
                s = diff[name].values
                sp, sn = np.clip(s, 0, None), np.clip(s, None, 0)
                ax.bar(x, sp, width=0.82, bottom=pos_b, color=PALETTE[name],
                       edgecolor="white", linewidth=0.6)
                pos_b += sp
                ax.bar(x, sn, width=0.82, bottom=neg_b, color=PALETTE[name],
                       edgecolor="white", linewidth=0.6)
                neg_b += sn
            ax.scatter(x, diff.sum(axis=1).values, marker="x", s=64,
                       color="#444444", linewidths=1.6)
        ax.axhline(mag, color="black", alpha=0.55, linewidth=2, linestyle="--")
        ax.axhline(0, color="#BBBBBB", linewidth=1)
        ax.grid(axis="y", linestyle=":", color="#E0E0E0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01)
        ax.tick_params(axis="y", labelsize=13)

        if global_max > 0:
            xs = np.arange(global_max)
            hours = np.arange(1, global_max + 1) / 4.0
            step = max(1, len(xs) // 8)
            ax.set_xlim(-0.5, global_max - 0.5)
            ax.set_xticks(xs[::step])
            ax.set_xticklabels([f"{h:.2f}" for h in hours[::step]],
                               rotation=45, ha="right", fontsize=12)

        # Panel label (a)-(f) in reading order (row-major).
        letter = LETTERS[ri * n_cols + ci]
        ax.annotate(f"({letter})", xy=(0.012, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=17, fontweight="bold")

    for r in range(n_rows):
        axes[r, 0].set_ylabel(f"{sorted_mags[r]} kW", rotation=0, va="center",
                              ha="right", labelpad=6, fontsize=15)
    for ci, ts in enumerate(timesteps):
        h, m = int(ts // 4), int((ts % 4) * 15)
        axes[0, ci].set_title(f"Start Time: {h:02d}:{m:02d}", fontsize=15,
                              pad=4, y=1.02)

    swatches = [Patch(facecolor=c, edgecolor="w", label=n) for n, c in PALETTE.items()]
    extra = [Line2D([0], [0], ls="--", c="k", alpha=0.55, lw=2, label="Flex target"),
             Line2D([0], [0], marker="x", ls="None", c="#444", ms=9, label="Sum of diffs")]
    fig.legend(handles=swatches + extra, loc="upper left",
               bbox_to_anchor=(0.87, 0.95), borderaxespad=0, title="Legend",
               fontsize=13, title_fontsize=15)

    fig.text(0.5, 0.06, "Flexibility Duration (Hours)", ha="center",
             va="center", fontsize=18, weight="bold")
    fig.text(0.015, 0.5, "Magnitude (kW)", ha="center", va="center",
             rotation="vertical", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.14,
                        wspace=0.08, hspace=0.14)
    fig.savefig(OUT / out_name, dpi=140)
    plt.close(fig)
    print(f"  -> {out_name}")


if __name__ == "__main__":
    print("Regenerating paper figures (formatting only)...")
    figure5()
    figure6()
    flex_grid([15, 65], [-100, -150, -200], "Figure_7.png")
    flex_grid([15, 65], [75, 50, 25], "Figure_8.png")
    print("Done.")
