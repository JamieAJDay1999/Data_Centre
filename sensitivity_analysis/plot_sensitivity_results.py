"""
plot_sensitivity_results.py — publication figures for the paper's new
"Sensitivity Analysis" subsection, from the CSVs produced by
sensitivity_sweep.py.

Produces (in static/images/sensitivity_outputs/):
  * sensitivity_tornado.{png,pdf}   — standalone single-column tornado of the
        Scenario 2 cost saving (%). Always produced from Tier 1.
  * sensitivity_tau.{png,pdf}       — standalone duration-sensitivity panel of
        max feasible duration tau (h) vs parameter level for the three
        Scenario 3 reference products. Produced when Tier 2 results exist.
  * sensitivity_two_panel.{png,pdf} — the (a)+(b) composite (double-column).

Design: tornado bars are split at the base-case saving; colour encodes the
parameter level (x0.5 vs x1.5), so no per-end multiplier text is needed and
nothing collides with the axis labels. Fonts are serif to sit cleanly next to
LaTeX body text.

Usage (after running the sweep tiers):
    python -m sensitivity_analysis.plot_sensitivity_results
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "static" / "data" / "sensitivity_outputs"
IMAGE_DIR = ROOT / "static" / "images" / "sensitivity_outputs"

TIER1_CSV = SWEEP_ROOT / "tier1_cost_sensitivity.csv"
TIER2_CSV = SWEEP_ROOT / "tier2_duration_sensitivity.csv"

PARAM_LABELS = {
    "flex_share": "Flexible workload share",
    "price_day": "Price day",
    "tes_capacity": "TES energy capacity",
    "ups_capacity": "UPS energy capacity",
    "cold_aisle_temp": "Cold-aisle upper limit",
}
# Base-case x-position per parameter (multiplier 1.0, or 23 C for temperature).
PARAM_BASE_X = {
    "flex_share": 1.0, "price_day": 0.0, "tes_capacity": 1.0,
    "ups_capacity": 1.0, "cold_aisle_temp": 23.0,
}

# Muted, print-safe diverging pair (low level / high level).
COL_LOW = "#c1683f"    # x0.5 — reduced resource / volatility
COL_HIGH = "#3a6ea5"   # x1.5 — increased resource / volatility
COL_BASE = "#222222"

PROBE_STYLE = {
    "P1": {"color": "#3a6ea5", "marker": "o", "label": r"P1:  $-$100 kW @ 00:15"},
    "P2": {"color": "#c1683f", "marker": "s", "label": r"P2:  $-$100 kW @ 17:30"},
    "P3": {"color": "#4c9a5d", "marker": "^", "label": r"P3:  $+$100 kW @ 18:00"},
}


def set_style():
    """Publication rcParams: serif family, restrained sizes and line weights."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# --- Panel (a): tornado --------------------------------------------------------
def _tornado_entries(tier1: pd.DataFrame):
    base = float(tier1.loc[tier1["param"] == "base", "saving_pct"].iloc[0])
    entries = []
    for param, grp in tier1[tier1["param"] != "base"].groupby("param"):
        lo = grp.loc[grp["saving_pct"].idxmin()]
        hi = grp.loc[grp["saving_pct"].idxmax()]
        entries.append({
            "param": param,
            "lo": float(lo["saving_pct"]), "lo_lv": float(lo["level"]),
            "hi": float(hi["saving_pct"]), "hi_lv": float(hi["level"]),
        })
    entries.sort(key=lambda e: e["hi"] - e["lo"])  # widest bar on top
    return base, entries


def plot_tornado(ax, tier1: pd.DataFrame, panel_tag: str = ""):
    base, entries = _tornado_entries(tier1)
    n = len(entries)
    bar_h = 0.6

    for i, e in enumerate(entries):
        # Split each bar at the base saving: left segment = low level, right = high.
        ax.barh(i, base - e["lo"], left=e["lo"], height=bar_h, color=COL_LOW,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.barh(i, e["hi"] - base, left=base, height=bar_h, color=COL_HIGH,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{e['lo']:.1f}", (e["lo"], i), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color="#333333")
        ax.annotate(f"{e['hi']:.1f}", (e["hi"], i), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color="#333333")

    ax.axvline(base, color=COL_BASE, linewidth=1.1, zorder=4)

    lo_min = min(e["lo"] for e in entries)
    hi_max = max(e["hi"] for e in entries)
    span = hi_max - lo_min
    ax.set_xlim(lo_min - 0.16 * span, hi_max + 0.11 * span)
    ax.set_ylim(-0.65, n - 0.35)

    ax.set_yticks(range(n))
    ax.set_yticklabels([PARAM_LABELS.get(e["param"], e["param"]) for e in entries])
    ax.set_xlabel("Daily cost saving (%)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [
        Patch(facecolor=COL_LOW, label="lower saving case"),
        Patch(facecolor=COL_HIGH, label="higher saving case"),
        Line2D([0], [0], color=COL_BASE, linewidth=1.1,
               label=f"base ({base:.2f}%)"),
    ]
    if panel_tag:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.32),
                  bbox_transform=ax.transAxes, frameon=False, ncol=3,
                  handlelength=1.2, columnspacing=1.2, handletextpad=0.5)
        ax.set_title(panel_tag, loc="left", fontweight="bold")
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.45, -0.05),
                  bbox_transform=ax.transAxes, frameon=False, ncol=1,
                  handlelength=1.2, handletextpad=0.5)


# --- Panel (b): duration probes ------------------------------------------------
def plot_tau_panels(fig, axes, tier2: pd.DataFrame, params_order: list, panel_tag=""):
    base_rows = tier2[tier2["param"] == "base"]
    base_tau = {r["probe"]: float(r["tau_hours"]) for _, r in base_rows.iterrows()}

    for ax, param in zip(axes, params_order):
        grp = tier2[tier2["param"] == param]
        for probe, style in PROBE_STYLE.items():
            pts = grp[grp["probe"] == probe][["level", "tau_hours"]].copy()
            if probe in base_tau:
                pts = pd.concat([pts, pd.DataFrame(
                    [{"level": PARAM_BASE_X[param], "tau_hours": base_tau[probe]}])])
            pts = pts.dropna().sort_values("level")
            if pts.empty:
                continue
            ax.plot(pts["level"], pts["tau_hours"], marker=style["marker"],
                    color=style["color"], markersize=4, markeredgecolor="white",
                    markeredgewidth=0.5, linewidth=1.3, label=style["label"], zorder=3)
        if param == "cold_aisle_temp":
            ax.set_xlabel("Upper limit (°C)")
        elif param == "price_day":
            ax.set_xlabel("Price day")
            price_rows = (
                tier2[tier2["param"] == "price_day"][["level", "price_scenario"]]
                .drop_duplicates()
                .sort_values("level")
            )
            ticks = [PARAM_BASE_X[param]] + [float(v) for v in price_rows["level"]]
            labels = ["Base"] + [
                str(v).replace("_", "\n").replace("day", "").strip()
                for v in price_rows["price_scenario"]
            ]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=35, ha="right")
        else:
            ax.set_xlabel("Multiplier")
            ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.8, zorder=1)
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=8)
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.12)

    axes[0].set_ylabel(r"$\tau$ (hours)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.6)
    tag = "(b) " if panel_tag else ""
    axes[0].annotate(f"{tag}Duration sensitivity", xy=(0, 1.18),
                     xycoords="axes fraction", fontweight="bold", fontsize=9,
                     ha="left", va="bottom")


# --- Orchestration -------------------------------------------------------------
def _savefig(fig, name):
    for ext in ("png", "pdf"):
        out = IMAGE_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")


def main():
    set_style()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    tier1 = pd.read_csv(TIER1_CSV) if TIER1_CSV.exists() else None
    tier2 = pd.read_csv(TIER2_CSV) if TIER2_CSV.exists() else None
    if tier1 is None and tier2 is None:
        raise SystemExit(
            f"No sweep results found under {SWEEP_ROOT}. "
            "Run python -m sensitivity_analysis.sensitivity_sweep first."
        )

    tau_params = []
    if tier2 is not None:
        tau_params = [p for p in PARAM_LABELS if p in set(tier2["param"])]

    # 1) Standalone tornado (single-column: ~3.5in wide). Extra figure height
    # plus a reserved bottom margin (via rect) make room for the legend that
    # plot_tornado() places below the x-axis label.
    if tier1 is not None:
        fig, ax = plt.subplots(figsize=(3.5, 2.75))
        plot_tornado(ax, tier1)
        fig.tight_layout()
        _savefig(fig, "sensitivity_tornado")
        plt.close(fig)

    # 2) Standalone tau panel (double-column width).
    if tau_params:
        fig, axes = plt.subplots(1, len(tau_params), figsize=(7.16, 2.3),
                                 sharey=True)
        axes = [axes] if len(tau_params) == 1 else list(axes)
        plot_tau_panels(fig, axes, tier2, tau_params)
        fig.tight_layout(rect=(0, 0.06, 1, 0.98))
        _savefig(fig, "sensitivity_tau")
        plt.close(fig)

    # 3) Composite (a)+(b) for a single double-column figure.
    if tier1 is not None and tau_params:
        fig = plt.figure(figsize=(7.16, 5.0))
        gs = fig.add_gridspec(2, len(tau_params), height_ratios=[1.0, 1.05],
                              hspace=0.75, wspace=0.28)
        ax_t = fig.add_subplot(gs[0, :])
        plot_tornado(ax_t, tier1, panel_tag="(a)")
        axes = [fig.add_subplot(gs[1, i]) for i in range(len(tau_params))]
        for a in axes[1:]:
            a.sharey(axes[0])
            plt.setp(a.get_yticklabels(), visible=False)
        plot_tau_panels(fig, axes, tier2, tau_params, panel_tag="(b)")
        _savefig(fig, "sensitivity_two_panel")
        plt.close(fig)

    # Compact numeric summary for the paper text / response letter.
    if tier1 is not None:
        print("\n--- Tier 1 summary (cost saving %) ---")
        cols = [c for c in ["case_id", "level", "base_cost_gbp", "opt_cost_gbp",
                            "saving_pct", "price_scenario",
                            "realized_flex_multiplier"] if c in tier1.columns]
        print(tier1[cols].to_string(index=False))
    if tier2 is not None:
        print("\n--- Tier 2 summary (tau, hours) ---")
        print(tier2.pivot_table(index=["param", "level"], columns="probe",
                                values="tau_hours").to_string())


if __name__ == "__main__":
    main()
