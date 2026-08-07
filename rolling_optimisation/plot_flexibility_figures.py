"""Draw the representative-day flexibility figures from stored results.

The sweep itself is expensive, so plotting is separated from solving. Both the
sweep (``run_representative_day_flexibility``) and a direct run of this module
produce identical figures; running this module re-reads the cached per-cell
dispatch and the flexibility grid, so figure design can be revised without
re-solving a single MILP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

from figure_style import (  # noqa: E402
    COMPONENT_COLOURS,
    DARK_GREY,
    DURATION_CMAP,
    GREY,
    TEXT_WIDTH,
    VERMILION,
    legend_above,
    save,
    use_paper_style,
)

DISPATCH = ROOT / "static/data/representative_day_flexibility"
REPORT = ROOT / "reports/representative_day_flexibility"
IMAGES = ROOT / "paper/images"
REPRESENTATIVE_DAY_FILE = ROOT / "reports/final_annual_results/representative_day.txt"

UPWARD_PANELS = ((24, 60), (-100.0, -200.0))
DOWNWARD_PANELS = ((0, 24), (25.0, 75.0))


def component_deltas(
    current: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Return an additive grid-import decomposition for a flexibility event."""

    return {
        "IT load": current["p_it_total_kw"].to_numpy()
        - reference["p_it_total_kw"].to_numpy(),
        "CRAC chiller": current["p_chiller_hvac_kw"].to_numpy()
        - reference["p_chiller_hvac_kw"].to_numpy(),
        "TES charging chiller": current["p_chiller_tes_kw"].to_numpy()
        - reference["p_chiller_tes_kw"].to_numpy(),
        "UPS net": (
            current["p_ups_charge_kw"].to_numpy()
            - reference["p_ups_charge_kw"].to_numpy()
            - current["p_ups_discharge_kw"].to_numpy()
            + reference["p_ups_discharge_kw"].to_numpy()
        ),
    }


# --------------------------------------------------------------------------
# Figure 6: recovery-certified duration grid
# --------------------------------------------------------------------------


def _duration_grid(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    heat = results.pivot(
        index="magnitude_kw", columns="start_hour", values="duration_hours"
    ).sort_index(ascending=False)
    lower = (
        results.pivot(
            index="magnitude_kw",
            columns="start_hour",
            values="duration_is_lower_bound",
        )
        .reindex(index=heat.index, columns=heat.columns)
        .fillna(False)
        .astype(bool)
    )
    return heat, lower


def _draw_block(
    axis,
    block: pd.DataFrame,
    lower: pd.DataFrame,
    vmax: float,
    label_x: bool,
) -> None:
    values = block.to_numpy(dtype=float)
    mesh = axis.pcolormesh(
        np.arange(values.shape[1] + 1),
        np.arange(values.shape[0] + 1),
        values,
        cmap=DURATION_CMAP,
        vmin=0.0,
        vmax=vmax,
        edgecolors="white",
        linewidth=0.4,
    )
    mesh.set_rasterized(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value <= 0:
                continue
            suffix = "+" if bool(lower.iat[row, column]) else ""
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:g}{suffix}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if value > 0.58 * vmax else "#123A5C",
            )

    axis.set_yticks(np.arange(values.shape[0]) + 0.5)
    axis.set_yticklabels([f"{value:+.0f}" for value in block.index])
    axis.set_xticks(np.arange(values.shape[1]) + 0.5)
    if label_x:
        axis.set_xticklabels([f"{int(value):02d}" for value in block.columns])
        axis.set_xlabel("Event start time (local clock, h)")
    else:
        axis.set_xticklabels([])
    axis.tick_params(length=0)
    axis.grid(False)
    axis.set_ylim(values.shape[0], 0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_duration_grid(results: pd.DataFrame) -> None:
    use_paper_style()
    heat, lower = _duration_grid(results)
    downward = heat[heat.index > 0]
    upward = heat[heat.index < 0]
    vmax = float(np.nanmax(heat.to_numpy()))

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.55), layout="constrained")
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[len(downward), len(upward)],
        width_ratios=[1.0, 0.022],
        hspace=0.10,
        wspace=0.02,
    )
    top = fig.add_subplot(grid[0, 0])
    bottom = fig.add_subplot(grid[1, 0])
    _draw_block(top, downward, lower.loc[downward.index], vmax, label_x=False)
    _draw_block(bottom, upward, lower.loc[upward.index], vmax, label_x=True)

    top.set_title("Downward flexibility: increased import", loc="left", pad=4)
    bottom.set_title("Upward flexibility: reduced import", loc="left", pad=4)
    fig.supylabel("Requested grid-power deviation (kW)", fontsize=8)

    colour_axis = fig.add_subplot(grid[:, 1])
    bar = fig.colorbar(top.collections[0], cax=colour_axis)
    bar.set_label("Certified feasible duration (h)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor(DARK_GREY)
    save(fig, IMAGES / "Figure_6.png")


# --------------------------------------------------------------------------
# Figures 7 and 8: component decomposition
# --------------------------------------------------------------------------


def plot_components(
    detailed: dict[tuple[int, float], pd.DataFrame],
    reference: pd.DataFrame,
    starts: tuple[int, int],
    magnitudes: tuple[float, float],
    filename: str,
) -> None:
    use_paper_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXT_WIDTH, 3.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    panels: dict[tuple[int, int], tuple[float, dict[str, np.ndarray]]] = {}
    longest = 1.0
    low = min(0.0, *magnitudes)
    high = max(0.0, *magnitudes)
    for row, magnitude in enumerate(magnitudes):
        for column, start in enumerate(starts):
            planned = detailed.get((start, magnitude))
            if planned is None:
                continue
            duration = int(planned.attrs["flexibility_duration_steps"])
            stop = start + duration
            components = component_deltas(
                planned.iloc[:duration], reference.iloc[start:stop]
            )
            panels[(row, column)] = (duration / 4.0, components)
            longest = max(longest, duration / 4.0)
            stacked = np.vstack(list(components.values()))
            high = max(high, float(np.clip(stacked, 0, None).sum(axis=0).max()))
            low = min(low, float(np.clip(stacked, None, 0).sum(axis=0).min()))
    pad = 0.07 * (high - low)

    for row, magnitude in enumerate(magnitudes):
        for column, start in enumerate(starts):
            axis = axes[row, column]
            panel = chr(ord("a") + row * len(starts) + column)
            title = (
                f"({panel}) start {start / 4:02.0f}:00, "
                f"$\\Delta P={magnitude:+.0f}$ kW"
            )
            axis.set_title(title, loc="left")
            axis.set_xlim(0, longest)
            axis.set_ylim(low - pad, high + pad)
            if row == len(magnitudes) - 1:
                axis.set_xlabel("Time from event start (h)")
            if column == 0:
                axis.set_ylabel("Component change (kW)")

            if (row, column) not in panels:
                axis.text(
                    0.5, 0.5, "Infeasible", ha="center", va="center",
                    transform=axis.transAxes, color=GREY,
                )
                continue

            hours, components = panels[(row, column)]
            x = np.arange(int(round(hours * 4))) / 4.0
            positive = np.zeros(len(x))
            negative = np.zeros(len(x))
            for label, values in components.items():
                above = np.clip(values, 0, None)
                below = np.clip(values, None, 0)
                axis.bar(
                    x, above, width=0.25, bottom=positive, align="edge",
                    color=COMPONENT_COLOURS[label], label=label, linewidth=0,
                )
                axis.bar(
                    x, below, width=0.25, bottom=negative, align="edge",
                    color=COMPONENT_COLOURS[label], linewidth=0,
                )
                positive += above
                negative += below

            axis.axhline(0, color=DARK_GREY, linewidth=0.5, zorder=3)
            axis.axhline(
                magnitude, color=VERMILION, linestyle=(0, (3.2, 1.4)),
                linewidth=1.0, zorder=4,
                label="Requested $\\Delta P$",
            )
            if hours < longest:
                axis.axvspan(
                    hours, longest, color="#EFF1F3", zorder=0, linewidth=0
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    legend_above(
        axes[0, 0], 5, list(unique.values()), list(unique.keys())
    )
    save(fig, IMAGES / filename)


# --------------------------------------------------------------------------


def _load_detailed(
    date: str, panels: tuple[tuple[int, int], tuple[float, float]]
) -> dict[tuple[int, float], pd.DataFrame]:
    results = pd.read_csv(REPORT / "flexibility_results.csv")
    starts, magnitudes = panels
    detailed: dict[tuple[int, float], pd.DataFrame] = {}
    for start in starts:
        for magnitude in magnitudes:
            sign = "pos" if magnitude >= 0 else "neg"
            path = (
                DISPATCH
                / f"{date}_start_{start:02d}_magnitude_{sign}_{abs(magnitude):.0f}.csv"
            )
            if not path.exists():
                continue
            match = results[
                (results["start_step"] == start)
                & (results["magnitude_kw"] == magnitude)
            ]
            if match.empty or int(match.iloc[0]["duration_steps"]) == 0:
                continue
            planned = pd.read_csv(path)
            planned.attrs["flexibility_duration_steps"] = int(
                match.iloc[0]["duration_steps"]
            )
            planned.attrs["baseline_start_step"] = start
            detailed[(start, magnitude)] = planned
    return detailed


def main() -> None:
    date = REPRESENTATIVE_DAY_FILE.read_text(encoding="utf-8").strip()
    results = pd.read_csv(REPORT / "flexibility_results.csv")
    reference = pd.read_csv(DISPATCH / f"{date}_baseline_planned.csv")

    plot_duration_grid(results)
    plot_components(
        _load_detailed(date, UPWARD_PANELS),
        reference,
        *UPWARD_PANELS,
        "Figure_7.png",
    )
    plot_components(
        _load_detailed(date, DOWNWARD_PANELS),
        reference,
        *DOWNWARD_PANELS,
        "Figure_8.png",
    )
    print(f"Regenerated Figures 6-8 for {date} from cached results")


if __name__ == "__main__":
    main()
