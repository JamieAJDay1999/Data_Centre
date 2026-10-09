"""Generate Figure 2 from the current 96 quarter-hour workload inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "load_profiles.csv"
OUTPUT = HERE / "images" / "Figure_2.png"


def hourly_means(values: list[float]) -> np.ndarray:
    """Average each consecutive group of four quarter-hour values."""

    array = np.asarray(values, dtype=float)
    if array.size != 96:
        raise ValueError(f"Expected 96 quarter-hour values, found {array.size}")
    return array.reshape(24, 4).mean(axis=1)


def generate(output: Path = OUTPUT) -> float:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 96:
        raise ValueError(f"Expected at least 96 workload rows, found {len(rows)}")
    rows = rows[:96]

    inflexible_qh = [100.0 * float(row["inflexible_load"]) for row in rows]
    flexible_qh = [100.0 * float(row["flexible_load"]) for row in rows]
    inflexible = hourly_means(inflexible_qh)
    flexible = hourly_means(flexible_qh)
    total = inflexible + flexible
    average_total = float(np.mean(np.asarray(inflexible_qh) + np.asarray(flexible_qh)))

    # Match the original wide, shallow composition at the printed column width.
    # A compact five-row legend fits in the unused upper-right plot area.
    column_width = 252.0 / 72.27
    fig, ax = plt.subplots(figsize=(column_width, 1.68))
    x = np.arange(24)

    ax.stackplot(
        x,
        inflexible,
        flexible,
        colors=["#ff8c2a", "#3f8fc2"],
        labels=["Inflexible workload", "Flexible workload"],
        linewidth=0,
    )
    ax.plot(x, total, color="#2ca02c", linestyle="--", linewidth=1.15,
            label="Total workload")
    ax.axhline(average_total, color="#800080", linestyle=":", linewidth=1.1,
               label=f"Average total ({average_total:.1f}%)")
    ax.axhline(100.0, color="#7f7f7f", linestyle="--", linewidth=0.9,
               label="Capacity limit (100%)")

    ax.set_xlim(-1.15, 24.15)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Workload ratio (%)", fontsize=6.2, labelpad=2)
    ax.set_xlabel("Time (hours)", fontsize=6.2, labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{hour:02d}-{hour + 1:02d}" for hour in range(24)],
        rotation=48,
        ha="right",
        rotation_mode="anchor",
        fontsize=4.4,
    )
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="y", labelsize=5.2, length=2, width=0.5, pad=1.5)
    ax.tick_params(axis="x", length=1.8, width=0.5, pad=1.0)
    ax.grid(True, which="major", color="#b8b8b8", linestyle=":", linewidth=0.35, alpha=0.65)
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    ax.spines[["top", "right"]].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        fontsize=4.7,
        frameon=True,
        framealpha=0.88,
        borderpad=0.25,
        labelspacing=0.12,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.985, bottom=0.225)
    # Check actual rendered bounds, rather than assuming the legend clears data.
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    legend_bottom = ax.transData.inverted().transform((legend_box.x0, legend_box.y0))[1]
    if legend_bottom <= total.max() + 3:
        raise ValueError(f"Legend overlaps workload area: bottom={legend_bottom:.2f}%")
    print(f"Legend lower edge: {legend_bottom:.2f}%; highest hourly total: {total.max():.2f}%")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)
    return average_total


if __name__ == "__main__":
    value = generate()
    print(f"Wrote {OUTPUT} with average total workload {value:.4f}%")
