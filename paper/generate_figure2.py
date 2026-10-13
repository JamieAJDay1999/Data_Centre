"""Generate Figure 2 from the current 96 quarter-hour workload inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from figure_style import (BLUE, COLUMN_WIDTH, DARK_GREY, ORANGE, VERMILION,
                          legend_above, plt, save, use_paper_style)


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "load_profiles.csv"
OUTPUT = HERE / "images" / "Figure_2.pdf"


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
    # Slots 1-96 are the daily profile; the model does not read later rows.
    rows = rows[:96]

    inflexible_qh = [100.0 * float(row["inflexible_load"]) for row in rows]
    flexible_qh = [100.0 * float(row["flexible_load"]) for row in rows]
    inflexible = hourly_means(inflexible_qh)
    flexible = hourly_means(flexible_qh)
    total = inflexible + flexible
    average_total = float(np.mean(np.asarray(inflexible_qh) + np.asarray(flexible_qh)))

    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.25), layout="constrained")
    # Hourly means are drawn as steps spanning each labelled hour.
    edges = np.arange(25)
    step = lambda values: np.append(values, values[-1])
    ax.fill_between(edges, 0, step(inflexible), step="post", color=ORANGE,
                    label="Inflexible", linewidth=0)
    ax.fill_between(edges, step(inflexible), step(total), step="post", color=BLUE,
                    alpha=0.85, label="Flexible", linewidth=0)
    ax.step(edges, step(total), where="post", color=DARK_GREY, linewidth=0.9,
            label="Total")
    ax.axhline(average_total, color=VERMILION, linestyle=":", linewidth=1.1,
               label=f"Daily mean total ({average_total:.1f}%)")

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{hour:02d}" for hour in range(0, 25, 3)])
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Arriving CPU utilisation (%)")
    ax.set_xlabel("Local time (h)")
    legend_above(ax, ncol=2)
    save(fig, output)
    return average_total


if __name__ == "__main__":
    value = generate()
    print(f"Wrote {OUTPUT} with average total workload {value:.4f}%")
