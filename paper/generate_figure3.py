"""Generate Figure 3: equal computational demand R at different utilisations."""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import Rectangle

from figure_style import (BLUE, COLUMN_WIDTH, DARK_GREY, GREEN, PINK, plt,
                          save, use_paper_style)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "images" / "Figure_3.pdf"

# (utilisation, duration multiple of the reference interval, label, colour)
CASES = [
    (1.0, 1, r"$u=U$, $\Delta T$", BLUE),
    (0.5, 2, r"$u=U/2$, $2\Delta T$", GREEN),
    (0.25, 4, r"$u=U/4$, $4\Delta T$", PINK),
]


def generate(output: Path = OUTPUT) -> None:
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.55), sharey=True,
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    for axis, (utilisation, duration, label, colour) in zip(axes, CASES):
        axis.add_patch(Rectangle((0, 0), duration, utilisation, facecolor=colour,
                                 alpha=0.35, edgecolor=colour, linewidth=1.0))
        axis.text(duration / 2, utilisation / 2, r"$R$", ha="center", va="center",
                  fontsize=9, color=DARK_GREY)
        axis.set_xlim(0, 4.2)
        axis.set_ylim(0, 1.12)
        # Mark only the start and end of each execution period.
        axis.set_xticks([0, duration])
        axis.set_xticklabels(["0", label.split(", ")[1]])
        axis.set_title(label)
        axis.grid(False)
    axes[0].set_yticks([0, 0.25, 0.5, 1.0])
    axes[0].set_yticklabels(["0", r"$U/4$", r"$U/2$", r"$U$"])
    axes[0].set_ylabel("CPU utilisation")
    fig.supxlabel("Execution time", fontsize=8)
    save(fig, output)


if __name__ == "__main__":
    generate()
    print(f"Wrote {OUTPUT}")
