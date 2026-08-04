"""Shared plotting style for the paper figures.

Every figure is generated at the exact size it occupies in the two-column
IEEEtran layout, so ``\\includegraphics`` never rescales it and each label
keeps the point size chosen here. The previous figures were drawn at 11--20
inches wide and then shrunk to a 3.5 inch column, which reduced 10 pt labels
to roughly 3 pt on the printed page.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import fontManager

# IEEEtran, lettersize journal: \columnwidth = 252 pt, \textwidth = 516 pt,
# measured with a probe compile. TeX points are 1/72.27 inch.
TEX_PT = 72.27
COLUMN_WIDTH = 252.0 / TEX_PT  # 3.487 in
TEXT_WIDTH = 516.0 / TEX_PT  # 7.140 in
HALF_TEXT_WIDTH = 0.49 * TEXT_WIDTH  # side-by-side subfloats

# Okabe-Ito, which stays distinguishable in greyscale and for the common
# colour-vision deficiencies.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
PINK = "#CC79A7"
VERMILION = "#D55E00"
GREY = "#8C8C8C"
DARK_GREY = "#3F3F3F"
LIGHT_GREY = "#C9CDD2"

COMPONENT_COLOURS = {
    "IT load": BLUE,
    "CRAC chiller": ORANGE,
    "TES charging chiller": GREEN,
    "UPS net": PINK,
}

# Light-to-dark sequential ramp. Zero-duration cells stay near white so the
# feasible region of the flexibility grid reads at a glance.
DURATION_CMAP = LinearSegmentedColormap.from_list(
    "duration",
    ["#F7FBFC", "#CDE7EA", "#8FC9CD", "#4FA3AE", "#24738F", "#123A5C"],
)


def _serif_family() -> list[str]:
    installed = {font.name for font in fontManager.ttflist}
    preferred = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    return [name for name in preferred if name in installed] or ["serif"]


def use_paper_style() -> None:
    """Apply the shared rcParams. Safe to call repeatedly."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _serif_family(),
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "normal",
            "axes.titlepad": 3.5,
            "axes.labelsize": 8,
            "axes.labelpad": 2.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": DARK_GREY,
            "axes.labelcolor": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#DFE3E8",
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": DARK_GREY,
            "ytick.color": DARK_GREY,
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.borderaxespad": 0.0,
            "patch.linewidth": 0.0,
            "savefig.dpi": 400,
            "figure.dpi": 400,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def legend_above(axis, ncol: int, handles=None, labels=None):
    """Place a borderless legend in a reserved strip above the axes.

    ``loc="outside upper center"`` needs a constrained layout, which every
    figure here uses; it reserves the strip instead of drawing the legend on
    top of the data.
    """

    if handles is None:
        handles, labels = axis.get_legend_handles_labels()
    return axis.figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=ncol,
    )


def hour_axis(axis, hours: float = 24.0, step: float = 4.0) -> None:
    """Label a local-clock axis compactly."""

    ticks = [value for value in _frange(0.0, hours, step)]
    axis.set_xlim(0.0, hours)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{int(value):02d}" for value in ticks])


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-9:
        yield value
        value += step


def save(fig, path, **kwargs) -> None:
    """Write a figure without altering its configured physical size.

    ``bbox_inches="tight"`` is deliberately not used: it changes the output
    width, which would reintroduce silent rescaling in LaTeX.
    """

    fig.savefig(path, **kwargs)
    plt.close(fig)
