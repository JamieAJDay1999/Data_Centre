"""Plot IMRP actuals as rolling annual series aligned to the first timestep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "static" / "data" / "imrp_actuals.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "static" / "images" / "imrp_actuals_by_rolling_year.png"
REQUIRED_COLUMNS = {"IMRP_Date", "Settlement_Period", "IMRP_Amount"}


def load_imrp_data(csv_path: Path) -> pd.DataFrame:
    """Load, validate, and chronologically order the IMRP observations."""
    data = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    data["IMRP_Date"] = pd.to_datetime(data["IMRP_Date"], errors="raise")
    data["IMRP_Amount"] = pd.to_numeric(data["IMRP_Amount"], errors="raise")
    return data.sort_values(["IMRP_Date", "Settlement_Period"], kind="stable").reset_index(drop=True)


def rolling_years(data: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, bool]]:
    """Split observations into one-year windows anchored at the first date."""
    anchor = data["IMRP_Date"].min()
    final_date = data["IMRP_Date"].max()
    annual_series: list[tuple[str, np.ndarray, np.ndarray, bool]] = []

    year_index = 0
    while True:
        start = anchor + pd.DateOffset(years=year_index)
        stop = anchor + pd.DateOffset(years=year_index + 1)
        subset = data.loc[(data["IMRP_Date"] >= start) & (data["IMRP_Date"] < stop)]
        if subset.empty:
            break

        is_complete = final_date >= stop - pd.Timedelta(days=1)
        label = f"{start.year}\N{EN DASH}{str(stop.year)[-2:]}"
        if not is_complete:
            label += " (partial)"

        elapsed_days = np.arange(len(subset), dtype=float) / 24.0
        amounts = subset["IMRP_Amount"].to_numpy(dtype=float)
        annual_series.append((label, elapsed_days, amounts, is_complete))
        year_index += 1

    return annual_series


def plot_rolling_years(data: pd.DataFrame, output_path: Path) -> None:
    """Render all rolling annual windows on a shared elapsed-day axis."""
    annual_series = rolling_years(data)
    anchor = data["IMRP_Date"].min()
    final_date = data["IMRP_Date"].max()

    fig, ax = plt.subplots(figsize=(13, 7.5))
    colours = plt.get_cmap("tab10").colors

    for index, (label, elapsed_days, amounts, is_complete) in enumerate(annual_series):
        if is_complete:
            colour = colours[index % len(colours)]
            linestyle = "-"
            linewidth = 0.8
            alpha = 0.78
        else:
            colour = "0.15"
            linestyle = "--"
            linewidth = 1.4
            alpha = 0.95

        ax.plot(
            elapsed_days,
            amounts,
            label=label,
            color=colour,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlim(0, 366)
    ax.set_xticks([0, 61, 122, 183, 244, 305, 366])
    ax.set_xlabel(f"Elapsed days from annual start ({anchor:%d %B})")
    ax.set_ylabel("IMRP amount")
    ax.set_title(
        "IMRP actuals by rolling year\n"
        f"First timestep: {anchor:%d %B %Y} | Latest timestep: {final_date:%d %B %Y}"
    )
    ax.grid(True, color="0.82", linewidth=0.7)
    ax.axhline(0, color="0.35", linewidth=0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        title="Rolling annual window",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.26)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input IMRP CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_imrp_data(args.input)
    plot_rolling_years(data, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
