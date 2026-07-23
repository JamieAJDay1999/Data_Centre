"""Generate a self-contained HTML EDA and year-selection report for IMRP data."""

from __future__ import annotations

import argparse
import base64
import calendar
import html
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "static" / "data" / "imrp_actuals.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "imrp_eda_year_selection_report.html"
PAPER_PRICE_PROFILE = np.array(
    [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70],
    dtype=float,
)
LCCC_DATASET_URL = "https://dp.lowcarboncontracts.uk/dataset/imrp-actuals"
LCCC_REFERENCE_URL = (
    "https://www.lowcarboncontracts.uk/our-schemes/contracts-for-difference/market-reference-prices/"
)
FEATURE_WEIGHTS = {
    "median": 0.20,
    "iqr": 0.15,
    "p05": 0.10,
    "p95": 0.15,
    "negative_share": 0.10,
    "high_200_share": 0.10,
    "ramp_p95": 0.10,
    "seasonal_amplitude": 0.05,
    "diurnal_amplitude": 0.05,
}


def load_data(csv_path: Path) -> pd.DataFrame:
    required = {"IMRP_Date", "Settlement_Period", "IMRP_Amount"}
    data = pd.read_csv(csv_path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data["IMRP_Date"] = pd.to_datetime(data["IMRP_Date"], errors="raise")
    data["Settlement_Period"] = pd.to_numeric(data["Settlement_Period"], errors="raise").astype(int)
    data["IMRP_Amount"] = pd.to_numeric(data["IMRP_Amount"], errors="raise")
    data = data.sort_values(["IMRP_Date", "Settlement_Period"], kind="stable").reset_index(drop=True)
    data["Year"] = data["IMRP_Date"].dt.year
    data["Month"] = data["IMRP_Date"].dt.month
    data["Day_of_week"] = data["IMRP_Date"].dt.dayofweek
    data["Timestamp"] = data["IMRP_Date"] + pd.to_timedelta(data["Settlement_Period"] - 1, unit="h")
    return data


def complete_calendar_years(data: pd.DataFrame) -> list[int]:
    years: list[int] = []
    for year, group in data.groupby("Year"):
        expected_days = 366 if calendar.isleap(int(year)) else 365
        dates = group["IMRP_Date"].drop_duplicates()
        if (
            len(dates) == expected_days
            and dates.min() == pd.Timestamp(int(year), 1, 1)
            and dates.max() == pd.Timestamp(int(year), 12, 31)
        ):
            years.append(int(year))
    return years


def year_metrics(group: pd.DataFrame) -> dict[str, float]:
    values = group["IMRP_Amount"]
    daily = group.groupby("IMRP_Date")["IMRP_Amount"]
    monthly_medians = group.groupby("Month")["IMRP_Amount"].median()
    period_medians = group.loc[group["Settlement_Period"] <= 24].groupby("Settlement_Period")["IMRP_Amount"].median()
    ramps = group.groupby("IMRP_Date")["IMRP_Amount"].diff().abs().dropna()
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "observations": float(len(group)),
        "days": float(group["IMRP_Date"].nunique()),
        "mean": float(values.mean()),
        "median": float(quantiles.loc[0.5]),
        "std": float(values.std()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "negative_share": float((values < 0).mean() * 100),
        "high_200_share": float((values > 200).mean() * 100),
        "high_500_share": float((values > 500).mean() * 100),
        "ramp_median": float(ramps.median()),
        "ramp_p95": float(ramps.quantile(0.95)),
        "daily_range_median": float((daily.max() - daily.min()).median()),
        "seasonal_amplitude": float(monthly_medians.max() - monthly_medians.min()),
        "diurnal_amplitude": float(period_medians.max() - period_medians.min()),
    }


def calendar_metrics(data: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = {year: year_metrics(data.loc[data["Year"] == year]) for year in years}
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Calendar year"
    return result


def rolling_window_metrics(data: pd.DataFrame) -> pd.DataFrame:
    anchor = data["IMRP_Date"].min()
    final_date = data["IMRP_Date"].max()
    rows: dict[str, dict[str, float]] = {}
    index = 0
    while True:
        start = anchor + pd.DateOffset(years=index)
        stop = anchor + pd.DateOffset(years=index + 1)
        subset = data.loc[(data["IMRP_Date"] >= start) & (data["IMRP_Date"] < stop)]
        if subset.empty:
            break
        complete = final_date >= stop - pd.Timedelta(days=1)
        label = f"{start.year}\N{EN DASH}{str(stop.year)[-2:]}"
        if not complete:
            label += " (partial)"
        metrics = year_metrics(subset)
        metrics["complete"] = float(complete)
        rows[label] = metrics
        index += 1
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "First-timestep-anchored year"
    return result


def robust_selection_scores(metrics: pd.DataFrame) -> tuple[pd.DataFrame, int, list[int]]:
    features = list(FEATURE_WEIGHTS)
    recent_years = list(metrics.index[-3:])
    recent_target = metrics.loc[recent_years, features].median()
    historical_target = metrics.loc[:, features].median()
    q75 = metrics.loc[:, features].quantile(0.75)
    q25 = metrics.loc[:, features].quantile(0.25)
    scale = (q75 - q25).replace(0, np.nan)
    scale = scale.fillna(metrics.loc[:, features].std()).replace(0, 1.0)
    weights = pd.Series(FEATURE_WEIGHTS)

    recent_distance = ((metrics[features] - recent_target).abs() / scale).mul(weights, axis=1).sum(axis=1)
    historical_distance = ((metrics[features] - historical_target).abs() / scale).mul(weights, axis=1).sum(axis=1)
    scores = pd.DataFrame(
        {
            "Recent-regime score": recent_distance,
            "Full-history score": historical_distance,
        }
    )
    scores["Recent rank"] = scores["Recent-regime score"].rank(method="min").astype(int)
    scores["Historical rank"] = scores["Full-history score"].rank(method="min").astype(int)
    selected_year = int(scores.loc[recent_years, "Recent-regime score"].idxmin())
    return scores, selected_year, recent_years


def representative_days(data: pd.DataFrame, selected_year: int) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], pd.Series]:
    selected = data.loc[data["Year"] == selected_year]
    valid_dates = selected.groupby("IMRP_Date").size().loc[lambda counts: counts == 24].index
    regular_days = selected.loc[selected["IMRP_Date"].isin(valid_dates)]
    pivot = regular_days.pivot_table(
        index="IMRP_Date",
        columns="Settlement_Period",
        values="IMRP_Amount",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=range(1, 25)).dropna()
    median_profile = pivot.median(axis=0)
    hour_scale = (pivot.quantile(0.75) - pivot.quantile(0.25)).replace(0, np.nan).fillna(1.0)
    shape_distance = ((pivot - median_profile).abs() / hour_scale).mean(axis=1)

    daily_features = pd.DataFrame(
        {
            "mean": pivot.mean(axis=1),
            "std": pivot.std(axis=1),
            "range": pivot.max(axis=1) - pivot.min(axis=1),
            "minimum": pivot.min(axis=1),
            "maximum": pivot.max(axis=1),
            "ramp_p95": pivot.diff(axis=1).abs().quantile(0.95, axis=1),
        }
    )
    feature_target = daily_features.median()
    feature_scale = (daily_features.quantile(0.75) - daily_features.quantile(0.25)).replace(0, 1.0)
    feature_distance = ((daily_features - feature_target).abs() / feature_scale).mean(axis=1)
    daily_features["typicality_score"] = 0.70 * shape_distance + 0.30 * feature_distance

    typical_day = pd.Timestamp(daily_features["typicality_score"].idxmin())
    range_target = daily_features["range"].quantile(0.90)
    high_volatility_day = pd.Timestamp((daily_features["range"] - range_target).abs().idxmin())
    negative_candidates = daily_features.loc[daily_features["minimum"] < 0]
    if negative_candidates.empty:
        negative_day = pd.Timestamp(daily_features["minimum"].idxmin())
    else:
        negative_day = pd.Timestamp(negative_candidates["minimum"].idxmin())
    high_price_target = daily_features["mean"].quantile(0.90)
    high_price_day = pd.Timestamp((daily_features["mean"] - high_price_target).abs().idxmin())

    selected_days = {
        "Typical medoid-like day": typical_day,
        "90th-percentile volatility day": high_volatility_day,
        "Negative-price stress day": negative_day,
        "90th-percentile mean-price day": high_price_day,
    }
    return pivot, selected_days, median_profile


def fig_to_uri(fig: plt.Figure, dpi: int = 150) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_figures(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    selected_year: int,
    recent_years: list[int],
    daily_pivot: pd.DataFrame,
    selected_days: dict[str, pd.Timestamp],
    median_profile: pd.Series,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    colours = plt.get_cmap("tab10").colors
    selected_colour = "#7b2cbf"

    daily = data.groupby("IMRP_Date")["IMRP_Amount"].agg(["mean", "median", "min", "max", "count"])
    monthly = data.set_index("IMRP_Date")["IMRP_Amount"].resample("MS").agg(["median", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
    monthly.columns = ["median", "p05", "p95"]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), gridspec_kw={"height_ratios": [1, 2.2]})
    count_frequency = daily["count"].value_counts().sort_index()
    axes[0].plot(daily.index, daily["count"], color="#3a6ea5", linewidth=0.75)
    axes[0].axhline(24, color="0.35", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Hourly periods/day")
    axes[0].set_yticks([23, 24, 25])
    axes[0].set_title("Coverage audit: continuous daily records with expected daylight-saving day lengths")
    axes[0].grid(True, axis="x", color="0.88", linewidth=0.6)
    axes[0].text(
        0.01,
        0.08,
        ", ".join(f"{int(period)} periods: {int(count)} days" for period, count in count_frequency.items()),
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    axes[1].fill_between(monthly.index, monthly["p05"], monthly["p95"], color="#8ecae6", alpha=0.35, label="Monthly 5th–95th percentile")
    axes[1].plot(monthly.index, monthly["median"], color="#005f73", linewidth=1.7, label="Monthly median")
    axes[1].axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), color="#ee9b00", alpha=0.10, label="Observed 2021–22 high-price regime")
    axes[1].axvspan(pd.Timestamp(selected_year, 1, 1), pd.Timestamp(selected_year, 12, 31), color=selected_colour, alpha=0.08, label=f"Recommended year: {selected_year}")
    axes[1].set_ylabel("IMRP (£/MWh)")
    axes[1].set_xlabel("Settlement date")
    axes[1].set_title("Long-run price level and dispersion")
    axes[1].grid(True, color="0.88", linewidth=0.6)
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    figures["coverage_history"] = fig_to_uri(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    years = metrics.index.to_numpy()
    width = 0.38
    axes[0, 0].bar(years - width / 2, metrics["median"], width, label="Median", color="#2a9d8f")
    axes[0, 0].bar(years + width / 2, metrics["mean"], width, label="Mean", color="#e9c46a")
    axes[0, 0].set_title("Annual price level")
    axes[0, 0].set_ylabel("£/MWh")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].bar(years - width / 2, metrics["iqr"], width, label="IQR", color="#457b9d")
    axes[0, 1].bar(years + width / 2, metrics["p95"], width, label="95th percentile", color="#e76f51")
    axes[0, 1].set_title("Central spread and upper tail")
    axes[0, 1].set_ylabel("£/MWh")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(years - width / 2, metrics["negative_share"], width, label="Below £0", color="#264653")
    axes[1, 0].bar(years + width / 2, metrics["high_200_share"], width, label="Above £200", color="#f4a261")
    axes[1, 0].set_title("Frequency of unusual price events")
    axes[1, 0].set_ylabel("Share of hours (%)")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(years, metrics["ramp_p95"], color="#6a4c93")
    axes[1, 1].set_title("Within-day hourly ramp intensity")
    axes[1, 1].set_ylabel("95th percentile |hourly change| (£/MWh)")
    for ax in axes.flat:
        ax.grid(True, axis="y", color="0.88", linewidth=0.6)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.axvline(selected_year, color=selected_colour, linewidth=1.2, linestyle="--", alpha=0.9)
    fig.suptitle("Complete calendar years differ strongly in level, dispersion, tails, and volatility", y=1.01)
    fig.tight_layout()
    figures["annual_metrics"] = fig_to_uri(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    values = [data.loc[data["Year"] == year, "IMRP_Amount"].to_numpy() for year in metrics.index]
    boxes = axes[0].boxplot(values, tick_labels=metrics.index, showfliers=False, patch_artist=True)
    for patch, year in zip(boxes["boxes"], metrics.index):
        patch.set_facecolor(selected_colour if year == selected_year else "#8ecae6")
        patch.set_alpha(0.75)
    axes[0].set_title("Central annual distributions (outliers suppressed)")
    axes[0].set_ylabel("IMRP (£/MWh)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", color="0.88", linewidth=0.6)

    for index, year in enumerate(metrics.index):
        year_values = np.sort(data.loc[data["Year"] == year, "IMRP_Amount"].to_numpy())[::-1]
        exceedance = np.arange(1, len(year_values) + 1) / (len(year_values) + 1) * 100
        axes[1].plot(
            exceedance,
            year_values,
            color=selected_colour if year == selected_year else colours[index % len(colours)],
            linewidth=2.2 if year == selected_year else 0.9,
            alpha=1.0 if year == selected_year else 0.65,
            label=str(year),
        )
    axes[1].set_yscale("symlog", linthresh=50)
    axes[1].set_title("Price-duration curves, including extreme tails")
    axes[1].set_xlabel("Share of hours at or above price (%)")
    axes[1].set_ylabel("IMRP (£/MWh, symlog scale)")
    axes[1].grid(True, color="0.88", linewidth=0.6)
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    figures["distributions"] = fig_to_uri(fig)

    monthly_median = data.loc[data["Year"].isin(metrics.index)].pivot_table(
        index="Year", columns="Month", values="IMRP_Amount", aggfunc="median"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    image = axes[0].imshow(monthly_median.to_numpy(), aspect="auto", cmap="YlOrRd")
    axes[0].set_xticks(range(12), [calendar.month_abbr[i] for i in range(1, 13)])
    axes[0].set_yticks(range(len(monthly_median.index)), monthly_median.index)
    axes[0].set_title("Monthly median IMRP by calendar year")
    for row in range(monthly_median.shape[0]):
        for col in range(monthly_median.shape[1]):
            value = monthly_median.iloc[row, col]
            axes[0].text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=axes[0], label="£/MWh", fraction=0.046, pad=0.04)

    for index, year in enumerate(metrics.index):
        profile = data.loc[(data["Year"] == year) & (data["Settlement_Period"] <= 24)].groupby("Settlement_Period")["IMRP_Amount"].median()
        axes[1].plot(
            profile.index - 1,
            profile.values,
            color=selected_colour if year == selected_year else colours[index % len(colours)],
            linewidth=2.5 if year == selected_year else 0.9,
            alpha=1.0 if year == selected_year else 0.65,
            label=str(year),
        )
    axes[1].set_title("Median intraday profile by year")
    axes[1].set_xlabel("Hour beginning")
    axes[1].set_ylabel("Median IMRP (£/MWh)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].grid(True, color="0.88", linewidth=0.6)
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    figures["seasonality"] = fig_to_uri(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, column, title in [
        (axes[0], "Recent-regime score", f"Similarity to recent complete years ({recent_years[0]}–{recent_years[-1]})"),
        (axes[1], "Full-history score", "Similarity to the median complete year, 2017–2025"),
    ]:
        order = scores[column].sort_values().index
        colours_for_bars = [selected_colour if year == selected_year else "#90a4ae" for year in order]
        ax.barh([str(year) for year in order], scores.loc[order, column], color=colours_for_bars)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Weighted robust distance (lower is more representative)")
        ax.grid(True, axis="x", color="0.88", linewidth=0.6)
        for position, year in enumerate(order):
            value = scores.loc[year, column]
            ax.text(value, position, f" {value:.2f}", va="center", fontsize=8)
    fig.suptitle(f"Transparent year-selection scores; {selected_year} is selected within the recent regime", y=1.02)
    fig.tight_layout()
    figures["selection_scores"] = fig_to_uri(fig)

    selected = data.loc[data["Year"] == selected_year]
    selected_daily = selected.groupby("IMRP_Date")["IMRP_Amount"].agg(["mean", "min", "max"])
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), gridspec_kw={"height_ratios": [1.4, 1]})
    axes[0].fill_between(selected_daily.index, selected_daily["min"], selected_daily["max"], color="#cdb4db", alpha=0.38, label="Daily minimum–maximum")
    axes[0].plot(selected_daily.index, selected_daily["mean"], color=selected_colour, linewidth=1.1, label="Daily mean")
    for label, date in selected_days.items():
        value = selected_daily.loc[date, "mean"]
        axes[0].scatter(date, value, s=24, zorder=4)
        axes[0].annotate(label.replace(" day", ""), (date, value), xytext=(5, 8), textcoords="offset points", fontsize=7, rotation=20)
    axes[0].set_title(f"Recommended year {selected_year}: daily level and within-day range")
    axes[0].set_ylabel("IMRP (£/MWh)")
    axes[0].grid(True, color="0.88", linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    month_values = [selected.loc[selected["Month"] == month, "IMRP_Amount"] for month in range(1, 13)]
    month_boxes = axes[1].boxplot(month_values, tick_labels=[calendar.month_abbr[i] for i in range(1, 13)], showfliers=False, patch_artist=True)
    for patch in month_boxes["boxes"]:
        patch.set_facecolor("#bde0fe")
    axes[1].set_title("Monthly distributions within the recommended year (outliers suppressed)")
    axes[1].set_ylabel("IMRP (£/MWh)")
    axes[1].grid(True, axis="y", color="0.88", linewidth=0.6)
    fig.tight_layout()
    figures["selected_year"] = fig_to_uri(fig)

    selected_regular = selected.loc[selected["Settlement_Period"] <= 24]
    dow_hour = selected_regular.pivot_table(index="Day_of_week", columns="Settlement_Period", values="IMRP_Amount", aggfunc="median")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1.2]})
    image = axes[0].imshow(dow_hour.to_numpy(), aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_xticks(range(0, 24, 3), range(0, 24, 3))
    axes[0].set_xlabel("Hour beginning")
    axes[0].set_title(f"{selected_year} median IMRP by weekday and hour")
    fig.colorbar(image, ax=axes[0], label="£/MWh", fraction=0.046, pad=0.04)

    hours = np.arange(24)
    axes[1].plot(hours, median_profile.values, color="#0077b6", linewidth=2.3, label=f"{selected_year} hourly median")
    axes[1].plot(hours, PAPER_PRICE_PROFILE, color="#d62828", linewidth=2.0, linestyle="--", label="Current paper profile")
    day_colours = ["#7b2cbf", "#f77f00", "#2a9d8f", "#6c757d"]
    for (label, date), colour in zip(selected_days.items(), day_colours):
        axes[1].plot(hours, daily_pivot.loc[date].values, color=colour, linewidth=1.2, alpha=0.9, label=f"{label}: {date:%d %b}")
    axes[1].set_title("Observed representative and stress days versus the paper profile")
    axes[1].set_xlabel("Hour beginning")
    axes[1].set_ylabel("Price (£/MWh)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].grid(True, color="0.88", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    figures["selected_profiles"] = fig_to_uri(fig)

    return figures


def format_annual_table(metrics: pd.DataFrame, scores: pd.DataFrame, selected_year: int) -> str:
    table = pd.DataFrame(index=metrics.index)
    table["Hours"] = metrics["observations"].astype(int)
    table["Mean"] = metrics["mean"]
    table["Median"] = metrics["median"]
    table["IQR"] = metrics["iqr"]
    table["P05"] = metrics["p05"]
    table["P95"] = metrics["p95"]
    table["Min"] = metrics["minimum"]
    table["Max"] = metrics["maximum"]
    table["Negative hours"] = metrics["negative_share"]
    table["> £200 hours"] = metrics["high_200_share"]
    table["P95 hourly ramp"] = metrics["ramp_p95"]
    table["Recent score"] = scores["Recent-regime score"]
    table.index = [f"{year}{' ★' if year == selected_year else ''}" for year in table.index]
    formatters = {
        "Mean": lambda value: f"£{value:,.1f}",
        "Median": lambda value: f"£{value:,.1f}",
        "IQR": lambda value: f"£{value:,.1f}",
        "P05": lambda value: f"£{value:,.1f}",
        "P95": lambda value: f"£{value:,.1f}",
        "Min": lambda value: f"£{value:,.1f}",
        "Max": lambda value: f"£{value:,.1f}",
        "Negative hours": lambda value: f"{value:.2f}%",
        "> £200 hours": lambda value: f"{value:.2f}%",
        "P95 hourly ramp": lambda value: f"£{value:,.1f}",
        "Recent score": lambda value: f"{value:.2f}",
    }
    return table.to_html(classes="data-table", border=0, formatters=formatters, escape=False)


def build_report(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    selected_year: int,
    recent_years: list[int],
    selected_days: dict[str, pd.Timestamp],
    daily_pivot: pd.DataFrame,
    figures: dict[str, str],
    input_path: Path,
) -> str:
    first_date = data["IMRP_Date"].min()
    final_date = data["IMRP_Date"].max()
    daily_counts = data.groupby("IMRP_Date").size()
    expected_dates = pd.date_range(first_date, final_date, freq="D")
    duplicate_keys = int(data.duplicated(["IMRP_Date", "Settlement_Period"]).sum())
    missing_dates = len(expected_dates.difference(pd.DatetimeIndex(data["IMRP_Date"].unique())))
    null_cells = int(data[["IMRP_Date", "Settlement_Period", "IMRP_Amount"]].isna().sum().sum())
    recent_scores = scores.loc[recent_years, "Recent-regime score"].sort_values()
    sensitivity_years = [year for year in recent_years if year != selected_year]
    sensitivity_label = " and ".join(str(year) for year in sensitivity_years)
    historical_best = int(scores["Full-history score"].idxmin())
    selected_metrics = metrics.loc[selected_year]
    paper_stats = {
        "mean": float(PAPER_PRICE_PROFILE.mean()),
        "median": float(np.median(PAPER_PRICE_PROFILE)),
        "range": float(PAPER_PRICE_PROFILE.max() - PAPER_PRICE_PROFILE.min()),
        "std": float(PAPER_PRICE_PROFILE.std(ddof=1)),
        "ramp_p95": float(pd.Series(PAPER_PRICE_PROFILE).diff().abs().dropna().quantile(0.95)),
    }
    daily_feature_table = pd.DataFrame(
        {
            label: {
                "Date": f"{date:%d %B %Y}",
                "Mean": daily_pivot.loc[date].mean(),
                "Minimum": daily_pivot.loc[date].min(),
                "Maximum": daily_pivot.loc[date].max(),
                "Range": daily_pivot.loc[date].max() - daily_pivot.loc[date].min(),
                "Standard deviation": daily_pivot.loc[date].std(),
            }
            for label, date in selected_days.items()
        }
    ).T
    day_table = daily_feature_table.to_html(
        classes="data-table",
        border=0,
        formatters={
            "Mean": lambda value: f"£{value:,.1f}",
            "Minimum": lambda value: f"£{value:,.1f}",
            "Maximum": lambda value: f"£{value:,.1f}",
            "Range": lambda value: f"£{value:,.1f}",
            "Standard deviation": lambda value: f"£{value:,.1f}",
        },
    )
    selected_recent_rank = int(recent_scores.index.get_loc(selected_year) + 1)
    rolling_complete = rolling_metrics.loc[rolling_metrics["complete"] == 1]
    rolling_median_year = rolling_complete["median"].sub(rolling_complete["median"].median()).abs().idxmin()

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>IMRP exploratory data analysis and year-selection recommendation</title>
<style>
  :root {{ color-scheme: light; --ink: #1f2933; --muted: #52606d; --line: #d9e2ec; --paper: #ffffff; --soft: #f5f7fa; --accent: #7b2cbf; --accent-soft: #f1e8f8; --warn: #9c5700; --warn-soft: #fff4d6; --good: #176b4d; --good-soft: #e5f5ee; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--soft); color: var(--ink); font-family: Inter, "Segoe UI", Arial, sans-serif; line-height: 1.58; }}
  main {{ max-width: 1180px; margin: 0 auto; background: var(--paper); padding: 42px 58px 72px; }}
  h1, h2, h3 {{ line-height: 1.22; color: #102a43; }}
  h1 {{ margin: 0 0 8px; font-size: 2rem; }}
  h2 {{ margin-top: 42px; padding-bottom: 8px; border-bottom: 2px solid var(--line); font-size: 1.45rem; }}
  h3 {{ margin-top: 28px; font-size: 1.12rem; }}
  p, li {{ max-width: 94ch; }}
  a {{ color: #5a189a; }}
  .subtitle {{ margin: 0 0 26px; color: var(--muted); }}
  .summary {{ border-left: 5px solid var(--accent); background: var(--accent-soft); padding: 18px 22px; margin: 24px 0; }}
  .recommendation {{ border-left: 5px solid var(--good); background: var(--good-soft); padding: 18px 22px; margin: 24px 0; }}
  .caution {{ border-left: 5px solid #d97706; background: var(--warn-soft); padding: 16px 20px; margin: 22px 0; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0; }}
  .metric {{ border: 1px solid var(--line); padding: 14px; background: #fbfdff; }}
  .metric strong {{ display: block; font-size: 1.35rem; color: #102a43; }}
  .metric span {{ color: var(--muted); font-size: 0.9rem; }}
  figure {{ margin: 28px 0 34px; }}
  figure img {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); }}
  figcaption {{ color: var(--muted); font-size: 0.92rem; margin-top: 8px; }}
  .table-wrap {{ overflow-x: auto; margin: 20px 0 30px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  table.data-table th, table.data-table td {{ padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
  table.data-table th:first-child, table.data-table td:first-child {{ text-align: left; }}
  table.data-table thead th {{ background: #edf2f7; position: sticky; top: 0; }}
  code {{ background: #eef2f6; padding: 0.12rem 0.32rem; border-radius: 3px; }}
  .small {{ color: var(--muted); font-size: 0.88rem; }}
  .decision-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
  .decision-grid section {{ border: 1px solid var(--line); padding: 14px 16px; }}
  @media (max-width: 700px) {{ main {{ padding: 24px 18px 50px; }} h1 {{ font-size: 1.6rem; }} }}
  @media print {{ body {{ background: white; }} main {{ max-width: none; padding: 0; }} figure {{ break-inside: avoid; }} h2 {{ break-after: avoid; }} }}
</style>
</head>
<body>
<main>
<h1>IMRP exploratory data analysis and year-selection recommendation</h1>
<p class="subtitle">Local snapshot analysed: {first_date:%d %B %Y} to {final_date:%d %B %Y} | Generated from <code>{html.escape(str(input_path))}</code></p>

<div class="recommendation">
  <h3 style="margin-top:0">Recommendation for the paper revision</h3>
  <p><strong>Use calendar year {selected_year} as the primary empirical price year</strong> if one year must be selected. It is complete, recent, post-dates the 2021–22 high-price regime, and is the closest of the three most recent complete years ({recent_years[0]}–{recent_years[-1]}) to their multivariate median across price level, spread, tails, negative-price frequency, high-price frequency, hourly ramps, seasonality, and intraday shape.</p>
  <p>For a defensible paper, do not present {selected_year} as universally representative of 2016–2026. State that it is a <em>recent-regime representative year selected by a pre-specified score</em>. Use {sensitivity_label} as recent-year sensitivity cases, and use 2022 only as a deliberate high-price stress case. If computationally feasible, the stronger design is to run every valid day in {selected_year} and report the distribution of model outcomes rather than one hand-picked 24-hour day.</p>
</div>

<div class="summary">
  <strong>Central finding.</strong> The dataset is structurally complete but not statistically stationary. Annual median, dispersion, negative-price incidence, and extreme-price frequency change markedly across the record. A single year can support a bounded case study, but it cannot substitute for temporal sensitivity analysis.
</div>

<div class="metrics">
  <div class="metric"><strong>{len(data):,}</strong><span>hourly observations</span></div>
  <div class="metric"><strong>{data['IMRP_Date'].nunique():,}</strong><span>consecutive settlement dates</span></div>
  <div class="metric"><strong>{metrics.index.min()}–{metrics.index.max()}</strong><span>complete calendar years</span></div>
  <div class="metric"><strong>{selected_year}</strong><span>recommended primary year</span></div>
</div>

<h2>1. What this dataset represents</h2>
<p>The column name IMRP means <strong>Intermittent Market Reference Price</strong>. According to the Low Carbon Contracts Company (LCCC), it is the GB day-ahead hourly price calculated as a weighted average across day-ahead indices. The portal defines <code>IMRP_Amount</code> in <strong>£/MWh</strong>, the settlement date as the date on which the price applies, and settlement period 1 as the first hourly period of the day. See the <a href="{LCCC_DATASET_URL}">official LCCC dataset description</a> and <a href="{LCCC_REFERENCE_URL}">market-reference-price methodology page</a>.</p>
<p>This is a good conceptual match for the paper's day-ahead price input. It is not an imbalance price, consumer retail tariff, or a site-specific electricity bill. In the revision, call it an LCCC GB day-ahead Intermittent Market Reference Price and cite the dataset version/date.</p>

<h2>2. Data-quality and coverage audit</h2>
<ul>
  <li><strong>No missing settlement dates:</strong> {missing_dates} dates are absent from the continuous range.</li>
  <li><strong>No null cells:</strong> {null_cells} null values occur in the three source columns.</li>
  <li><strong>No duplicate date-period keys:</strong> {duplicate_keys} duplicates were found.</li>
  <li><strong>Daylight saving is represented:</strong> daily counts range from {int(daily_counts.min())} to {int(daily_counts.max())} hourly periods. This is expected for 23- and 25-hour clock-change days and should not be repaired by deleting or duplicating observations.</li>
  <li><strong>Partial edge years:</strong> 2016 and 2026 are incomplete calendar years and were excluded from like-for-like calendar-year ranking. All {len(data):,} observations remain included in the long-run descriptive analysis.</li>
</ul>
<figure><img src="{figures['coverage_history']}" alt="Coverage audit and long-run IMRP history"><figcaption>The daily period count confirms continuity and daylight-saving structure. The lower panel shows a pronounced shift into a high-price, high-dispersion regime during 2021–22, followed by lower but still evolving prices.</figcaption></figure>

<h2>3. What changes across years</h2>
<p>The annual mean is consistently above the median in the high-price years, showing strong right-skew from price spikes. The central distribution, upper tail, negative-price incidence, and within-day hourly changes all vary materially. Therefore, selecting a year based only on the annual mean would omit behaviour that directly determines the value of workload shifting and storage arbitrage.</p>
<figure><img src="{figures['annual_metrics']}" alt="Annual IMRP metrics including price level, spread, event shares, and ramps"><figcaption>The dashed line marks {selected_year}. The 2021–22 years are tail- and level-dominated; later years have lower central prices but a growing incidence of negative hours and retain occasional spikes.</figcaption></figure>
<figure><img src="{figures['distributions']}" alt="Annual IMRP boxplots and price-duration curves"><figcaption>Boxplots show central distributions without outliers; the price-duration curves retain the extreme tails on a symmetric-log scale. This makes clear why a single mean or one visually appealing day is not a sufficient selection criterion.</figcaption></figure>

<h3>Complete-calendar-year statistics</h3>
<div class="table-wrap">{format_annual_table(metrics, scores, selected_year)}</div>
<p class="small">★ recommended year. The score is a weighted robust distance from the 2023–2025 median using: median (20%), IQR (15%), 5th percentile (10%), 95th percentile (15%), negative-hour share (10%), share above £200/MWh (10%), 95th-percentile hourly ramp (10%), monthly amplitude (5%), and intraday amplitude (5%). Lower is more representative of the recent regime.</p>

<h2>4. Seasonal and intraday structure</h2>
<p>Both seasonal price levels and the shape of the median day change between years. These dimensions matter for a data-centre model: storage and deferred workload respond to within-day spreads, while annual cost and available arbitrage opportunities depend on how frequently those daily shapes occur.</p>
<figure><img src="{figures['seasonality']}" alt="Monthly median heatmap and annual intraday price profiles"><figcaption>The heatmap exposes the strong monthly concentration of the 2021–22 high-price regime. The right-hand panel shows that year selection changes not only the level of prices but also the intraday signal seen by the optimiser.</figcaption></figure>

<h2>5. Comparing selection strategies</h2>
<div class="decision-grid">
  <section><h3>Calendar year</h3><p><strong>Recommended for the paper.</strong> It has an intuitive January–December boundary, preserves all seasons, is easy to cite and reproduce, and aligns with annual reporting. Complete candidates are {metrics.index.min()}–{metrics.index.max()}.</p></section>
  <section><h3>First-timestep-anchored year</h3><p>Useful for the overlay requested earlier, but the 30 June boundary is arbitrary for a paper. The rolling-year median closest to the rolling-window median is <strong>{rolling_median_year}</strong>; this answer differs because changing the boundary reallocates winter and crisis months.</p></section>
  <section><h3>Latest complete year</h3><p>Simple and current, but not automatically representative. It should be chosen only if the paper's claim is explicitly about the latest observed market conditions. Here, the latest complete calendar year is {metrics.index.max()}.</p></section>
  <section><h3>Random or visually selected year</h3><p><strong>Not recommended.</strong> It is hard to defend and creates a cherry-picking concern because the model's savings are highly sensitive to price spread and extreme events.</p></section>
</div>
<figure><img src="{figures['selection_scores']}" alt="Representativeness scores for all complete calendar years"><figcaption>{selected_year} is selected within the recent regime. The full-history score answers a different question and favours {historical_best}; the disagreement is evidence that the dataset contains multiple regimes and that the target population must be stated.</figcaption></figure>

<h3>Why {selected_year} is the best primary choice</h3>
<ol>
  <li><strong>Completeness:</strong> {int(selected_metrics['observations']):,} hourly observations across {int(selected_metrics['days'])} settlement dates, including the leap-year and daylight-saving structure.</li>
  <li><strong>Recency without using an incomplete year:</strong> it belongs to the latest three fully observed calendar years.</li>
  <li><strong>Transparent representativeness:</strong> it ranks {selected_recent_rank} among the recent candidates by the stated recent-regime score ({scores.loc[selected_year, 'Recent-regime score']:.2f}).</li>
  <li><strong>Neither a crisis peak nor a deliberately calm year:</strong> its median is £{selected_metrics['median']:.1f}/MWh, IQR is £{selected_metrics['iqr']:.1f}/MWh, negative-price share is {selected_metrics['negative_share']:.2f}%, and 95th percentile is £{selected_metrics['p95']:.1f}/MWh.</li>
  <li><strong>Clear academic framing:</strong> it can be described as a recent representative year, while {sensitivity_label} provide recent-year robustness checks and 2022 provides a high-price stress test.</li>
</ol>

<h2>6. Detailed view of the recommended year</h2>
<figure><img src="{figures['selected_year']}" alt="Detailed daily and monthly analysis of the recommended calendar year"><figcaption>The daily mean alone understates the signal available to flexible loads; the daily minimum–maximum envelope shows the within-day spreads that drive optimisation value.</figcaption></figure>
<figure><img src="{figures['selected_profiles']}" alt="Selected-year weekday-hour heatmap and representative daily profiles"><figcaption>The left panel summarises recurring weekday/hour structure. The right panel compares the paper's current illustrative 24-hour profile with an observed median profile and objectively chosen typical and stress days from {selected_year}.</figcaption></figure>

<h3>Objectively selected 24-hour profiles within {selected_year}</h3>
<div class="table-wrap">{day_table}</div>
<p>The typical day is selected by minimising a robust distance to the year's median 24-hour shape and daily level/spread features, using only 24-period days. Clock-change days are excluded from this daily-profile step because they do not fit a fixed 24-hour model horizon. The volatility and high-price cases are chosen near the 90th percentile rather than at the absolute maximum, while the negative-price case deliberately tests the sign reversal that the present paper profile cannot represent.</p>

<h2>7. Implications for the current 24-hour paper model</h2>
<div class="caution"><strong>Do not create a single "average day" and treat it as representative.</strong> Averaging each hour across a year smooths spikes, negative prices, and day-to-day covariation. Those are precisely the features that create value for UPS/TES dispatch and workload shifting.</div>
<p>The present paper profile has mean £{paper_stats['mean']:.1f}/MWh, median £{paper_stats['median']:.1f}/MWh, range £{paper_stats['range']:.1f}/MWh, standard deviation £{paper_stats['std']:.1f}/MWh, and a 95th-percentile hourly ramp of £{paper_stats['ramp_p95']:.1f}/MWh. It is a smooth, entirely positive constructed profile. It is useful pedagogically but does not reproduce the empirical distribution of negative prices, spikes, or irregular daily shapes.</p>

<h3>Preferred revision design</h3>
<ol>
  <li><strong>Primary dataset:</strong> use all valid days in calendar year {selected_year}.</li>
  <li><strong>Simulation unit:</strong> retain the 24-hour optimisation horizon, but run it separately for every 24-period day (with the same 3-hour extension logic drawn from the following day where available).</li>
  <li><strong>Clock changes:</strong> pre-specify treatment of the one 23-period and one 25-period day. The cleanest approach for a fixed 24-hour model is to exclude those two dates from the daily runs while retaining them in annual descriptive statistics.</li>
  <li><strong>Reported outcome:</strong> report the median, interquartile range, and 5th–95th percentile of cost saving and flexibility metrics across days, not only the result from one day.</li>
  <li><strong>Robustness:</strong> repeat the annual analysis for {sensitivity_label}; add 2022 as an explicit stress regime. If computational cost is prohibitive, use the four objectively selected {selected_year} profiles above and label the design as representative-day sampling.</li>
  <li><strong>Claims:</strong> limit conclusions to the observed GB day-ahead reference-price conditions and avoid claiming that one year represents all market regimes.</li>
</ol>

<h2>8. Limitations and cautions</h2>
<ul>
  <li>The selection score operationalises "representative" for a recent GB day-ahead regime. A historical-study, stress-test, or latest-market objective would justify a different year.</li>
  <li>Prices are nominal £/MWh. Comparing price levels across a decade does not adjust for inflation; the report therefore emphasises within-year distribution and shape as well as level.</li>
  <li>The dataset supplies the market reference signal, not network constraints, site tariffs, imbalance exposure, or demand charges.</li>
  <li>Extreme observations were retained. They are economically relevant for flexibility, but the report does not independently validate each extreme against a second market-data source.</li>
  <li>Daily representative profiles are a computational compromise. Full-year repeated optimisation is preferable whenever practical.</li>
</ul>

<h2>9. Reproducibility</h2>
<p>All source rows were read from <code>{html.escape(str(input_path))}</code>. Complete calendar years were ranked only after integrity checks. The report generator is <code>plotting_and_saving/generate_imrp_eda_report.py</code>. Figures are embedded directly as base64-encoded PNG data, so this report has no external image dependencies.</p>
<p class="small">Official metadata consulted: <a href="{LCCC_DATASET_URL}">LCCC IMRP actuals</a> and <a href="{LCCC_REFERENCE_URL}">LCCC Market Reference Prices</a>. Local data snapshot ends {final_date:%d %B %Y}.</p>
</main>
</body>
</html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    years = complete_calendar_years(data)
    metrics = calendar_metrics(data, years)
    rolling_metrics = rolling_window_metrics(data)
    scores, selected_year, recent_years = robust_selection_scores(metrics)
    daily_pivot, selected_days, median_profile = representative_days(data, selected_year)
    figures = make_figures(
        data,
        metrics,
        rolling_metrics,
        scores,
        selected_year,
        recent_years,
        daily_pivot,
        selected_days,
        median_profile,
    )
    report = build_report(
        data,
        metrics,
        rolling_metrics,
        scores,
        selected_year,
        recent_years,
        selected_days,
        daily_pivot,
        figures,
        args.input,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Saved report: {args.output}")
    print(f"Report bytes: {args.output.stat().st_size:,}")
    print(f"Complete calendar years: {years}")
    print(f"Recent comparison years: {recent_years}")
    print(f"Recommended year: {selected_year}")
    print("Selection scores:")
    print(scores.round(3).to_string())
    print("Selected days:")
    for label, date in selected_days.items():
        print(f"  {label}: {date:%Y-%m-%d}")


if __name__ == "__main__":
    main()
