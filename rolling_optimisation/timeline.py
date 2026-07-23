from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = {"IMRP_Date", "Settlement_Period", "IMRP_Amount"}
TRANCHE_DELAYS = {1: 2, 2: 4, 3: 8, 4: 12}


def combined_input_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validated_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    missing = PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    prices["IMRP_Date"] = pd.to_datetime(prices["IMRP_Date"], errors="raise").dt.normalize()
    prices["Settlement_Period"] = pd.to_numeric(
        prices["Settlement_Period"], errors="raise"
    ).astype(int)
    prices["IMRP_Amount"] = pd.to_numeric(prices["IMRP_Amount"], errors="raise")
    prices = prices.sort_values(["IMRP_Date", "Settlement_Period"], kind="stable")
    if prices.duplicated(["IMRP_Date", "Settlement_Period"]).any():
        raise ValueError("Duplicate IMRP date/settlement-period keys found")
    if prices[list(PRICE_COLUMNS)].isna().any().any():
        raise ValueError("Null values found in required IMRP columns")
    for date, group in prices.groupby("IMRP_Date", sort=True):
        expected = list(range(1, len(group) + 1))
        if group["Settlement_Period"].tolist() != expected:
            raise ValueError(f"Non-contiguous settlement periods on {date.date()}")
    return prices


def build_annual_timeline(
    price_path: Path,
    load_profile_path: Path,
    shiftability_path: Path,
    year: int,
    lookahead_steps: int = 12,
    tail_price_mode: str = "actual",
) -> pd.DataFrame:
    """Build a continuous 15-minute UTC timeline with local reporting labels."""

    if tail_price_mode not in {"actual", "repeat_last"}:
        raise ValueError("tail_price_mode must be 'actual' or 'repeat_last'")
    if lookahead_steps % 4:
        raise ValueError("lookahead_steps must be a whole number of hours")

    prices = _validated_prices(price_path)
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year + 1}-01-01")
    target_hourly = prices[
        (prices["IMRP_Date"] >= start_date) & (prices["IMRP_Date"] < end_date)
    ].copy()
    expected_hours = 8784 if start_date.is_leap_year else 8760
    if len(target_hourly) != expected_hours:
        raise ValueError(
            f"{year} contains {len(target_hourly)} source hours; expected {expected_hours}"
        )

    tail_hours = math.ceil(lookahead_steps / 4)
    tail_hourly = prices[prices["IMRP_Date"] >= end_date].head(tail_hours).copy()
    if len(tail_hourly) != tail_hours:
        raise ValueError(f"Need {tail_hours} source hours after {year}-12-31")
    if tail_price_mode == "repeat_last":
        tail_hourly["IMRP_Amount"] = float(target_hourly["IMRP_Amount"].iloc[-1])

    hourly = pd.concat([target_hourly, tail_hourly], ignore_index=True)
    repeated = hourly.loc[hourly.index.repeat(4)].reset_index(drop=True)
    repeated["quarter_in_hour"] = np.tile(np.arange(4), len(hourly))

    utc = pd.date_range(
        f"{year}-01-01 00:00:00+00:00",
        periods=len(repeated),
        freq="15min",
    )
    local = utc.tz_convert("Europe/London")
    repeated["timestamp_utc"] = utc
    repeated["timestamp_local"] = local
    repeated["local_date"] = pd.Index(local.date).astype(str)
    repeated["profile_slot"] = local.hour * 4 + local.minute // 15 + 1
    repeated = repeated.rename(
        columns={
            "IMRP_Date": "source_date",
            "Settlement_Period": "source_period",
            "IMRP_Amount": "settlement_price_gbp_per_mwh",
        }
    )
    repeated["source_date"] = repeated["source_date"].dt.strftime("%Y-%m-%d")

    # The source date uses the GB local trading day. This assertion catches a
    # shifted or differently defined price file before it reaches the model.
    if not (repeated["source_date"] == repeated["local_date"]).all():
        mismatch = repeated.loc[
            repeated["source_date"] != repeated["local_date"],
            ["timestamp_utc", "timestamp_local", "source_date"],
        ].head()
        raise ValueError(f"Source dates do not align with Europe/London labels:\n{mismatch}")

    loads = pd.read_csv(load_profile_path, index_col="time_slot")
    required_loads = {"inflexible_load", "flexible_load"}
    if not required_loads.issubset(loads.columns):
        raise ValueError(f"{load_profile_path} must contain {sorted(required_loads)}")
    if not set(range(1, 97)).issubset(set(loads.index.astype(int))):
        raise ValueError("Load profile must contain time slots 1..96")
    loads.index = loads.index.astype(int)

    shift = pd.read_csv(shiftability_path, index_col="time_slot")
    shift.index = shift.index.astype(int)
    shift.columns = shift.columns.astype(int)
    if not set(range(1, 97)).issubset(set(shift.index)):
        raise ValueError("Shiftability profile must contain time slots 1..96")
    if not set(TRANCHE_DELAYS).issubset(set(shift.columns)):
        raise ValueError("Shiftability profile must contain tranches 1..4")
    shift_sums = shift[list(TRANCHE_DELAYS)].sum(axis=1)
    if not np.allclose(shift_sums.loc[1:96], 1.0, atol=1e-8):
        raise ValueError("Shiftability fractions must sum to one in every profile slot")

    slots = repeated["profile_slot"].astype(int)
    repeated["inflexible_cpu"] = slots.map(loads["inflexible_load"]).astype(float)
    repeated["flexible_cpu"] = slots.map(loads["flexible_load"]).astype(float)
    for tranche in TRANCHE_DELAYS:
        repeated[f"shift_fraction_{tranche}"] = slots.map(shift[tranche]).astype(float)

    repeated["is_target_year"] = repeated["timestamp_utc"] < pd.Timestamp(
        f"{year + 1}-01-01 00:00:00+00:00"
    )
    target = repeated[repeated["is_target_year"]]
    if len(target) != expected_hours * 4:
        raise AssertionError("Unexpected committed quarter-hour count")
    if target["timestamp_utc"].duplicated().any():
        raise AssertionError("Duplicate UTC intervals in annual timeline")
    spacing = target["timestamp_utc"].diff().dropna()
    if not (spacing == pd.Timedelta(minutes=15)).all():
        raise AssertionError("Gap in annual UTC timeline")
    if target["local_date"].nunique() != (366 if start_date.is_leap_year else 365):
        raise AssertionError("Not every local calendar date is represented")
    return repeated.reset_index(drop=True)


def local_day_core_indices(timeline: pd.DataFrame, year: int) -> list[tuple[str, np.ndarray]]:
    target = timeline[timeline["is_target_year"]]
    rows: list[tuple[str, np.ndarray]] = []
    for date, group in target.groupby("local_date", sort=False):
        if int(date[:4]) == year:
            rows.append((date, group.index.to_numpy(dtype=int)))
    expected_days = 366 if pd.Timestamp(f"{year}-01-01").is_leap_year else 365
    if len(rows) != expected_days:
        raise AssertionError(f"Expected {expected_days} local dates, found {len(rows)}")
    return rows


def add_optimisation_prices(
    timeline: pd.DataFrame,
    treatment: str,
) -> pd.DataFrame:
    """Add an objective-price column while preserving signed settlement prices."""

    if treatment not in {"signed", "floor_zero", "shift_year_min"}:
        raise ValueError("Unknown price treatment")
    result = timeline.copy()
    settlement = result["settlement_price_gbp_per_mwh"]
    if treatment == "signed":
        optimisation = settlement
    elif treatment == "floor_zero":
        optimisation = settlement.clip(lower=0.0)
    else:
        target_minimum = float(
            result.loc[result["is_target_year"], "settlement_price_gbp_per_mwh"].min()
        )
        optimisation = settlement + max(0.0, -target_minimum)
    result["optimisation_price_gbp_per_mwh"] = optimisation
    return result
