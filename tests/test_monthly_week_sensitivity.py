from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rolling_optimisation.timeline import build_annual_timeline
from run_monthly_week_sensitivity import (
    PRICE_FEATURES,
    select_representative_weeks,
)


ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "static" / "data" / "imrp_actuals.csv"
LOAD = ROOT / "static" / "data" / "inputs" / "load_profiles.csv"
SHIFT = ROOT / "static" / "data" / "inputs" / "shiftability_profile.csv"


@pytest.fixture(scope="module")
def selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    timeline = build_annual_timeline(PRICE, LOAD, SHIFT, 2025)
    return select_representative_weeks(timeline, 2025)


def test_selects_one_complete_monday_week_per_month(
    selection: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    selected, candidates = selection

    assert selected["month"].tolist() == list(range(1, 13))
    assert len(selected) == 12
    assert set(candidates["month"]) == set(range(1, 13))
    for row in selected.itertuples(index=False):
        start = pd.Timestamp(row.week_start)
        end = pd.Timestamp(row.week_end)
        assert start.weekday() == 0
        assert end - start == pd.Timedelta(days=6)
        assert start.month == row.month
        assert end.month == row.month
        assert row.week_intervals in {668, 672, 676}
        assert row.month_intervals > row.week_intervals
        assert row.annualisation_weight > 1


def test_joint_selection_matches_annual_negative_price_share(
    selection: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    selected, candidates = selection

    assert selected["selection_total_score"].nunique() == 1
    assert (
        abs(
            selected["selected_weighted_negative_price_share"].iloc[0]
            - selected["annual_negative_price_share"].iloc[0]
        )
        <= selected["negative_share_tolerance"].iloc[0]
    )
    assert (
        abs(
            selected["selected_weighted_mean_price_gbp_per_mwh"].iloc[0]
            - selected["annual_mean_price_gbp_per_mwh"].iloc[0]
        )
        <= selected["mean_price_tolerance_gbp_per_mwh"].iloc[0]
    )
    assert (selected["selection_score"] >= 0).all()
    assert candidates["selection_score"].notna().all()
    for feature in PRICE_FEATURES:
        assert f"monthly_{feature}" in selected.columns
