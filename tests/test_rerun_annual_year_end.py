from __future__ import annotations

import pandas as pd
import pytest

from tasks.rerun_annual_year_end import (
    _maximum_numeric_frame_difference,
    _tail_audit,
)


def test_tail_audit_requires_first_three_hours_of_following_year() -> None:
    core = pd.DataFrame(
        {
            "source_date": ["2025-12-31"] * 96,
            "source_period": [1] * 96,
            "settlement_price_gbp_per_mwh": [0.0] * 96,
        }
    )
    tail = pd.DataFrame(
        {
            "source_date": ["2026-01-01"] * 12,
            "source_period": [1] * 4 + [2] * 4 + [3] * 4,
            "settlement_price_gbp_per_mwh": (
                [46.49] * 4 + [40.19] * 4 + [28.99] * 4
            ),
        }
    )

    audit = _tail_audit(pd.concat([core, tail], ignore_index=True), 96, 2025)

    assert audit == {
        "source_date": "2026-01-01",
        "source_periods": [1, 2, 3],
        "prices_gbp_per_mwh": [46.49, 40.19, 28.99],
        "quarter_hour_intervals": 12,
    }


def test_tail_audit_rejects_same_year_wraparound() -> None:
    horizon = pd.DataFrame(
        {
            "source_date": ["2025-01-01"] * 12,
            "source_period": [1] * 4 + [2] * 4 + [3] * 4,
            "settlement_price_gbp_per_mwh": [0.0] * 12,
        }
    )

    with pytest.raises(RuntimeError, match="2026-01-01"):
        _tail_audit(horizon, 0, 2025)


def test_numeric_frame_difference_reports_maximum_absolute_change() -> None:
    source = pd.DataFrame(
        {"timestamp": ["a", "b"], "grid_kw": [1.0, 2.0], "cost": [3.0, 4.0]}
    )
    rerun = source.copy()
    rerun.loc[1, "grid_kw"] += 0.25
    rerun.loc[0, "cost"] -= 0.1

    assert _maximum_numeric_frame_difference(source, rerun) == pytest.approx(0.25)
