# test_flex_duration.py
import pandas as pd
import pytest
import pathlib
import re

DATA_DIR = pathlib.Path("static/data")
# Tolerance for constraint slack in kW. 1e-3 kW = 1 W
TOL_KW = 1e-3


def get_flex_from_filename(path: pathlib.Path) -> float:
    """
    Extracts the flex magnitude (can be negative) from filenames like:
      flex_duration_detailed_results_ts1_flex100.csv
      flex_duration_detailed_results_ts1_flex-50.csv
      flex_duration_detailed_results_ts1_flexneg50.csv
    """
    match = re.search(r"_flex(-?\d+)", path.stem)
    if match:
        return float(match.group(1))

    match_neg = re.search(r"_flexneg(\d+)", path.stem)
    if match_neg:
        return -float(match_neg.group(1))

    raise ValueError(f"Could not extract flex magnitude from filename {path}")


@pytest.fixture
def results_df():
    csv_candidates = list(DATA_DIR.glob("flex_duration_detailed_results_ts*_flex*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No flex_duration_detailed_results CSV found in {DATA_DIR}")
    csv_file = csv_candidates[0]
    df = pd.read_csv(csv_file, index_col="Time_Slot_EXT")
    df.attrs["source_file"] = csv_file
    return df


def test_cpu_load_consistency(results_df):
    """Check that inflexible + flexible CPU load equals total CPU load (opt)."""
    diff = (
        results_df["Inflexible_Load_CPU_Opt"]
        + results_df["Flexible_Load_CPU_Opt"]
        - results_df["Total_CPU_Load_opt"]
    )
    max_abs = abs(diff).max()
    assert max_abs < 1e-6, f"CPU load mismatch: max diff {max_abs}"


def test_power_constraint(results_df):
    """
    Check that optimized GRID draw does not exceed baseline grid draw + flex target.

    Prints a per-timestep PASS/FAIL summary showing:
      - optimized grid draw (kW)
      - baseline grid draw (kW)
      - allowed grid draw = baseline + flex target (kW)
      - increase_from_base = opt - base
      - diff_from_allowed = opt - allowed
    """
    csv_file = results_df.attrs["source_file"]
    flex_target = get_flex_from_filename(csv_file)

    grid_cols_base = [
        "P_Grid_IT_kW_base",
        "P_Grid_Cooling_kW_base",
        "P_Grid_Other_kW_base",
        "P_UPS_Charge_kW_base",
    ]
    grid_cols_opt = [
        "P_Grid_IT_kW_opt",
        "P_Grid_Cooling_kW_opt",
        "P_Grid_Other_kW_opt",
        "P_UPS_Charge_kW_opt",
    ]

    missing = [c for c in (grid_cols_base + grid_cols_opt) if c not in results_df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in results CSV: {missing}")

    grid_base = results_df[grid_cols_base].sum(axis=1).fillna(0.0)
    grid_opt = results_df[grid_cols_opt].sum(axis=1).fillna(0.0)
    allowed = grid_base + flex_target

    diff_from_allowed = grid_opt - allowed
    pass_mask = diff_from_allowed <= TOL_KW

    # Report
    status_lines = []
    for i, idx in enumerate(results_df.index):
        gopt = float(grid_opt.iloc[i])
        gbase = float(grid_base.iloc[i])
        allow = float(allowed.iloc[i])
        inc_base = gopt - gbase
        diff_allow = float(diff_from_allowed.iloc[i])
        status = "PASS" if pass_mask.iloc[i] else "FAIL"
        status_lines.append(
            f"{idx}: {status} "
            f"(opt={gopt:.4f} kW, base={gbase:.4f} kW, allowed={allow:.4f} kW, "
            f"increase_from_base={inc_base:.4f} kW, "
            f"diff_from_allowed={diff_allow:.6f} kW)"
        )

    print(f"\nGrid draw check per timestep (TOL_KW = {TOL_KW:.6f} kW):")
    print("\n".join(status_lines))

    if not pass_mask.all():
        failing_indices = [results_df.index[i] for i in range(len(pass_mask)) if not pass_mask.iloc[i]]
        failing_lines = [ln for i, ln in enumerate(status_lines) if not pass_mask.iloc[i]]

        num_fail = len(failing_indices)
        max_violation_kw = float(diff_from_allowed[~pass_mask].max())
        max_violation_idx = int(diff_from_allowed[~pass_mask].idxmax()) if not diff_from_allowed[~pass_mask].empty else None

        fail_message = (
            f"{num_fail} timestep(s) violate the grid constraint (slack={TOL_KW} kW).\n"
            f"Max violation = {max_violation_kw:.6f} kW at timestep {max_violation_idx}.\n"
            f"Failing timesteps: {list(failing_indices)}\n\n"
            "Details:\n" + "\n".join(failing_lines)
        )
        pytest.fail(fail_message)


def test_no_negative_values(results_df):
    """Ensure no negative values in power/load columns (except _diff columns)."""
    for col in results_df.columns:
        if "_diff" in col or "Price" in col:
            continue
        if (results_df[col] < -1e-9).any():
            bad_rows = results_df.index[results_df[col] < -1e-9].tolist()
            pytest.fail(f"Negative values found in column {col} at timesteps: {bad_rows}")


def test_price_series_length(results_df):
    """Electricity price column should match dataframe length."""
    assert "Price_GBP_per_MWh" in results_df.columns, "Price_GBP_per_MWh column missing"
    assert len(results_df["Price_GBP_per_MWh"]) == len(results_df)
