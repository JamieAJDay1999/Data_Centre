"""
Thin orchestration wrappers around the existing modelling scripts.

These functions *reuse* the model-building, solving and post-processing code in
``nominal_calculation.py`` and ``optimisation.py`` exactly as it is written.
They only add two things the interactive dashboard needs:

  1. a small, whitelisted set of parameter overrides so a scenario can be
     explored from the browser, and
  2. reshaping of the returned DataFrames into the same JSON structure the
     read-only ``data_access`` layer produces, so the front-end can render
     live results and stored results with identical code.

No file in the repository root is modified by importing this module.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless – never try to open a plotting window

import numpy as np

import nominal_calculation as nc
import optimisation as opt
from inputs.parameters_optimisation import ModelParameters

DT_HOURS = 0.25
SLOTS_PER_DAY = 96

# Whitelisted overrides: key -> (attribute setter). Everything else is ignored.
NUMERIC_OVERRIDES = {
    "cop_hvac": "COP_HVAC",
    "max_power_kw": "max_power_kw",
    "idle_power_kw": "idle_power_kw",
    "overhead_kw": "nominal_overhead_addition",
    "p_chiller_max_w": "P_chiller_max",
}


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _apply_param_overrides(params: ModelParameters, ov: dict) -> None:
    """Apply the whitelisted scalar overrides in-place."""
    for key, attr in NUMERIC_OVERRIDES.items():
        if ov.get(key) is not None:
            setattr(params, attr, float(ov[key]))

    # UPS energy capacity – recompute the derived SoC bounds.
    if ov.get("ups_capacity_kwh") is not None:
        cap = float(ov["ups_capacity_kwh"])
        params.e_nom_kwh = cap
        params.e_min_kwh = params.soc_min * cap
        params.e_max_kwh = params.soc_max * cap
        params.e_start_kwh = params.soc_max * cap

    # TES thermal storage capacity.
    if ov.get("tes_capacity_kwh") is not None:
        cap = float(ov["tes_capacity_kwh"])
        params.TES_kwh_cap = cap
        params.TES_capacity_kWh = cap
        params.TES_initial_charge_kWh = 0.5 * cap


def _transform_tariff(data: dict, ov: dict) -> None:
    """Scale the design tariff to explore price-level / volatility sensitivity.

    ``tariff_scale``      – multiplies the whole price series.
    ``tariff_volatility`` – amplifies (or damps) deviations about the daily mean
                            while preserving the average level.
    Operates on ``data['electricity_price']`` before the model is built, so the
    objective and all reported costs reflect the scenario tariff.
    """
    scale = float(ov.get("tariff_scale", 1.0) or 1.0)
    vol = float(ov.get("tariff_volatility", 1.0) or 1.0)
    if scale == 1.0 and vol == 1.0:
        return
    price = np.array(data["electricity_price"], dtype=float)
    body = price[1:]  # index 0 is a sentinel 0
    mean = body.mean()
    body = mean + vol * (body - mean)
    body = np.clip(body * scale, 0.0, None)
    price[1:] = body
    data["electricity_price"] = price


# ---------------------------------------------------------------------------
# Payload reshaping (mirrors webapp.data_access column mapping)
# ---------------------------------------------------------------------------
def _col(df, name):
    if name not in df.columns:
        return []
    return [_clean(v) for v in df[name].tolist()]


def _hours(n):
    return [round(i * DT_HOURS, 4) for i in range(n)]


def _nominal_payload(df) -> dict:
    day = df.iloc[:SLOTS_PER_DAY] if len(df) >= SLOTS_PER_DAY else df
    it_kwh = day["P_Grid_IT_kW"].sum() * DT_HOURS
    cooling_kwh = day["P_Grid_Cooling_kW"].sum() * DT_HOURS
    other_kwh = (day["P_Grid_Other_kW"].sum() + day["P_UPS_Charge_kW"].sum()) * DT_HOURS
    total_kwh = day["P_Total_kW"].sum() * DT_HOURS
    cost = day["Nominal_Cost"].sum()
    peak = day["P_Total_kW"].max()
    return {
        "available": True,
        "hours": _hours(len(df)),
        "price": _col(df, "Price_GBP_per_MWh"),
        "cost_per_step": _col(df, "Nominal_Cost"),
        "p_total": _col(df, "P_Total_kW"),
        "p_it": _col(df, "P_Grid_IT_kW"),
        "p_cooling": _col(df, "P_Grid_Cooling_kW"),
        "p_other": _col(df, "P_Grid_Other_kW"),
        "t_it": _col(df, "T_IT_Celsius"),
        "t_cold_aisle": _col(df, "T_ColdAisle_Celsius"),
        "e_tes": _col(df, "E_TES_kWh"),
        "cpu_inflexible": _col(df, "Inflexible_Load_CPU"),
        "cpu_flexible": _col(df, "Flexible_Load_CPU"),
        "summary": {
            "cost_gbp": _clean(cost),
            "energy_total_kwh": _clean(total_kwh),
            "pct_it": _clean(100 * it_kwh / total_kwh) if total_kwh else 0,
            "pct_cooling": _clean(100 * cooling_kwh / total_kwh) if total_kwh else 0,
            "pct_other": _clean(100 * other_kwh / total_kwh) if total_kwh else 0,
            "peak_kw": _clean(peak),
            "pue": _clean(total_kwh / it_kwh) if it_kwh else None,
        },
    }


def _opt_payload(df, nominal_df) -> dict:
    # Use scenario-consistent nominal series for the comparison.
    p_total_nom = nominal_df["P_Total_kW"].tolist()
    nominal_cost_step = nominal_df["Nominal_Cost"].tolist()
    n = min(len(df), len(p_total_nom))

    day = min(SLOTS_PER_DAY, n)
    nominal_cost = float(np.sum(nominal_cost_step[:day]))
    opt_cost = float(df["Optimized_Cost_per_Step"].iloc[:day].sum())
    saving_abs = nominal_cost - opt_cost
    saving_rel = (100 * saving_abs / nominal_cost) if nominal_cost else 0
    peak_nom = float(np.max(p_total_nom[:day]))
    peak_opt = float(df["P_Total_kW"].iloc[:day].max())

    return {
        "available": True,
        "hours": _hours(len(df)),
        "price": _col(df, "Price_GBP_per_MWh"),
        "p_total_opt": _col(df, "P_Total_kW"),
        "p_total_nominal": [_clean(v) for v in p_total_nom],
        "p_it": _col(df, "P_Grid_IT_kW"),
        "p_hvac": _col(df, "P_Chiller_HVAC_kW"),
        "p_tes": _col(df, "P_Chiller_TES_kW"),
        "p_other": _col(df, "P_Grid_Other_kW"),
        "p_ups_charge": _col(df, "P_UPS_Charge_kW"),
        "p_ups_discharge": _col(df, "P_UPS_Discharge_kW"),
        "cpu_inflexible": _col(df, "Inflexible_Load_CPU_Opt"),
        "cpu_flexible_opt": _col(df, "Flexible_Load_CPU_Opt"),
        "cpu_flexible_nom": _col(df, "Flexible_Load_CPU_Nom"),
        "cost_opt_step": _col(df, "Optimized_Cost_per_Step"),
        "cost_nom_step": [_clean(v) for v in nominal_cost_step],
        "summary": {
            "nominal_cost_gbp": _clean(nominal_cost),
            "optimised_cost_gbp": _clean(opt_cost),
            "saving_abs_gbp": _clean(saving_abs),
            "saving_rel_pct": _clean(saving_rel),
            "peak_nominal_kw": _clean(peak_nom),
            "peak_optimised_kw": _clean(peak_opt),
            "peak_shave_kw": _clean(peak_nom - peak_opt),
            "peak_shave_pct": _clean(100 * (peak_nom - peak_opt) / peak_nom)
            if peak_nom
            else 0,
        },
    }


# ---------------------------------------------------------------------------
# Public entry points used by the Flask API
# ---------------------------------------------------------------------------
def run_nominal(overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    params = nc.ModelParameters()
    params = nc.configure_nominal_params(params)
    _apply_param_overrides(params, overrides)
    data = nc.load_and_prepare_data(params)
    _transform_tariff(data, overrides)

    cost, df, _ = nc.run_single_calculation(params, data, msg=False, linear=False)
    if df is None:
        return {"available": False, "error": "Solver did not find an optimal solution."}
    return _nominal_payload(df)


def run_optimisation(overrides: dict | None = None) -> dict:
    """Run a scenario-consistent baseline *and* optimisation, then compare."""
    overrides = overrides or {}

    # 1) scenario-consistent nominal baseline (so savings are like-for-like)
    nparams = nc.ModelParameters()
    nparams = nc.configure_nominal_params(nparams)
    _apply_param_overrides(nparams, overrides)
    ndata = nc.load_and_prepare_data(nparams)
    _transform_tariff(ndata, overrides)
    _, nominal_df, _ = nc.run_single_calculation(nparams, ndata, msg=False, linear=False)
    if nominal_df is None:
        return {"available": False, "error": "Baseline solve failed."}

    # 2) cost-minimising optimisation with the same scenario parameters
    oparams = ModelParameters()
    _apply_param_overrides(oparams, overrides)
    odata = opt.load_and_prepare_data(oparams)
    _transform_tariff(odata, overrides)
    cost, opt_df, _ = opt.run_single_optimization(oparams, odata, msg=False)
    if opt_df is None:
        return {"available": False, "error": "Optimisation solve failed."}

    return _opt_payload(opt_df, nominal_df)
