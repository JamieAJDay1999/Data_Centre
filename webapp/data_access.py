"""
Read-only access layer for the Data Centre dashboard.

This module reads the CSV / image artefacts that the existing modelling
scripts (``nominal_calculation.py``, ``optimisation.py``,
``flexibility_duration.py``) already produce and reshapes them into small,
JSON-friendly structures for the web front-end.

It deliberately does **not** import or modify any of the modelling logic –
it only reads files from ``static/data`` so the dashboard can display results
that already exist on disk without re-solving anything.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd

# --- Paths (relative to the repository root, which is the app working dir) ---
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "static" / "data"
INPUTS = DATA / "inputs"
NOMINAL = DATA / "nominal_outputs"
OPT = DATA / "optimisation_outputs"
FLEX = DATA / "flexibility_outputs"

SLOTS_PER_DAY = 96          # 24 h at 15-minute resolution
DT_HOURS = 0.25             # 15-minute time step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """Convert a value into something ``json.dumps`` will accept."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _series(df: pd.DataFrame, column: str) -> list:
    """Return a JSON-safe list for one column, or an empty list if missing."""
    if column not in df.columns:
        return []
    return [_clean(v) for v in df[column].tolist()]


def _read_csv(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _hours_axis(n: int) -> list:
    """A time axis in hours for ``n`` 15-minute slots (slot 1 -> hour 0)."""
    return [round(i * DT_HOURS, 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Availability – which artefacts exist on disk right now
# ---------------------------------------------------------------------------
def availability() -> dict:
    return {
        "nominal": (NOMINAL / "nominal_case_results.csv").exists(),
        "optimisation": (OPT / "optimised_baseline.csv").exists(),
        "flexibility": (FLEX / "flex_duration_results_full.csv").exists()
        or (FLEX / "flex_duration_results.csv").exists(),
        "inputs": (INPUTS / "load_profiles.csv").exists(),
    }


# ---------------------------------------------------------------------------
# Nominal baseline
# ---------------------------------------------------------------------------
def nominal_timeseries() -> dict:
    df = _read_csv(NOMINAL / "nominal_case_results.csv")
    if df is None:
        return {"available": False}

    n = len(df)
    hours = _hours_axis(n)
    return {
        "available": True,
        "hours": hours,
        "slots": _series(df, "Time_Slot_EXT"),
        "price": _series(df, "Price_GBP_per_MWh"),
        "cost_per_step": _series(df, "Nominal_Cost"),
        "p_total": _series(df, "P_Total_kW"),
        "p_it": _series(df, "P_Grid_IT_kW"),
        "p_cooling": _series(df, "P_Grid_Cooling_kW"),
        "p_hvac": _series(df, "P_Chiller_HVAC_kW"),
        "p_tes": _series(df, "P_Chiller_TES_kW"),
        "p_other": _series(df, "P_Grid_Other_kW"),
        "p_ups_charge": _series(df, "P_UPS_Charge_kW"),
        "t_it": _series(df, "T_IT_Celsius"),
        "t_rack": _series(df, "T_Rack_Celsius"),
        "t_cold_aisle": _series(df, "T_ColdAisle_Celsius"),
        "t_hot_aisle": _series(df, "T_HotAisle_Celsius"),
        "e_tes": _series(df, "E_TES_kWh"),
        "e_ups": _series(df, "E_UPS_kWh"),
        "cpu_total": _series(df, "Total_CPU_Load"),
        "cpu_inflexible": _series(df, "Inflexible_Load_CPU"),
        "cpu_flexible": _series(df, "Flexible_Load_CPU"),
        "summary": _nominal_summary(df),
    }


def _nominal_summary(df: pd.DataFrame) -> dict:
    """Energy breakdown + headline cost for the 24-hour operating day."""
    day = df.iloc[:SLOTS_PER_DAY] if len(df) >= SLOTS_PER_DAY else df

    it_kwh = day["P_Grid_IT_kW"].sum() * DT_HOURS
    cooling_kwh = day["P_Grid_Cooling_kW"].sum() * DT_HOURS
    other_kwh = (
        day["P_Grid_Other_kW"].sum() + day.get("P_UPS_Charge_kW", pd.Series([0])).sum()
    ) * DT_HOURS
    total_kwh = day["P_Total_kW"].sum() * DT_HOURS
    cost = day["Nominal_Cost"].sum()
    peak_kw = day["P_Total_kW"].max()
    mean_kw = day["P_Total_kW"].mean()

    pue = (total_kwh / it_kwh) if it_kwh else None
    return {
        "cost_gbp": _clean(cost),
        "energy_total_kwh": _clean(total_kwh),
        "energy_it_kwh": _clean(it_kwh),
        "energy_cooling_kwh": _clean(cooling_kwh),
        "energy_other_kwh": _clean(other_kwh),
        "pct_it": _clean(100 * it_kwh / total_kwh) if total_kwh else 0,
        "pct_cooling": _clean(100 * cooling_kwh / total_kwh) if total_kwh else 0,
        "pct_other": _clean(100 * other_kwh / total_kwh) if total_kwh else 0,
        "peak_kw": _clean(peak_kw),
        "mean_kw": _clean(mean_kw),
        "load_factor": _clean(mean_kw / peak_kw) if peak_kw else None,
        "pue": _clean(pue),
    }


# ---------------------------------------------------------------------------
# Cost optimisation
# ---------------------------------------------------------------------------
def optimisation_timeseries() -> dict:
    df = _read_csv(OPT / "optimised_baseline.csv")
    if df is None:
        return {"available": False}

    n = len(df)
    hours = _hours_axis(n)
    return {
        "available": True,
        "hours": hours,
        "price": _series(df, "Price_GBP_per_MWh"),
        "p_total_opt": _series(df, "P_Total_kW"),
        "p_total_nominal": _series(df, "P_Total_kW_Nominal"),
        "p_it": _series(df, "P_Grid_IT_kW"),
        "p_hvac": _series(df, "P_Chiller_HVAC_kW"),
        "p_tes": _series(df, "P_Chiller_TES_kW"),
        "p_other": _series(df, "P_Grid_Other_kW"),
        "p_ups_charge": _series(df, "P_UPS_Charge_kW"),
        "p_ups_discharge": _series(df, "P_UPS_Discharge_kW"),
        "q_tes_discharge_w": _series(df, "Q_Discharge_TES_Watts"),
        "cpu_inflexible": _series(df, "Inflexible_Load_CPU_Opt"),
        "cpu_flexible_opt": _series(df, "Flexible_Load_CPU_Opt"),
        "cpu_flexible_nom": _series(df, "Flexible_Load_CPU_Nom"),
        "cost_opt_cum": _series(df, "Optimized_Cost"),
        "cost_opt_step": _series(df, "Optimized_Cost_per_Step"),
        "cost_nom_step": _series(df, "Nominal_Cost"),
        "summary": _optimisation_summary(df),
    }


def _optimisation_summary(df: pd.DataFrame) -> dict:
    day = df.iloc[:SLOTS_PER_DAY] if len(df) >= SLOTS_PER_DAY else df
    nominal_cost = day["Nominal_Cost"].sum()
    opt_cost = day["Optimized_Cost_per_Step"].sum()
    saving_abs = nominal_cost - opt_cost
    saving_rel = (100 * saving_abs / nominal_cost) if nominal_cost else 0

    peak_opt = day["P_Total_kW"].max()
    peak_nom = day["P_Total_kW_Nominal"].max()
    peak_shave = peak_nom - peak_opt
    peak_shave_pct = (100 * peak_shave / peak_nom) if peak_nom else 0
    return {
        "nominal_cost_gbp": _clean(nominal_cost),
        "optimised_cost_gbp": _clean(opt_cost),
        "saving_abs_gbp": _clean(saving_abs),
        "saving_rel_pct": _clean(saving_rel),
        "peak_nominal_kw": _clean(peak_nom),
        "peak_optimised_kw": _clean(peak_opt),
        "peak_shave_kw": _clean(peak_shave),
        "peak_shave_pct": _clean(peak_shave_pct),
    }


# ---------------------------------------------------------------------------
# Flexibility (demand response)
# ---------------------------------------------------------------------------
def flex_heatmap() -> dict:
    df = _read_csv(FLEX / "flex_duration_results_full.csv")
    if df is None:
        df = _read_csv(FLEX / "flex_duration_results.csv")
    if df is None:
        return {"available": False}

    df = df.dropna(subset=["Timestep", "Flex_Magnitude_kW"])
    timesteps = sorted(df["Timestep"].unique().tolist())
    magnitudes = sorted(df["Flex_Magnitude_kW"].unique().tolist(), reverse=True)

    # z[row=magnitude][col=timestep] = achievable duration (hours)
    lookup = {
        (int(r.Timestep), int(r.Flex_Magnitude_kW)): float(r.Max_Duration_Min) / 60.0
        for r in df.itertuples()
    }
    z = [
        [lookup.get((int(ts), int(mag)), None) for ts in timesteps]
        for mag in magnitudes
    ]

    # which (ts, mag) pairs have a detailed time-series file on disk?
    detail_available = {}
    for ts in timesteps:
        for mag in magnitudes:
            detail_available[f"{int(ts)}_{int(mag)}"] = _detail_path(
                int(ts), int(mag)
            ).exists()

    start_hours = [round((int(ts) - 1) * DT_HOURS, 3) for ts in timesteps]
    durations = [d for row in z for d in row if d is not None]
    reductions = [int(m) for m in magnitudes if m < 0]
    increases = [int(m) for m in magnitudes if m > 0]
    return {
        "available": True,
        "timesteps": [int(t) for t in timesteps],
        "start_hours": start_hours,
        "magnitudes": [int(m) for m in magnitudes],
        "z": z,
        "detail_available": detail_available,
        "max_duration_h": _clean(max(durations)) if durations else 0,
        "reduction_magnitudes": reductions,
        "increase_magnitudes": increases,
        "max_reduction_kw": min(reductions) if reductions else None,
        "max_increase_kw": max(increases) if increases else None,
    }


def _detail_path(ts: int, mag: int) -> pathlib.Path:
    tag = str(mag).replace("-", "neg")
    return FLEX / f"flex_duration_detailed_results_ts{ts}_flex{tag}.csv"


def flex_detail(ts: int, mag: int) -> dict:
    path = _detail_path(ts, mag)
    df = _read_csv(path)
    if df is None:
        return {"available": False}

    n = len(df)
    # Detailed files are indexed from the event start slot; build an axis of
    # minutes-from-event-start for readability.
    minutes = [round(i * DT_HOURS * 60) for i in range(n)]
    return {
        "available": True,
        "ts": ts,
        "magnitude": mag,
        "minutes": minutes,
        "p_total_base": _series(df, "P_Total_kw_Base"),
        "p_total_opt": _series(df, "P_Total_kw_Opt"),
        "p_total_diff": _series(df, "P_Total_kw_Diff"),
        "p_it_diff": _series(df, "P_Grid_IT_kW_diff"),
        "p_cooling_diff": _series(df, "P_Grid_Cooling_kW_diff"),
        "p_ups_net_diff": _series(df, "P_UPS_NET_kw_Diff"),
        "e_ups_opt": _series(df, "E_UPS_kWh_Opt"),
        "e_tes_opt": _series(df, "E_TES_kWh_opt"),
        "t_it_opt": _series(df, "T_IT_Celsius_opt"),
        "t_cold_aisle_opt": _series(df, "T_ColdAisle_Celsius_opt"),
        "price": _series(df, "Price_GBP_per_MWh"),
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def inputs_bundle() -> dict:
    loads = _read_csv(INPUTS / "load_profiles.csv")
    prices = _read_csv(INPUTS / "spot_prices.csv")
    shift = _read_csv(INPUTS / "shiftability_profile.csv", index_col="time_slot")

    out: dict = {"available": loads is not None}
    if loads is not None:
        n = len(loads)
        out["hours"] = _hours_axis(n)
        out["inflexible_load"] = _series(loads, "inflexible_load")
        out["flexible_load"] = _series(loads, "flexible_load")
    if prices is not None:
        out["spot_price"] = _series(prices, "spot_price")
    # The design tariff used by the model (deterministic time-of-use profile).
    out["tariff"] = _tariff_profile()
    if shift is not None:
        out["shiftability"] = {
            "tranches": [str(c) for c in shift.columns],
            "rows": [[_clean(v) for v in row] for row in shift.values.tolist()],
        }
    return out


def _tariff_profile() -> list:
    """The hard-coded design tariff from parameters_optimisation.generate_tariff,
    expanded to 96 15-minute slots (kept in sync with the model)."""
    hourly = [60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100,
              98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]
    return [hourly[h] for h in range(24) for _ in range(4)]


# ---------------------------------------------------------------------------
# Overview – headline KPIs across the whole study
# ---------------------------------------------------------------------------
def overview() -> dict:
    nom = nominal_timeseries()
    opt = optimisation_timeseries()
    flex = flex_heatmap()

    kpis: dict = {"availability": availability()}
    if nom.get("available"):
        kpis["nominal"] = nom["summary"]
    if opt.get("available"):
        kpis["optimisation"] = opt["summary"]
    if flex.get("available"):
        # Best achievable duration for a curtailment (load reduction) event.
        reductions = flex["reduction_magnitudes"]
        best_reduction = None
        if reductions and flex["z"]:
            mags = flex["magnitudes"]
            best = 0.0
            best_mag = None
            for i, mag in enumerate(mags):
                if mag < 0:
                    row = [d for d in flex["z"][i] if d is not None]
                    if row and max(row) > best:
                        best = max(row)
                        best_mag = mag
            best_reduction = {"magnitude_kw": best_mag, "duration_h": _clean(best)}
        kpis["flexibility"] = {
            "max_duration_h": flex["max_duration_h"],
            "max_reduction_kw": _clean(min(reductions)) if reductions else None,
            "max_increase_kw": _clean(max(flex["increase_magnitudes"]))
            if flex["increase_magnitudes"]
            else None,
            "best_reduction": best_reduction,
        }
    return kpis
