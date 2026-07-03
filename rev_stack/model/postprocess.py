"""Extract results from a solved stacking model into tidy DataFrames."""
import pandas as pd
import pyomo.environ as pyo

from . import config
from .market_parameters import PRODUCTS


def revenue_summary(m) -> dict:
    """One row of headline financials for the day (GBP)."""
    return {
        "cost_energy": pyo.value(m.cost_energy),
        "rev_availability": pyo.value(m.rev_availability),
        "rev_utilisation": pyo.value(m.rev_utilisation),
        "rev_bm": pyo.value(m.rev_bm),
        "rev_dfs": pyo.value(m.rev_dfs),
        "rev_cm": pyo.value(m.rev_cm),
        "net_cost": pyo.value(m.objective),
    }


def commitments_frame(m) -> pd.DataFrame:
    """Commitments and per-asset allocation, one row per (product, window)."""
    rows = []
    for (j, w) in m.R_IDX:
        p = m._products[j]
        r = pyo.value(m.r[j, w])
        if r < 1e-6:
            continue
        row = {"product": j, "window": w, "direction": p.direction,
               "granularity": p.granularity, "tau_h": p.tau_h,
               "committed_kw": r}
        for a in ("IT", "UPS", "CL"):
            row[f"alloc_{a.lower()}_kw"] = (
                pyo.value(m.r_alloc[j, w, a]) if (j, w, a) in m.ALLOC_IDX else 0.0)
        rows.append(row)
    cols = ["product", "window", "direction", "granularity", "tau_h",
            "committed_kw", "alloc_it_kw", "alloc_ups_kw", "alloc_cl_kw"]
    return pd.DataFrame(rows, columns=cols)


def availability_revenue_by_product(m) -> pd.DataFrame:
    """Availability + expected utilisation revenue per product (GBP/day)."""
    dt = 0.25
    mkt = m._mkt
    rows = {}
    for (j, w) in m.R_IDX:
        p = m._products[j]
        r = pyo.value(m.r[j, w])
        if p.granularity == "window":
            slots, ap, up = mkt.dso_windows[w - 1]
            hours, price, uprice = len(slots) * dt, ap, up
        else:
            hours = (config.SLOTS_PER_EFA if p.granularity == "efa"
                     else config.SLOTS_PER_SP) * dt
            price = mkt.avail_price[j][w]
            uprice = mkt.util_price.get(j, 0.0)
        d = rows.setdefault(j, {"product": j, "availability_gbp": 0.0,
                                "utilisation_gbp": 0.0, "avg_committed_kw": 0.0,
                                "n_windows": 0})
        d["availability_gbp"] += price * r * hours / 1000.0
        d["utilisation_gbp"] += mkt.phi.get(j, 0.0) * uprice * r * hours / 1000.0
        d["avg_committed_kw"] += r
        d["n_windows"] += 1
    for d in rows.values():
        if d["n_windows"]:
            d["avg_committed_kw"] /= d["n_windows"]
    return pd.DataFrame(list(rows.values()))


def dispatch_frame(m, params) -> pd.DataFrame:
    """Slot-level physical schedule and market positions."""
    T = list(m.TEXT_SLOTS)
    df = pd.DataFrame({
        "slot": T,
        "P_grid_kw": [pyo.value(m.P_grid[t]) for t in T],
        "P_it_kw": [pyo.value(m.p_it_total_kw[t]) for t in T],
        "P_grid_it_kw": [pyo.value(m.p_grid_it_kw[t]) for t in T],
        "P_chiller_kw": [(pyo.value(m.p_chiller_hvac_w[t])
                          + pyo.value(m.p_chiller_tes_w[t])) / 1000.0 for t in T],
        "P_ups_ch_kw": [pyo.value(m.p_ups_ch_kw[t]) for t in T],
        "P_ups_disch_kw": [pyo.value(m.p_ups_disch_kw[t]) for t in T],
        "E_ups_kwh": [pyo.value(m.e_ups_kwh[t]) for t in T],
        "E_tes_kwh": [pyo.value(m.e_tes_kwh[t]) for t in T],
        "total_cpu": [pyo.value(m.total_cpu[t]) for t in T],
        "da_price": [m._mkt.da_price[t] for t in T],
    })
    T96 = set(range(1, config.N_SLOTS_DAY + 1))
    df["x_bm_off_kw"] = [pyo.value(m.x_bm_off[t]) if t in T96 else 0.0 for t in T]
    df["x_bm_bid_kw"] = [pyo.value(m.x_bm_bid[t]) if t in T96 else 0.0 for t in T]
    df["committed_down_kw"] = [
        pyo.value(m._committed_at(t, "down")) if t in T96 else 0.0 for t in T]
    df["committed_up_kw"] = [
        pyo.value(m._committed_at(t, "up")) if t in T96 else 0.0 for t in T]
    return df


def save_day_results(outdir, m, params, summary_extra=None):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = revenue_summary(m)
    if summary_extra:
        summary.update(summary_extra)
    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)
    commitments_frame(m).to_csv(outdir / "commitments.csv", index=False)
    availability_revenue_by_product(m).to_csv(outdir / "revenue_by_product.csv",
                                              index=False)
    dispatch_frame(m, params).to_csv(outdir / "dispatch.csv", index=False)
    return summary
