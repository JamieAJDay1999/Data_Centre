"""Generate synthetic GB market data in the exact schema the loaders expect.

Every file written here has the schema that the real data should be supplied
in later (doc 04 section 4 lists the real sources). Replace file-for-file and
the rest of the pipeline runs unchanged.

Files written to rev_stack/data/:

  da_prices.csv          date, period (1-48), price_gbp_mwh
  response_prices.csv    date, efa_block (1-6), product, avail_gbp_mw_h
  reserve_prices.csv     date, efa_block, product, avail_gbp_mw_h, util_gbp_mwh
  br_prices.csv          date, period (1-48), product, avail_gbp_mw_h
  bm_prices.csv          date, period, offer_gbp_mwh, bid_gbp_mwh
  dfs_events.csv         date, period_start, period_end, direction, price_gbp_mwh
  dso_windows.csv        date, period_start, period_end, avail_gbp_mw_h, util_gbp_mwh
  duos_bands.csv         period, weekday_gbp_mwh, weekend_gbp_mwh
  cm_params.csv          delivery_year, price_gbp_kw_yr, derating
  utilisation_factors.csv  product, phi

Price levels are calibrated loosely to the 2024-25 GB environment described
in rev_stack/01_gb_market_landscape.md: volatile wholesale with evening
peaks and occasional spikes/negatives, saturated (low single-digit pounds/MW/h)
frequency response, and modest reserve prices.

Run:  python -m rev_stack.model.generate_data
"""
import numpy as np
import pandas as pd

from . import config


def _daily_da_shape(rng, doy: int, weekday: int) -> np.ndarray:
    """Half-hourly day-ahead price shape for one day (48 values, GBP/MWh)."""
    sp = np.arange(48)
    hours = sp / 2.0
    season = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365.0)  # winter high
    base = 70.0 * season
    # Morning and evening peaks, deeper overnight trough.
    shape = (
        - 18.0 * np.cos(2 * np.pi * (hours - 3.0) / 24.0)
        + 14.0 * np.exp(-0.5 * ((hours - 8.0) / 1.5) ** 2)
        + 32.0 * np.exp(-0.5 * ((hours - 17.5) / 1.6) ** 2)
    )
    weekend_discount = 0.88 if weekday >= 5 else 1.0
    noise = rng.normal(0, 4.0, 48)
    price = (base + shape * season) * weekend_discount + noise
    # Occasional scarcity spike in the evening peak (winter-weighted).
    if rng.random() < 0.05 * season:
        price[33:39] += rng.uniform(80, 250)
    # Occasional negative midday prices in summer (solar surplus).
    if season < 0.9 and rng.random() < 0.08:
        price[22:30] -= rng.uniform(60, 110)
    return price


def generate(outdir=None, start=None, days=None, seed=None):
    outdir = outdir or config.DATA_DIR
    start = pd.Timestamp(start or config.SYNTH_START)
    days = days or config.SYNTH_DAYS
    rng = np.random.default_rng(seed if seed is not None else config.SYNTH_SEED)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, periods=days, freq="D")

    da_rows, resp_rows, resv_rows, br_rows, bm_rows = [], [], [], [], []
    dfs_rows, dso_rows = [], []

    # Saturated response prices: block-level lognormal, evening blocks dearer.
    resp_base = {"DCL": 3.0, "DCH": 1.2, "DML": 2.2, "DMH": 1.0,
                 "DRL": 3.8, "DRH": 1.8}
    resv_base = {"PQR": 2.5, "NQR": 1.4, "SR": 3.5}
    resv_util = {"PQR": 120.0, "NQR": 40.0, "SR": 110.0}
    block_mult = np.array([0.8, 0.7, 0.9, 1.0, 1.5, 1.1])  # EFA 1-6

    for d in dates:
        doy, wd = d.dayofyear, d.dayofweek
        season = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365.0)
        da = _daily_da_shape(rng, doy, wd)
        for h in range(48):
            da_rows.append((d.date(), h + 1, round(da[h], 2)))
            # BM: offers accepted above DA, bids below (turn-up paid little
            # or paying when system is long).
            off = da[h] + rng.uniform(15, 70)
            bid = da[h] - rng.uniform(30, 90)
            bm_rows.append((d.date(), h + 1, round(off, 2), round(bid, 2)))
            for prod in ("BRpos", "BRneg"):
                base = 2.8 if prod == "BRpos" else 1.3
                p = base * season * (1.6 if 33 <= h <= 38 else 1.0)
                p *= rng.lognormal(0, 0.45)
                br_rows.append((d.date(), h + 1, prod, round(p, 3)))
        for b in range(1, 7):
            for prod, base in resp_base.items():
                p = base * block_mult[b - 1] * season * rng.lognormal(0, 0.5)
                resp_rows.append((d.date(), b, prod, round(p, 3)))
            for prod, base in resv_base.items():
                p = base * block_mult[b - 1] * season * rng.lognormal(0, 0.5)
                resv_rows.append((d.date(), b, prod, round(p, 3),
                                  round(resv_util[prod] * rng.uniform(0.7, 1.4), 1)))
        # DFS: sparse events, winter-evening turn-down dominated; the evolved
        # (April 2026) design adds occasional turn-up events.
        if season > 1.1 and wd < 5 and rng.random() < 0.18:
            sp0 = rng.integers(34, 37)
            dfs_rows.append((d.date(), int(sp0), int(sp0 + rng.integers(1, 4)),
                             "down", round(rng.uniform(250, 900), 0)))
        if season < 0.9 and rng.random() < 0.05:
            sp0 = rng.integers(24, 28)
            dfs_rows.append((d.date(), int(sp0), int(sp0 + 2), "up",
                             round(rng.uniform(40, 120), 0)))
        # DSO: winter weekday evening availability windows (in-zone scenario).
        if season > 1.1 and wd < 5:
            dso_rows.append((d.date(), 33, 38,
                             round(4.0 * rng.uniform(0.7, 1.3), 2),
                             round(300.0 * rng.uniform(0.8, 1.2), 0)))

    pd.DataFrame(da_rows, columns=["date", "period", "price_gbp_mwh"]) \
        .to_csv(outdir / "da_prices.csv", index=False)
    pd.DataFrame(resp_rows, columns=["date", "efa_block", "product",
                                     "avail_gbp_mw_h"]) \
        .to_csv(outdir / "response_prices.csv", index=False)
    pd.DataFrame(resv_rows, columns=["date", "efa_block", "product",
                                     "avail_gbp_mw_h", "util_gbp_mwh"]) \
        .to_csv(outdir / "reserve_prices.csv", index=False)
    pd.DataFrame(br_rows, columns=["date", "period", "product",
                                   "avail_gbp_mw_h"]) \
        .to_csv(outdir / "br_prices.csv", index=False)
    pd.DataFrame(bm_rows, columns=["date", "period", "offer_gbp_mwh",
                                   "bid_gbp_mwh"]) \
        .to_csv(outdir / "bm_prices.csv", index=False)
    pd.DataFrame(dfs_rows, columns=["date", "period_start", "period_end",
                                    "direction", "price_gbp_mwh"]) \
        .to_csv(outdir / "dfs_events.csv", index=False)
    pd.DataFrame(dso_rows, columns=["date", "period_start", "period_end",
                                    "avail_gbp_mw_h", "util_gbp_mwh"]) \
        .to_csv(outdir / "dso_windows.csv", index=False)

    # DUoS red/amber/green residual adder (EHV-style, GBP/MWh).
    duos = []
    for h in range(1, 49):
        hour = (h - 1) / 2.0
        if 16 <= hour < 19:
            wk = 14.0          # red
        elif 8 <= hour < 16 or 19 <= hour < 22:
            wk = 2.5           # amber
        else:
            wk = 0.4           # green
        duos.append((h, wk, 0.4))
    pd.DataFrame(duos, columns=["period", "weekday_gbp_mwh",
                                "weekend_gbp_mwh"]) \
        .to_csv(outdir / "duos_bands.csv", index=False)

    pd.DataFrame([(2025, config.CM_PRICE_GBP_KW_YR, config.CM_DERATING)],
                 columns=["delivery_year", "price_gbp_kw_yr", "derating"]) \
        .to_csv(outdir / "cm_params.csv", index=False)

    # Expected utilisation factors phi_j (fraction of committed capacity-hours
    # actually called) - to be calibrated from 1 s frequency data / activation
    # statistics when real data arrives.
    phi = {"DCL": 0.02, "DCH": 0.02, "DML": 0.03, "DMH": 0.03,
           "DRL": 0.15, "DRH": 0.15, "BRpos": 0.05, "BRneg": 0.05,
           "PQR": 0.04, "NQR": 0.04, "SR": 0.06, "DSO": 0.10}
    pd.DataFrame(sorted(phi.items()), columns=["product", "phi"]) \
        .to_csv(outdir / "utilisation_factors.csv", index=False)

    print(f"Synthetic market data written to {outdir}  "
          f"({days} days from {start.date()})")


if __name__ == "__main__":
    config.ensure_dirs()
    generate()
