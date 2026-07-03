"""Load the market-data CSV pack into per-slot arrays for one trading day.

The MarketDay object is the single interface between data and model: swap
the synthetic CSVs for real ones (same schema, see generate_data.py) and
nothing downstream changes.

Slot convention matches Paper 1: arrays are indexed 1..108 (a zero is
prepended at index 0), where slots 1..96 are the trading day and 97..108
the 3-hour extension, priced at the following day's first periods.
"""
import functools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .market_parameters import PRODUCTS


@functools.lru_cache(maxsize=None)
def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _sp_to_slots(series_by_sp: dict, default=0.0) -> np.ndarray:
    """Expand a {settlement period: value} map to a slot array (index 0 unused)."""
    arr = np.full(config.N_SLOTS_EXT + 1, default, dtype=float)
    for h, v in series_by_sp.items():
        for t in config.sp_slots(h):
            arr[t] = v
    return arr


@dataclass
class MarketDay:
    date: object
    da_price: np.ndarray            # GBP/MWh, slots 0..108
    duos: np.ndarray                # GBP/MWh adder, slots 0..108
    bm_offer: np.ndarray            # GBP/MWh, slots 0..108
    bm_bid: np.ndarray              # GBP/MWh, slots 0..108
    avail_price: dict               # {product: {window: GBP/MW/h}}
    util_price: dict                # {product: GBP/MWh}
    phi: dict                       # {product: expected utilisation factor}
    dfs_events: list                # [(slots, direction, price)]
    dso_windows: list               # [(slots, avail_price, util_price)]
    cm_price: float                 # GBP/kW/yr
    cm_derating: float


def load_market_day(date) -> MarketDay:
    date = pd.Timestamp(date).date()
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).date()

    # --- day-ahead price incl. extension from the following day -------------
    da = _read("da_prices.csv")
    day = da[da["date"] == date].set_index("period")["price_gbp_mwh"]
    if day.empty:
        raise KeyError(f"No day-ahead data for {date}; run generate_data or "
                       f"extend the data pack.")
    nxt_day = da[da["date"] == nxt].set_index("period")["price_gbp_mwh"]
    if nxt_day.empty:
        nxt_day = day  # wrap if the pack ends here
    da_slot = np.zeros(config.N_SLOTS_EXT + 1)
    for t in range(1, config.N_SLOTS_DAY + 1):
        da_slot[t] = day[config.slot_to_sp(t)]
    for t in range(config.N_SLOTS_DAY + 1, config.N_SLOTS_EXT + 1):
        da_slot[t] = nxt_day[config.slot_to_sp(t - config.N_SLOTS_DAY)]

    # --- DUoS band adder -----------------------------------------------------
    duos_df = _read("duos_bands.csv").set_index("period")
    col = "weekday_gbp_mwh" if pd.Timestamp(date).dayofweek < 5 else "weekend_gbp_mwh"
    duos = np.zeros(config.N_SLOTS_EXT + 1)
    for t in range(1, config.N_SLOTS_EXT + 1):
        sp = config.slot_to_sp(t if t <= config.N_SLOTS_DAY
                               else t - config.N_SLOTS_DAY)
        duos[t] = duos_df.loc[sp, col]

    # --- BM prices (trading day only) ----------------------------------------
    bm = _read("bm_prices.csv")
    bm_day = bm[bm["date"] == date].set_index("period")
    bm_offer = _sp_to_slots(bm_day["offer_gbp_mwh"].to_dict())
    bm_bid = _sp_to_slots(bm_day["bid_gbp_mwh"].to_dict())

    # --- availability prices per product/window ------------------------------
    avail, util = {}, {}
    resp = _read("response_prices.csv")
    resp = resp[resp["date"] == date]
    for prod in ("DCL", "DCH", "DML", "DMH", "DRL", "DRH"):
        rows = resp[resp["product"] == prod]
        avail[prod] = dict(zip(rows["efa_block"], rows["avail_gbp_mw_h"]))
    resv = _read("reserve_prices.csv")
    resv = resv[resv["date"] == date]
    for prod in ("PQR", "NQR", "SR"):
        rows = resv[resv["product"] == prod]
        avail[prod] = dict(zip(rows["efa_block"], rows["avail_gbp_mw_h"]))
        util[prod] = float(rows["util_gbp_mwh"].mean()) if len(rows) else 0.0
    br = _read("br_prices.csv")
    br = br[br["date"] == date]
    for prod in ("BRpos", "BRneg"):
        rows = br[br["product"] == prod]
        avail[prod] = dict(zip(rows["period"], rows["avail_gbp_mw_h"]))

    # --- DFS events -----------------------------------------------------------
    dfs = _read("dfs_events.csv")
    dfs_events = []
    for _, r in dfs[dfs["date"] == date].iterrows():
        slots = [t for h in range(int(r["period_start"]),
                                  int(r["period_end"]) + 1)
                 for t in config.sp_slots(h)]
        dfs_events.append((slots, r["direction"], float(r["price_gbp_mwh"])))

    # --- DSO windows ------------------------------------------------------------
    dso = _read("dso_windows.csv")
    dso_windows = []
    for _, r in dso[dso["date"] == date].iterrows():
        slots = [t for h in range(int(r["period_start"]),
                                  int(r["period_end"]) + 1)
                 for t in config.sp_slots(h)]
        dso_windows.append((slots, float(r["avail_gbp_mw_h"]),
                            float(r["util_gbp_mwh"])))

    cm = _read("cm_params.csv").iloc[0]
    phi = _read("utilisation_factors.csv").set_index("product")["phi"].to_dict()

    return MarketDay(date=date, da_price=da_slot, duos=duos,
                     bm_offer=bm_offer, bm_bid=bm_bid,
                     avail_price=avail, util_price=util, phi=phi,
                     dfs_events=dfs_events, dso_windows=dso_windows,
                     cm_price=float(cm["price_gbp_kw_yr"]),
                     cm_derating=float(cm["derating"]))


def available_dates():
    """All dates in the data pack (excluding the final look-ahead day)."""
    d = sorted(_read("da_prices.csv")["date"].unique())
    return d[:-1]
