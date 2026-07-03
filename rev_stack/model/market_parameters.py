"""Product definitions: the paper's Table III as code.

Each Product records the direction convention (from the provider's side:
'down' = demand turn-down = upward system flexibility), the sustain duration
tau_j the commitment must be deliverable for, the procurement granularity,
and the set of facility assets physically fast enough to back it (the
feasible asset-product set F of Eq. (15) in the paper).

Assets: 'IT' (workload deferral), 'UPS' (battery via grid-interactive UPS),
'CL' (cooling: chiller modulation backed by TES).

Prices are NOT stored here; they are time series loaded by market_data.py.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    name: str
    direction: str            # 'down' | 'up'
    tau_h: float              # required sustain duration (hours)
    granularity: str          # 'efa' | 'sp' | 'window'
    assets: tuple             # eligible backing assets
    price_file: str           # which CSV carries its availability price
    util_priced: bool = False # has a utilisation (£/MWh) component
    family: str = "response"


# --- NESO dynamic response (EAC, EFA-block granularity, min 1 MW) -----------
DCL = Product("DCL", "down", 0.25, "efa", ("UPS",), "response_prices.csv")
DCH = Product("DCH", "up",   0.25, "efa", ("UPS",), "response_prices.csv")
DML = Product("DML", "down", 0.50, "efa", ("UPS",), "response_prices.csv")
DMH = Product("DMH", "up",   0.50, "efa", ("UPS",), "response_prices.csv")
DRL = Product("DRL", "down", 1.00, "efa", ("UPS",), "response_prices.csv")
DRH = Product("DRH", "up",   1.00, "efa", ("UPS",), "response_prices.csv")

# --- New reserve family ------------------------------------------------------
# Balancing Reserve: settlement-period granularity, delivered via the BM.
BR_POS = Product("BRpos", "down", 2.0, "sp", ("IT", "UPS", "CL"),
                 "br_prices.csv", family="reserve")
BR_NEG = Product("BRneg", "up", 2.0, "sp", ("UPS", "CL"),
                 "br_prices.csv", family="reserve")
# Quick Reserve: full delivery within 1 minute -> UPS (and CL valves) only.
PQR = Product("PQR", "down", 0.25, "efa", ("UPS",),
              "reserve_prices.csv", util_priced=True, family="reserve")
NQR = Product("NQR", "up", 0.25, "efa", ("UPS", "CL"),
              "reserve_prices.csv", util_priced=True, family="reserve")
# Slow Reserve: 15-min delivery, hour-scale sustain -> the IT product.
SR = Product("SR", "down", 2.0, "efa", ("IT", "UPS", "CL"),
             "reserve_prices.csv", util_priced=True, family="reserve")

# --- DSO flexibility (availability windows from dso_windows.csv) -------------
DSO = Product("DSO", "down", 2.0, "window", ("IT", "UPS", "CL"),
              "dso_windows.csv", util_priced=True, family="dso")

PRODUCTS = {p.name: p for p in
            [DCL, DCH, DML, DMH, DRL, DRH, BR_POS, BR_NEG, PQR, NQR, SR, DSO]}

DOWN_PRODUCTS = [p.name for p in PRODUCTS.values() if p.direction == "down"]
UP_PRODUCTS = [p.name for p in PRODUCTS.values() if p.direction == "up"]
ASSETS = ("IT", "UPS", "CL")

# Largest sustain duration among IT-eligible turn-down products. The IT
# utilisation floor (Eq. (16)) is built at this single, most conservative
# duration level in v1: only workload with at least this much remaining
# deferral slack is credited as IT headroom for ANY product.
TAU_IT_MAX_H = max(p.tau_h for p in PRODUCTS.values()
                   if "IT" in p.assets and p.direction == "down")


def slots_for_window(product: Product, w: int):
    """15-min slots covered by procurement window w of a product."""
    from . import config
    if product.granularity == "efa":
        return config.efa_slots(w)
    if product.granularity == "sp":
        return config.sp_slots(w)
    raise ValueError(f"window slots for {product.name} come from market data")


def n_windows(product: Product) -> int:
    from . import config
    if product.granularity == "efa":
        return config.N_EFA
    if product.granularity == "sp":
        return config.N_SP
    return 0  # 'window' products: window count is data-driven
