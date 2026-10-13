"""Build assets/prices_2025.npy: GB day-ahead reference price (GBP/MWh) as a
96 x 365 grid of local (Europe/London) quarter-hour slots by local day.

Source series: LCCC Intermittent Market Reference Price, 2025, as stored in
paper/data/price_timeline.csv. On clock-change days the local slots are
filled by nearest available quarter-hour.
"""
import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parents[1] / "paper" / "data" / "price_timeline.csv"
LON = ZoneInfo("Europe/London")

prices = {}
with open(SRC, newline="") as f:
    for row in csv.DictReader(f):
        ts = datetime.fromisoformat(row["timestamp_utc"])
        prices[ts] = float(row["settlement_price_gbp_per_mwh"])

grid = np.full((96, 365), np.nan, np.float32)
start = datetime(2025, 1, 1)
for d in range(365):
    day = start + timedelta(days=d)
    for s in range(96):
        local = datetime(day.year, day.month, day.day, tzinfo=LON) + timedelta(minutes=15 * s)
        utc = local.astimezone(ZoneInfo("UTC"))
        v = prices.get(utc)
        if v is None:
            v = prices.get(utc - timedelta(hours=1), prices.get(utc + timedelta(hours=1)))
        grid[s, d] = v
assert not np.isnan(grid).any()
out = ROOT / "assets" / "prices_2025.npy"
np.save(out, grid)
print(out, grid.shape, f"min {grid.min():.2f} median {np.median(grid):.2f} max {grid.max():.2f}",
      f"negative cells {(grid < 0).sum()}")
