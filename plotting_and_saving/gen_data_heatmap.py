import os
import re
import glob
from pathlib import Path
import pandas as pd

# Folder containing your CSVs (change if needed)
"""BASE_DIR = "static/data/flexibility_outputs"

# Pattern for all detailed result files
PATTERN = os.path.join(BASE_DIR, "flex_duration_detailed_results_*.csv")

rows = []
for path in sorted(glob.glob(PATTERN)):
    fname = os.path.basename(path)

    # -- Parse timestep (tsXX) --
    m_ts = re.search(r"ts(\d+)", fname)
    if not m_ts:
        # Skip files that don't match the naming convention
        continue
    ts = int(m_ts.group(1))

    # -- Parse flex magnitude --
    # Accept either "flexneg100"  -> -100
    # or       "flex25" (ending before .csv) -> +25
    m_neg = re.search(r"flexneg(\d+)", fname)
    m_pos = re.search(r"flex(\d+)(?=\.csv$)", fname)
    if m_neg:
        flex_kw = -int(m_neg.group(1))
    elif m_pos:
        flex_kw = int(m_pos.group(1))
    else:
        # If neither is present, skip the file
        continue

    # -- Read file and count timesteps --
    df = pd.read_csv(path)
    n_rows = len(df)

    # -- Compute durations --
    # Each step = 15 minutes; subtract 12 steps as specified
    max_duration_min = max(0, (n_rows - 12) * 15)
    max_duration_hours = max_duration_min / 60.0

    # 'Timestep_Hours' follows the example: Timestep * 15 min
    timestep_hours = ts * 15 / 60.0

    rows.append({
        "Timestep": ts,
        "Flex_Magnitude_kW": float(flex_kw),
        "Max_Duration_Min": float(max_duration_min),
        "Timestep_Hours": float(timestep_hours),
        "Max_Duration_Hours": float(max_duration_hours),
    })

# Assemble results and save in the same column order as your example
result_cols = [
    "Timestep",
    "Flex_Magnitude_kW",
    "Max_Duration_Min",
    "Timestep_Hours",
    "Max_Duration_Hours",
]

result_df = pd.DataFrame(rows).sort_values(["Timestep", "Flex_Magnitude_kW"]).reset_index(drop=True)
out_path = os.path.join(BASE_DIR, "flex_duration_results_full.csv")
result_df[result_cols].to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(result_df.head())


"""

# Fill missing (Timestep, Flex_Magnitude_kW) combos with zeros in an
# existing flex_duration_results_full.csv, then overwrite it.

import pandas as pd

# ---- paths ----
PATH = (
    Path(__file__).resolve().parents[1]
    / "static"
    / "data"
    / "flexibility_outputs"
    / "flex_duration_results_full.csv"
)

# ---- load ----
df = pd.read_csv(PATH)

# Ensure required columns exist
required = ["Timestep", "Flex_Magnitude_kW", "Max_Duration_Min",
            "Timestep_Hours", "Max_Duration_Hours"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# ---- define the complete grid you want ----
# If you want to hard-code the set, uncomment the next two lines:
# TIMESTEPS = [1] + list(range(5, 100, 5))           # 1,5,10,...,95
# FLEX = [-100, -200, -300, -400, -500, 25, 50, 75]  # as per your spec

# Or infer the sets from what’s already in the file (safer if you change them later):
TIMESTEPS = sorted(set(df["Timestep"].tolist() + [1] + list(range(5, 100, 5))))
FLEX = sorted(set(df["Flex_Magnitude_kW"]).union({-100,-200,-300,-400,-500,25,50,75}))

# ---- build full grid ----
grid = (
    pd.MultiIndex
      .from_product([TIMESTEPS, FLEX], names=["Timestep","Flex_Magnitude_kW"])
      .to_frame(index=False)
)

# ---- merge and zero-fill ----
merged = grid.merge(
    df[required],
    on=["Timestep","Flex_Magnitude_kW"],
    how="left"
)

# Fill duration gaps with zeros
merged["Max_Duration_Min"] = merged["Max_Duration_Min"].fillna(0.0)
merged["Max_Duration_Hours"] = merged["Max_Duration_Hours"].fillna(0.0)

# Recompute Timestep_Hours when absent (15 min per step)
merged["Timestep_Hours"] = merged["Timestep_Hours"].fillna(merged["Timestep"] * 15 / 60.0)

# Optional: if duplicates existed originally, keep the max duration per combo
merged = (merged
          .groupby(["Timestep","Flex_Magnitude_kW"], as_index=False)
          .agg({
              "Max_Duration_Min":"max",
              "Timestep_Hours":"first",
              "Max_Duration_Hours":"max"
          }))

# ---- save (overwrite) ----
merged = merged.sort_values(["Timestep","Flex_Magnitude_kW"]).reset_index(drop=True)
merged.to_csv(PATH, index=False)

print(f"Completed zero-fill. Rows: {len(merged)}. Saved to {PATH}")
