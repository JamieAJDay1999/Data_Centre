import json
from pathlib import Path

root = Path("static/data/rolling_year_outputs")
scenarios = [
    ("2025_baseline_reformulated", "Baseline"),
    ("2025_optimised_cohort_trace", "Central Case"),
    ("2025_flex_0p5", "Flex 0.5x"),
    ("2025_flex_0p75", "Flex 0.75x"),
    ("2025_flex_1p25", "Flex 1.25x"),
    ("2025_flex_1p5", "Flex 1.5x"),
    ("2025_ups_0p5", "UPS 0.5x"),
    ("2025_ups_0p75", "UPS 0.75x"),
    ("2025_ups_1p25", "UPS 1.25x"),
    ("2025_ups_1p5", "UPS 1.5x"),
    ("2025_tes_0p5", "TES 0.5x"),
    ("2025_tes_0p75", "TES 0.75x"),
    ("2025_tes_1p25", "TES 1.25x"),
    ("2025_tes_1p5", "TES 1.5x"),
]

print("=== ANNUAL OPTIMISATION SCENARIOS ===")
print(f"{'Scenario':<24} | {'Status':<11} | {'Progress':<12} | {'Cost / Details'}")
print("-" * 75)

for sc, desc in scenarios:
    p = root / sc
    if not p.exists():
        print(f"{desc:<24} | {'MISSING':<11} | 0/365 days   | -")
        continue
    summary = p / "annual_summary.json"
    cps_dir = p / "checkpoints"
    cps = sorted(cps_dir.glob("*.json")) if cps_dir.exists() else []
    day_count = len(cps)
    latest = cps[-1].stem if cps else "none"

    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            cost = data.get("settlement_cost_gbp", 0.0)
            print(f"{desc:<24} | {'COMPLETE':<11} | 365/365 days | GBP {cost:,.2f}")
        except Exception:
            print(f"{desc:<24} | {'COMPLETE':<11} | 365/365 days | Done")
    else:
        print(f"{desc:<24} | {'RUNNING':<11} | {day_count:>3}/365 days | Latest: {latest}")

# Flexibility progress
flex_dir = Path("reports/representative_day_flexibility")
flex_cells = [f for f in flex_dir.glob("*.json") if f.name != "summary.json"] if flex_dir.exists() else []
print("\n=== REPRESENTATIVE-DAY FLEXIBILITY SWEEP ===")
print(f"Completed Cells: {len(flex_cells)} / 288 ({len(flex_cells)/288:.1%})")
