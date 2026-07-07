"""
sensitivity_sweep.py — one-at-a-time (OAT) sensitivity analysis wrapper.

Implements paper/sensitivity_analysis_plan.md (reviewer comment R2-3):

  Tier 1 (--tier 1): cost-saving sensitivity. For each parameter case, re-run
      Scenario 1 (nominal) and Scenario 2 (cost optimisation) in an isolated
      case directory under static/data/sensitivity_outputs/<case_id>/ and
      record the cost saving plus its component breakdown.

  Tier 2 (--tier 2): duration sensitivity. For each case, re-use the case's
      Scenario 2 baseline produced by Tier 1 and binary-search the maximum
      feasible duration tau for three reference flexibility products
      (Scenario 3 probes). Cold-aisle temperature cases re-use the *base*
      baseline and only relax the temperature bound inside the Scenario 3
      model, matching the paper's framing of the 18-23 C flexibility range.

Nothing in nominal_calculation.py / optimisation.py / flexibility_duration.py
is modified. This wrapper redirects their module-level path constants into the
per-case directories and monkey-patches generate_tariff for representative
price-day cases. Other scalar parameters remain one-at-a-time multiplier
sweeps.

The nominal case needs no special handling for the TES/UPS cases:
nominal_calculation.configure_nominal_params() zeroes the TES charge/discharge
limits and the nominal power balance excludes UPS discharge, so the base cost
only depends on the price cases (verified via the stored storage-use columns).

Usage (from the repo root):
    python sensitivity_sweep.py --tier 1
    python sensitivity_sweep.py --tier 2
    python sensitivity_sweep.py --tier 2 --jobs 4      # parallel across cases
    python sensitivity_sweep.py --tier 1 --dry-run     # list cases, run nothing
    python sensitivity_sweep.py --tier 1 --only tes    # case_id substring filter

Then:
    python plot_sensitivity_results.py
"""
import matplotlib
matplotlib.use("Agg")  # must precede pyplot use in the model modules (they call plt.show())

import argparse
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)  # the model modules create/use paths relative to the repo root

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import nominal_calculation as nom
import optimisation as opt
import flexibility_duration as fd
from inputs.parameters_optimisation import ModelParameters
from inputs.parameters_optimisation import (
    DEFAULT_PRICE_SCENARIO_FILE,
    generate_tariff_for_price_scenario,
    load_price_scenarios,
    price_scenario_summary,
)

ORIG_INPUTS = ROOT / "static" / "data" / "inputs"
SWEEP_ROOT = ROOT / "static" / "data" / "sensitivity_outputs"

# --- Case definitions ---------------------------------------------------------
MULTIPLIER_LEVELS = [0.5, 0.75, 1.25, 1.5]   # base = 1.0
UPS_LEVELS = [0.5, 1.5]                      # optional 5th parameter, 2 levels
TEMP_LEVELS_C = [24.0, 25.0]                 # base = 23 C, Tier 2 only
BASE_PRICE_SCENARIO = "baseline_synthetic"
PRICE_SCENARIO_IDS = [
    sid for sid in load_price_scenarios(DEFAULT_PRICE_SCENARIO_FILE)
    if sid != BASE_PRICE_SCENARIO
]

BASE_CASE = {
    "case_id": "base",
    "param": "base",
    "level": 1.0,
    "price_scenario": BASE_PRICE_SCENARIO,
}


def price_day_case_id(scenario_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(scenario_id)).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"price_day_{safe or 'scenario'}"


def tier1_cases():
    cases = [BASE_CASE]
    for param, levels in (
        ("flex_share", MULTIPLIER_LEVELS),
        ("tes_capacity", MULTIPLIER_LEVELS),
        ("ups_capacity", UPS_LEVELS),
    ):
        cases += [{"case_id": f"{param}_x{lv:g}", "param": param, "level": lv} for lv in levels]
    cases += [
        {
            "case_id": price_day_case_id(scenario_id),
            "param": "price_day",
            "level": i,
            "price_scenario": scenario_id,
        }
        for i, scenario_id in enumerate(PRICE_SCENARIO_IDS, start=1)
    ]
    return cases


def tier2_cases():
    return tier1_cases() + [
        {"case_id": f"cold_aisle_{lv:g}C", "param": "cold_aisle_temp", "level": lv}
        for lv in TEMP_LEVELS_C
    ]


# Scenario 3 reference products. Slot ts covers the interval ending at ts*15 min,
# so ts=1 -> 00:15, ts=70 -> 17:30, ts=72 -> 18:00. P1/P2 are the exact numbers
# quoted in the abstract and conclusion; P3 is the downward-flexibility probe.
PROBES = [
    {"probe": "P1", "t0_slot": 1,  "dP_kw": -100.0},  # easy upward,  base tau ~ 6.8 h
    {"probe": "P2", "t0_slot": 70, "dP_kw": -100.0},  # hard upward,  base tau ~ 0.2 h
    {"probe": "P3", "t0_slot": 72, "dP_kw": 100.0},   # downward
]

# Regression anchors: the published base-case numbers the multiplier-1.0 case
# must reproduce before any sweep result is trusted.
EXPECTED_BASE = {
    "base_cost_gbp": 1659.54,
    "opt_cost_gbp": 1493.19,
    "saving_pct": 10.02,
    "P1_tau_hours": 6.8,
    "P2_tau_hours": 0.2,
}
GATE_TOL = {"cost_gbp": 2.0, "saving_pp": 0.25, "tau_hours": 0.30}

# --- Per-case environment setup ------------------------------------------------
def case_dirs(case_id: str) -> dict:
    root = SWEEP_ROOT / case_id
    return {
        "root": root,
        "inputs": root / "inputs",
        "nominal": root / "nominal_outputs",
        "opt": root / "optimisation_outputs",
        "flex": root / "flexibility_outputs",
        "images": root / "images",
    }


def redirect_paths(d: dict):
    """Point the three model modules' module-level path constants at the case
    directory so the canonical static/data outputs are never touched."""
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    nom.DATA_DIR_INPUTS = d["inputs"]
    nom.DATA_DIR_OUTPUTS = d["nominal"]
    nom.IMAGE_DIR = d["images"]
    opt.DATA_DIR_INPUTS_1 = d["inputs"]
    opt.DATA_DIR_INPUTS_2 = d["nominal"]
    opt.DATA_DIR_OUTPUTS = d["opt"]
    opt.IMAGE_DIR = d["images"]
    fd.DATA_DIR_INPUTS = d["opt"]
    fd.DATA_DIR_OUTPUTS = d["flex"]
    fd.IMAGE_DIR = d["images"]


def install_price_scenario(scenario: str):
    """Monkey-patch generate_tariff in all three modules for a named price day.

    The index-0 dummy zero is preserved and the selected 24-hour price profile
    is tiled as needed across the extended horizon.
    """
    def scenario_tariff(num_steps, dt_seconds, _scenario=scenario):
        return generate_tariff_for_price_scenario(
            _scenario, num_steps, dt_seconds, DEFAULT_PRICE_SCENARIO_FILE
        )
    nom.generate_tariff = scenario_tariff
    opt.generate_tariff = scenario_tariff
    fd.generate_tariff = scenario_tariff


def case_price_scenario(case: dict) -> str:
    return case.get("price_scenario", BASE_PRICE_SCENARIO)


def write_case_inputs(case: dict, d: dict) -> dict:
    """Write the case's input CSVs and return input-side diagnostics.

    For flex_share cases, load is *moved between* the flexible and inflexible
    pools rather than scaling the flexible column unconditionally. The transfer
    is (m-1)*flex but is capped at the inflexible pool available in each slot,
    so the per-slot total is preserved exactly (isolating flexibility *share*
    from demand level) and neither pool can go negative for any m >= 0. This
    reduces to new_flex = m*flex exactly for m < 1; for m > 1 the flexible
    share saturates at the physical 100% limit in slots that are already
    almost fully flexible, so the realised aggregate multiplier can fall below
    the nominal one (recorded here for transparent reporting)."""
    lp = pd.read_csv(ORIG_INPUTS / "load_profiles.csv", index_col="time_slot")
    diag = {}
    if case["param"] == "flex_share":
        m = case["level"]
        flex = lp["flexible_load"].astype(float)
        inflex = lp["inflexible_load"].astype(float)
        transfer = np.minimum((m - 1.0) * flex, inflex)   # inflex -> flex; < 0 when m < 1
        new_flex = flex + transfer
        new_inflex = inflex - transfer
        assert (new_flex >= -1e-9).all() and (new_inflex >= -1e-9).all(), "pool went negative"
        drift = float(((new_inflex + new_flex) - (inflex + flex)).abs().max())
        assert drift < 1e-9, f"total utilisation not preserved (drift {drift})"
        diag = {
            "nominal_flex_multiplier": m,
            "realized_flex_multiplier": round(float(new_flex.sum() / flex.sum()), 4),
            "base_flex_share": round(float(flex.sum() / (flex.sum() + inflex.sum())), 4),
            "realized_flex_share": round(float(new_flex.sum() / (new_flex.sum() + new_inflex.sum())), 4),
            "saturated_slots": int(((m - 1.0) * flex > inflex + 1e-9).sum()) if m > 1 else 0,
        }
        lp = pd.DataFrame(
            {"inflexible_load": new_inflex.clip(lower=0.0),
             "flexible_load": new_flex.clip(lower=0.0)},
            index=lp.index,
        )
        lp.index.name = "time_slot"
    lp.to_csv(d["inputs"] / "load_profiles.csv")
    shutil.copy(ORIG_INPUTS / "shiftability_profile.csv", d["inputs"] / "shiftability_profile.csv")
    return diag


def build_params(case: dict) -> ModelParameters:
    """Construct ModelParameters with the case override applied, re-deriving
    every attribute that ModelParameters computes at construction time."""
    params = ModelParameters()
    param, lv = case["param"], case["level"]
    if param == "tes_capacity":
        params.TES_kwh_cap = params.TES_kwh_cap * lv
        params.TES_capacity_kWh = params.TES_kwh_cap
        params.TES_initial_charge_kWh = 0.5 * params.TES_kwh_cap
        # Charge/discharge power limits deliberately unchanged: this sweeps the
        # sizing of the *energy* buffer only (stated in the paper text).
    elif param == "ups_capacity":
        params.e_nom_kwh = params.e_nom_kwh * lv
        params.e_start_kwh = params.e_nom_kwh          # UPS starts full
        params.e_min_kwh = params.soc_min * params.e_nom_kwh
        params.e_max_kwh = params.soc_max * params.e_nom_kwh
    elif param == "cold_aisle_temp":
        params.T_cAisle_upper_limit_Celsius = lv       # absolute deg C, not a multiplier
    # flex_share and price_day act through the inputs/tariff, not params.
    return params


# --- Tier 1: cost-saving sensitivity -------------------------------------------
def replicate_print_summary(results_df: pd.DataFrame) -> dict:
    """Calculates cost summary, using the corrected sum over the 12 extension slots."""
    opt_step = results_df["Optimized_Cost_per_Step"]
    nom_step = results_df["Nominal_Cost"]
    ext_diff_sum = float((opt_step.iloc[96:108] - nom_step.iloc[96:108]).sum())

    base_cost = float(nom_step.iloc[:96].sum())
    opt_cost = float(opt_step.iloc[:96].sum()) + ext_diff_sum
    saving = base_cost - opt_cost
    
    return {
        "base_cost_gbp": round(base_cost, 2),
        "opt_cost_gbp": round(opt_cost, 2),
        "saving_gbp": round(saving, 2),
        "saving_pct": round(100.0 * saving / base_cost, 3) if base_cost > 0 else 0.0,
        "opt_cost_ext_sum_gbp": round(opt_cost, 2),
        "saving_pct_ext_sum": round(100.0 * saving / base_cost, 3) if base_cost > 0 else 0.0,
    }


def component_costs(results_df: pd.DataFrame, nom_df: pd.DataFrame, dt_hours: float = 0.25) -> dict:
    """Grid-cost decomposition by component over the 96 core slots, for the
    optimised and base cases (feeds the R2-4 cost-composition response)."""
    out = {}
    components = [
        ("it", "P_Grid_IT_kW"),
        ("cooling", "P_Grid_Cooling_kW"),
        ("ups_charge", "P_UPS_Charge_kW"),
        ("other", "P_Grid_Other_kW"),
    ]
    for tag, df in (("opt", results_df), ("base", nom_df)):
        price = df["Price_GBP_per_MWh"].iloc[:96] / 1000.0
        for label, col in components:
            out[f"{tag}_cost_{label}_gbp"] = round(float((df[col].iloc[:96] * price * dt_hours).sum()), 2)
    return out


def run_tier1_case(case: dict) -> dict:
    """Scenario 1 + Scenario 2 for one case, in its own directory tree."""
    t_start = time.time()
    d = case_dirs(case["case_id"])
    redirect_paths(d)
    scenario = case_price_scenario(case)
    install_price_scenario(scenario)
    input_diag = write_case_inputs(case, d)
    input_diag.update(price_scenario_summary(scenario, DEFAULT_PRICE_SCENARIO_FILE))

    # Scenario 1. configure_nominal_params() pins TES off and the nominal power
    # balance excludes UPS discharge, so the base case never uses storage.
    nom.run_nominal_case_generation(include_charts=False)
    nom_df = pd.read_csv(d["nominal"] / "nominal_case_results.csv")
    storage_use = {
        "nominal_max_ups_kw": round(float(max(
            nom_df["P_UPS_Charge_kW"].abs().max(), nom_df["P_UPS_Discharge_kW"].abs().max())), 6),
        "nominal_max_tes_w": round(float(max(
            nom_df["Q_Charge_TES_Watts"].abs().max(), nom_df["Q_Discharge_TES_Watts"].abs().max())), 6),
    }
    if storage_use["nominal_max_ups_kw"] > 1.0 or storage_use["nominal_max_tes_w"] > 1000.0:
        print(f"⚠️  Nominal case used storage in {case['case_id']} — base cost may be "
              f"parameter-dependent: {storage_use}")

    # Scenario 2
    params = build_params(case)
    data = opt.load_and_prepare_data(params)
    total_cost, results_df, _flex_origin = opt.run_single_optimization(params, data, msg=False)
    if total_cost is None:
        raise RuntimeError(f"Scenario 2 solver failed for case {case['case_id']}")
    # run_single_optimization also wrote load_profiles_opt.csv and
    # shiftability_profile_opt.csv into d['opt']; the baseline CSV completes
    # the Scenario 3 input set for Tier 2.
    results_df.to_csv(d["opt"] / "optimised_baseline.csv", index=False, float_format="%.4f")

    row = dict(case)
    row.update(replicate_print_summary(results_df))
    row.update(component_costs(results_df, nom_df))
    row.update(storage_use)
    row.update(input_diag)
    row["runtime_s"] = round(time.time() - t_start, 1)
    with open(d["root"] / "tier1_result.json", "w") as f:
        json.dump(row, f, indent=2)
    print(f"✅ Tier 1 {case['case_id']}: saving {row['saving_pct']:.2f}% "
          f"(base {row['base_cost_gbp']:.2f} GBP) in {row['runtime_s']}s")
    return row


# --- Tier 2: duration probes ----------------------------------------------------
def slot_clock(slot: int) -> str:
    minutes = slot * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def run_tier2_case(case: dict) -> list:
    """Three Scenario 3 probes for one case. Parameter cases read their own
    Tier 1 baseline; cold-aisle cases read the base baseline and only relax
    the temperature bound inside the Scenario 3 model."""
    t_start = time.time()
    baseline_case = "base" if case["param"] == "cold_aisle_temp" else case["case_id"]
    d = case_dirs(baseline_case)
    if not (d["opt"] / "optimised_baseline.csv").exists():
        raise RuntimeError(
            f"Missing Scenario 2 baseline for '{baseline_case}' "
            f"({d['opt'] / 'optimised_baseline.csv'}). Run --tier 1 first."
        )
    redirect_paths(d)
    install_price_scenario(case_price_scenario(case))

    params = build_params(case)
    # flexibility_duration.main() re-expresses the post-Scenario-2 shiftability
    # in 12 one-step tranches before building models; replicate that here.
    params.tranche_max_delay = {x: x for x in range(1, 13)}
    params.K_TRANCHES = range(1, 13)
    params.T_SLOTS = range(1, 97)

    baseline_df = pd.read_csv(d["opt"] / "optimised_baseline.csv")
    if "Time_Slot_EXT" in baseline_df.columns:
        baseline_df = baseline_df.set_index("Time_Slot_EXT")
    data = fd.load_and_prepare_data(params)

    rows = []
    for pr in PROBES:
        t_probe = time.time()
        # No result banking: every probe is solved fresh for this case.
        steps = fd.find_max_duration(
            params, data, baseline_df, pr["t0_slot"], pr["dP_kw"], search_type="binary"
        )
        rows.append({
            "case_id": case["case_id"],
            "param": case["param"],
            "level": case["level"],
            "price_scenario": case_price_scenario(case),
            "probe": pr["probe"],
            "t0_slot": pr["t0_slot"],
            "t0_clock": slot_clock(pr["t0_slot"]),
            "dP_kw": pr["dP_kw"],
            "tau_steps": int(steps),
            "tau_hours": round(steps * params.dt_seconds / 3600.0, 2),
            "search_s": round(time.time() - t_probe, 1),
        })

    out_dir = case_dirs(case["case_id"])["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tier2_result.json", "w") as f:
        json.dump(rows, f, indent=2)
    taus = ", ".join(f"{r['probe']}={r['tau_hours']:g}h" for r in rows)
    print(f"✅ Tier 2 {case['case_id']}: {taus} in {round(time.time() - t_start, 1)}s")
    return rows


# --- Aggregation, gating, orchestration ----------------------------------------
def rebuild_tier1_csv() -> pathlib.Path:
    rows = []
    for f in sorted(SWEEP_ROOT.glob("*/tier1_result.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    out = SWEEP_ROOT / "tier1_cost_sensitivity.csv"
    if rows:
        pd.DataFrame(rows).sort_values(["param", "level"]).to_csv(out, index=False)
    return out


def rebuild_tier2_csv() -> pathlib.Path:
    rows = []
    for f in sorted(SWEEP_ROOT.glob("*/tier2_result.json")):
        with open(f) as fh:
            rows.extend(json.load(fh))
    out = SWEEP_ROOT / "tier2_duration_sensitivity.csv"
    if rows:
        pd.DataFrame(rows).sort_values(["param", "level", "probe"]).to_csv(out, index=False)
    return out


def _gate_check(label, actual, expected, tol):
    ok = abs(actual - expected) <= tol
    print(f"   {'PASS' if ok else 'FAIL'}  {label}: got {actual:.3f}, expected "
          f"{expected:.3f} (±{tol})")
    return ok


def gate_tier1(base_row: dict, no_gate: bool):
    print("\n--- Regression gate (Tier 1, base case) ---")
    ok = all([
        _gate_check("base cost (GBP)", base_row["base_cost_gbp"], EXPECTED_BASE["base_cost_gbp"], GATE_TOL["cost_gbp"]),
        _gate_check("optimised cost (GBP)", base_row["opt_cost_gbp"], EXPECTED_BASE["opt_cost_gbp"], GATE_TOL["cost_gbp"]),
        _gate_check("saving (%)", base_row["saving_pct"], EXPECTED_BASE["saving_pct"], GATE_TOL["saving_pp"]),
    ])
    if not ok and not no_gate:
        sys.exit("❌ Base case does not reproduce the published numbers. Fix this "
                 "before sweeping (or rerun with --no-gate to proceed anyway).")
    if not ok:
        print("⚠️  Gate failed but --no-gate set; continuing.")


def gate_tier2(base_rows: list, no_gate: bool):
    print("\n--- Regression gate (Tier 2, base-case probes) ---")
    by_probe = {r["probe"]: r for r in base_rows}
    ok = all([
        _gate_check("P1 tau (h)", by_probe["P1"]["tau_hours"], EXPECTED_BASE["P1_tau_hours"], GATE_TOL["tau_hours"]),
        _gate_check("P2 tau (h)", by_probe["P2"]["tau_hours"], EXPECTED_BASE["P2_tau_hours"], GATE_TOL["tau_hours"]),
    ])
    print(f"   INFO  P3 tau (h): {by_probe['P3']['tau_hours']:.2f} (downward probe, "
          f"recorded as the base anchor)")
    if not ok and not no_gate:
        sys.exit("❌ Base-case probes do not reproduce the published durations. "
                 "Fix this before sweeping (or rerun with --no-gate).")
    if not ok:
        print("⚠️  Gate failed but --no-gate set; continuing.")


def run_cases(cases, worker, jobs):
    """Run non-base cases, optionally in parallel. Each case is independent;
    a failure is recorded and does not abort the sweep."""
    errors = {}
    if jobs <= 1:
        for case in cases:
            try:
                worker(case)
            except Exception:
                errors[case["case_id"]] = traceback.format_exc()
                print(f"❌ {case['case_id']} failed:\n{errors[case['case_id']]}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(worker, case): case for case in cases}
            for fut in as_completed(futures):
                case = futures[fut]
                try:
                    fut.result()
                except Exception:
                    errors[case["case_id"]] = traceback.format_exc()
                    print(f"❌ {case['case_id']} failed:\n{errors[case['case_id']]}")
    return errors


def format_case_level(case: dict) -> str:
    if case["param"] == "price_day":
        return str(case["price_scenario"])
    level = case["level"]
    return f"{level:g}" if isinstance(level, (int, float)) else str(level)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=[1, 2], required=True)
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on case_id (the base case always runs)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes for the non-base cases (default 1)")
    ap.add_argument("--no-gate", action="store_true",
                    help="continue even if the base case fails the regression gate")
    ap.add_argument("--dry-run", action="store_true", help="list the cases and exit")
    args = ap.parse_args()

    cases = tier1_cases() if args.tier == 1 else tier2_cases()
    if args.only:
        cases = [c for c in cases if args.only in c["case_id"] or c["case_id"] == "base"]

    if args.dry_run:
        print(f"Tier {args.tier} cases ({len(cases)}):")
        for c in cases:
            print(f"  {c['case_id']:<34} param={c['param']:<18} level={format_case_level(c)}")
        return

    if not pyo.SolverFactory("scip").available(exception_flag=False):
        sys.exit("SCIP solver not found. Install it / add it to PATH before running the sweep.")

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # The base case always runs first, synchronously, and is gated before any
    # sweep case is trusted.
    non_base = [c for c in cases if c["case_id"] != "base"]
    if args.tier == 1:
        base_row = run_tier1_case(BASE_CASE)
        gate_tier1(base_row, args.no_gate)
        errors = run_cases(non_base, run_tier1_case, args.jobs)
        out_csv = rebuild_tier1_csv()
    else:
        base_rows = run_tier2_case(BASE_CASE)
        gate_tier2(base_rows, args.no_gate)
        errors = run_cases(non_base, run_tier2_case, args.jobs)
        out_csv = rebuild_tier2_csv()

    meta = {
        "tier": args.tier,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "cases_run": [c["case_id"] for c in cases],
        "errors": errors,
        "total_s": round(time.time() - t_start, 1),
    }
    with open(SWEEP_ROOT / f"tier{args.tier}_run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Tier {args.tier} complete in {meta['total_s']}s. "
          f"{len(cases) - len(errors)}/{len(cases)} cases succeeded.")
    if errors:
        print(f"Failed cases: {sorted(errors)} (tracebacks in "
              f"{SWEEP_ROOT / f'tier{args.tier}_run_metadata.json'})")
    print(f"Aggregate results: {out_csv}")
    print("Next: python plot_sensitivity_results.py")


if __name__ == "__main__":
    main()
