# Sensitivity Analysis — Implementation Plan

Response to **R2-3** (model response under different electricity price fluctuation ranges, IT load
flexibility ratios, and equipment parameters). Also converts the self-flagged limitation in the
Scenario 3 text (24–25 °C cold-aisle sensitivity "would be useful future work") into a result, and
indirectly strengthens **R2-1** (generality of the framework).

Design principle: **broad one-at-a-time (OAT) sensitivity on the cheap metric (Scenario 2 cost
saving), targeted probes on the expensive metric (Scenario 3 duration τ), no full heatmap re-runs
except optionally for supplementary material.**

> **Status: implemented.** The sweep lives in [`sensitivity_sweep.py`](../sensitivity_sweep.py)
> (Tiers 1 and 2) and [`plot_sensitivity_results.py`](../plot_sensitivity_results.py) (the §5.4
> two-panel figure). Run from the repo root:
> `python sensitivity_sweep.py --tier 1` → inspect gate + tornado numbers →
> `python sensitivity_sweep.py --tier 2` (optionally `--jobs 4`) →
> `python plot_sensitivity_results.py`. Outputs land in
> `static/data/sensitivity_outputs/` and `static/images/sensitivity_outputs/`.

---

## 1. How the pipeline actually works (grounding)

Three scripts form a file-coupled pipeline; each reads the previous stage's CSVs via
**module-level relative `pathlib.Path` constants**:

| Stage | Script | Reads | Writes |
|---|---|---|---|
| Scenario 1 (base) | `nominal_calculation.py` | `static/data/inputs/load_profiles.csv`, `shiftability_profile.csv` | `static/data/nominal_outputs/nominal_case_results.csv` |
| Scenario 2 (opt) | `optimisation.py` | same inputs + `nominal_case_results.csv` (for saving %) | `static/data/optimisation_outputs/optimised_baseline.csv`, `load_profiles_opt.csv`, `shiftability_profile_opt.csv` |
| Scenario 3 (τ) | `flexibility_duration.py` | all three Scenario 2 outputs | `static/data/flexibility_outputs/flex_duration_results.csv`, detailed per-(t₀, ΔP) CSVs, heatmap |

Key mechanics confirmed in code:

- **Tariff** is hard-coded in `generate_tariff()` ([inputs/parameters_optimisation.py:8](../inputs/parameters_optimisation.py)):
  24 hourly prices `[60, 55, 52, 50, 48, 48, 55, 65, 80, 90, 95, 100, 98, 95, 110, 120, 130, 140, 135, 120, 100, 90, 80, 70]` £/MWh.
  Daily mean ≈ **86.92 £/MWh**, min 48, max 140. Both `optimisation.py` and `flexibility_duration.py`
  call it inside their own `load_and_prepare_data()`; the returned array has a **dummy 0 inserted at
  index 0** (1-based slot indexing) and is tiled to cover the 108-slot extended horizon (96 + 12 recovery slots).
- **Workload split** lives in `static/data/inputs/load_profiles.csv` (`inflexible_load`,
  `flexible_load` columns, 96 rows, CPU fractions). `shiftability_profile.csv` splits the flexible
  column into 4 delay tranches whose fractions sum to 1.0 per slot, i.e. *all* of `flexible_load`
  is shiftable. `data['Rt'] = flexible × dt_hours` is the work-completion RHS.
- **Parameters** are all in `ModelParameters` / `setup_simulation_parameters()`
  ([inputs/parameters_optimisation.py:16](../inputs/parameters_optimisation.py)). Relevant ones:
  - TES: `TES_kwh_cap = 1000` kWh; **derived at construction time**: `TES_initial_charge_kWh = 0.5 × cap`,
    alias `TES_capacity_kWh`. Charge/discharge power 300 kW each, efficiencies 0.9/0.9, `E_TES_min = 0`.
    End-of-horizon cycling constraint: `e_tes(end) == e_tes(start)`.
  - UPS: `e_nom_kwh = 600`, `soc_min = 0.5`, `e_start_kwh = 600` (starts full); **derived**:
    `e_min_kwh`, `e_max_kwh`. End-of-horizon: `e_ups(end) >= e_start`. `p_max_ch = 270`, `p_max_disch = 2700` kW.
  - Cold aisle: `T_cAisle_upper_limit_Celsius = 23` — used directly as the `t_cold_aisle` variable
    upper bound in all three model builders (`optimisation.py:78`, `flexibility_duration.py:65`,
    `nominal_calculation.py:66`).
  - Tranches: `tranche_max_delay = {1:2, 2:4, 3:8, 4:12}` — but `flexibility_duration.main()`
    **overrides** this to `{k:k for k in 1..12}` with `K_TRANCHES = 1..12`, because Scenario 2's
    output re-expresses remaining shiftability in 12 tranches.
- **Scenario 3 search**: `find_max_duration()` binary/linear-searches τ, each step one SCIP MILP
  feasibility solve (60 s time limit, typically ~10 s). It has a **result-banking mechanism**
  (`include_banked_results`) that silently reuses `flex_duration_results.csv` — sweeps must
  disable this or point each case at its own bank file, otherwise stale base-case results leak in.
- **Timestep ↔ clock mapping**: slot `ts` ends at `ts × 15 min` (so ts=1 → 00:15, ts=70 → 17:30, ts=72 → 18:00).

### Base-case numbers the sweep must reproduce at multiplier 1.0 (regression anchors)

- Base cost **1659.54 GBP**, optimised **1493.19 GBP**, saving **10.02 %** (paper Results table).
- τ(t₀=00:15, ΔP=−100 kW) = **6.8 h**; τ(t₀=17:30, ΔP=−100 kW) = **0.2 h** (abstract + conclusion).

---

## 2. Parameters and levels (OAT design around the base case)

| # | Parameter | Code hook | Levels (multiplier / value) | Requires Scen 1 re-run? | Requires Scen 2 re-run? |
|---|---|---|---|---|---|
| S1 | Flexible workload share | scale `flexible_load` column, move remainder to `inflexible_load` (total preserved) | ×{0.5, 0.75, 1.25, 1.5} | **No** (total CPU profile unchanged ⇒ base cost identical) | Yes |
| S2 | Price volatility | π′(t) = π̄ + k·(π(t) − π̄), π̄ = 86.92 £/MWh | k ∈ {0.5, 0.75, 1.25, 1.5} | **Yes** (base cost changes with prices) | Yes |
| S3 | TES energy capacity | `TES_kwh_cap` (+ re-derive `TES_capacity_kWh`, `TES_initial_charge_kWh`) | ×{0.5, 0.75, 1.25, 1.5} | No* | Yes |
| S4 | Cold-aisle upper bound | `T_cAisle_upper_limit_Celsius` | {24, 25} °C (23 = base) | No | No — apply in Scenario 3 only (see §4) |
| S5 (optional) | UPS capacity | `e_nom_kwh` (+ re-derive `e_min/e_max/e_start`) | ×{0.5, 1.5} | No* | Yes |

Notes:
- **S1 mechanics** (implemented as a capped inter-pool transfer, not a raw multiply): move
  `transfer(t) = min((m−1)·flex(t), inflex(t))` from the inflexible to the flexible pool, giving
  `flex′ = flex + transfer`, `inflex′ = inflex − transfer`. This preserves the per-slot total
  exactly and keeps both pools ≥ 0 for any m. It equals `flex′ = m·flex` for m < 1. **Important
  data fact:** inflexible load is *not* 0.21–0.28 everywhere — in the early-morning trough
  (slots 15–23, ~03:45–05:45) it falls to 0.06 against flexible 0.27, i.e. already ~82% flexible.
  The uniform-multiplier feasibility ceiling is therefore m ≤ 1 + min(inflex/flex) ≈ **1.22**, so a
  raw ×1.25/×1.5 multiply is infeasible; the capped transfer instead saturates those slots at 100%
  flexible and records `realized_flex_multiplier`/`realized_flex_share` (both < nominal for
  ×1.5). Because total CPU per slot is unchanged, nominal power/cost is invariant ⇒ the saving
  denominator stays 1659.54.
- **S2 mechanics**: mean-preserving spread scaling isolates *volatility* from *price level* — the
  clean answer to "different electricity price fluctuation ranges". Apply by overwriting
  `data['electricity_price']` **after** `load_and_prepare_data()` in both Scenario 1/2 wrappers
  (keep the index-0 dummy zero; scale slots 1..108). At k=1.5, min price = 48 → ~28.5 £/MWh (still
  positive; no sign-flip complications). Report savings against the **re-run base cost at the same k**.
- **S3/S5**: scale energy capacity only; keep charge/discharge power limits fixed (isolates sizing
  of the *energy* buffer; say so in the text). Both have derived attributes computed in
  `__init__`/`setup_simulation_parameters` — the wrapper must recompute them after overriding, or
  take the multiplier as a constructor argument.
- ***No\* caveat — RESOLVED**: `nominal_calculation.configure_nominal_params()`
  (nominal_calculation.py:396) pins the nominal case properly: `TES_w_charge_max` and
  `TES_w_discharge_max` are zeroed (so all TES flow variables are bounded to 0), and
  `add_power_balance_constraints_nominal()` excludes UPS discharge from the power balance
  (`p_it_total == p_grid_it`), making UPS activity pure cost that the minimiser avoids. The base
  cost therefore depends **only on the tariff**, confirming the re-run table above. The sweep
  wrapper still records nominal storage-use columns per case and warns if they are ever non-zero.
- Deliberately **excluded**: `nominal_overhead_addition` (constant adder, trivially linear in cost),
  COP, power-system tolerance (0.1 kW), recovery-window length — no reviewer asked, no physical story.

---

## 3. Tier 1 — cost-saving sensitivity (Scenario 2, ~17–20 MILP solves)

For each case: run Scenario 1 (only where required per table), run Scenario 2, record:

- `base_cost_GBP`, `opt_cost_GBP`, `saving_GBP`, `saving_pct` (recomputed against the case's own base),
- decomposition of the saving by component (IT / cooling / UPS / TES grid-cost deltas) — this
  doubles as material for **R2-4** (cost-composition comment) at zero extra compute,
- solver status + wall time.

Each solve is seconds-to-minutes; the whole tier runs in well under an hour.

**Output**: `static/data/sensitivity_outputs/tier1_cost_sensitivity.csv` (one row per case:
`param, multiplier, base_cost, opt_cost, saving_abs, saving_pct, ...`).

## 4. Tier 2 — duration sensitivity (Scenario 3 probes, no heatmaps)

Fix **three reference products** — deliberately the exact numbers already quoted in the
abstract/conclusion, so the sensitivity section stress-tests figures the reader has just seen:

| Probe | t₀ | slot | ΔP | base τ |
|---|---|---|---|---|
| P1 (easy upward) | 00:15 | ts=1 | −100 kW | 6.8 h |
| P2 (hard upward) | 17:30 | ts=70 | −100 kW | 0.2 h |
| P3 (downward) | 18:00 | ts=72 | +100 kW | (read from base heatmap) |

For each Tier-1 case (S1, S2, S3, S5): re-run Scenario 2 → regenerate
`optimised_baseline.csv` + `load_profiles_opt.csv` + `shiftability_profile_opt.csv` for that case
→ run `find_max_duration` for the 3 probes with `search_type='binary'` and **banking disabled**.
τ is measured from the *re-optimised* baseline, which is the methodologically correct
interpretation (the flexibility envelope of a cost-optimally-operated DC under those parameters).

For **S4 (temperature)**: keep the base Scenario 2 baseline unchanged; override
`T_cAisle_upper_limit_Celsius` ∈ {24, 25} only in the `flexibility_duration` model build. This
matches the paper's framing (the 18–23 °C range is a *Scenario 3* flexibility range) and directly
answers the self-flagged future-work sentence. 2 levels × 3 probes = 6 probe runs.

Compute budget: ~16 parameter cases × 3 probes × ~7 binary-search MILPs × ~10 s ≈ **1–2 h**, plus
Scenario 2 re-runs (~16 × ~1 min). Comfortable overnight run with large margin (60 s solver cap
bounds the worst case at ~6 h).

Expected story worth checking for: τ at P2 (17:30, tight baseline) should be far more sensitive to
TES capacity and temperature headroom than τ at P1 — direct quantitative support for the paper's
baseline-state-dependence argument.

**Output**: `static/data/sensitivity_outputs/tier2_duration_sensitivity.csv`
(`param, multiplier, probe, t0, dP_kw, tau_hours`).

## 5. Tier 3 (optional, supplementary material only)

One full heatmap re-run at S1 low (×0.5) and S1 high (×1.5) — the parameter with the weakest
empirical grounding and the reviewer's first-named axis. Reuse `flexibility_duration.main()` with
the full timestep/magnitude grid. Only do this if time allows; Applied-Energy-style supplementary
files are the right home. Budget: several hours each — run overnight, background.

---

## 6. Implementation notes (wrapper design — do NOT modify the three scripts' behaviour)

Create `sensitivity_sweep.py` (repo root) that:

1. **Imports** the existing building blocks rather than shelling out:
   `from optimisation import build_model, load_and_prepare_data, run_single_optimization`,
   `import flexibility_duration as fd`, `import nominal_calculation as nom`.
2. **Redirects the module-level path globals per case** (they are plain module attributes, so
   `opt.DATA_DIR_OUTPUTS = case_dir / "optimisation_outputs"` etc. works) — or, equivalently,
   passes per-case directories after a light refactor of the constants into function defaults.
   Every case writes into `static/data/sensitivity_outputs/<param>_<level>/...` so the canonical
   base-case outputs are never overwritten.
3. **Applies overrides** in one place, `apply_case(params, data, case)`:
   - S2: rescale `data['electricity_price']` in place (mean-preserving formula above).
   - S1: rescale the two columns of the loaded `load_profiles.csv` DataFrame *before* the
     resampling logic (cleanest: write a per-case `load_profiles.csv` into the case input dir and
     point the loader at it; `shiftability_profile.csv` is unchanged since its tranche fractions
     are relative to the flexible column).
   - S3/S5: set the multiplier, then **re-derive** `TES_capacity_kWh`, `TES_initial_charge_kWh`
     (S3) or `e_min_kwh`, `e_max_kwh`, `e_start_kwh` (S5).
   - S4: override `params.T_cAisle_upper_limit_Celsius` (Scenario 3 model build only).
4. **Computes savings internally** (don't rely on `print_summary`, which reads the static-path
   nominal CSV and includes the slot 97–108 extension-cost correction — replicate that correction
   in the wrapper so the 10.02 % base number reproduces exactly).
5. **Disables Scenario 3 banking** (`include_banked_results=None` semantics / fresh bank file per
   case) and passes `generate_plots=False`; remember the `tranche_max_delay`/`K_TRANCHES` override
   to 12 tranches that `fd.main()` performs — the wrapper must do the same before probe runs.
6. **Regression gate**: case `base` (all multipliers 1.0) must reproduce 1659.54 / 1493.19 /
   10.02 % and τ = {6.8, 0.2} h for P1/P2 before any other case is run. Make this an assert.
7. Writes the two tier CSVs plus a JSON of run metadata (solver times, statuses) for the response
   letter's reproducibility claim.

Order of execution: verification step 0 (nominal storage check) → regression gate → Tier 1
(fast; inspect tornado) → decide S5 inclusion → Tier 2 overnight → figures → Tier 3 if desired.

---

## 7. Paper integration (length-neutral strategy)

Main-text budget: **one two-panel figure + ~350–400 words**, as new subsection
**5.4 "Sensitivity Analysis"** placed before "Value of Integrated Co-optimisation"
([main_new.tex:775](main_new.tex)).

- **Panel (a)** — tornado chart of cost-saving % (one horizontal bar per parameter spanning its
  low→high multiplier effect around the 10.02 % base). Most space-efficient "which parameter
  matters" visual; reviewers parse it instantly.
- **Panel (b)** — τ vs. multiplier line plots for the three probes (P1/P2/P3), one mini-panel per
  parameter or a shared normalised x-axis. The P1-vs-P2 sensitivity contrast is the science.
- Full numeric grids → supplementary material, not the main text.
- **Methods**: 2–3 sentences at the end of Section 4 defining the OAT design and the
  mean-preserving price-spread scaling. No new methods subsection.

Consistency edits (must-do, or the new section contradicts existing text):

1. [main_new.tex:783](main_new.tex) (conclusion, future work): delete "testing sensitivity to
   electricity-price volatility, workload flexibility ratios and TES capacity" — it's now done.
   Keep revenue stacking / pre-conditioning / portfolio aggregation.
2. [main_new.tex:694](main_new.tex) (end of Scenario 3 description): delete/rewrite "A sensitivity
   study using 24–25 °C bounds would be useful future work" → point forward to Section 5.4 results.
3. Offset length with the already-accepted condensation of Section 4 scenario descriptions
   (accepted half of R2-3) — the Scenario 3 tranche exposition has the most slack. Target ≈ 0.4–0.5
   page freed, making the revision roughly length-neutral; say so in the response letter.
4. If a clean headline emerges (e.g. near-linear saving vs. price spread), consider spending one of
   the 85-char highlights on it ([main_new.tex:81](main_new.tex) area).

Response-letter framing for R2-3: "we added a sensitivity analysis covering all three parameter
classes the reviewer identified — price fluctuation range, IT load flexibility ratio, and equipment
sizing (TES [and UPS] capacity, cold-aisle temperature limit) — using an OAT design around the
base case," while declining the full multi-dimensional case matrix on scope grounds. This replaces
the current flat refusal (🔴) with a substantive concession + bounded scope, and cross-references
R2-1 (robustness ⇒ generality) and R2-4 (saving decomposition from Tier 1).

---

## 8. Gotchas checklist (things that will silently corrupt results)

- [x] Nominal-case storage use verified ≈ 0 — pinned by construction in
      `configure_nominal_params()` + nominal power balance (§2 note); wrapper re-checks per case.
- [ ] `electricity_price[0]` dummy zero preserved after scaling; scaling covers all 108 slots.
- [ ] Price-volatility savings reported against the *matching* re-run base cost, and stated so in the text.
- [x] `inflex′(t) ≥ 0` for S1 multipliers > 1 — guaranteed by the capped inter-pool transfer;
      realized (vs nominal) multiplier recorded because ×1.25/×1.5 saturate the 03:45–05:45 trough.
- [ ] `TES_initial_charge_kWh` re-derived when `TES_kwh_cap` changes (it is fixed at construction).
- [ ] UPS `e_start_kwh = e_nom_kwh` and `e_min/e_max` re-derived for S5.
- [ ] Scenario 3 banking (`include_banked_results`) disabled per case; per-case output dirs so
      `optimised_baseline.csv` etc. never cross-contaminate cases.
- [ ] 12-tranche override applied before probe runs (as `fd.main()` does).
- [ ] Extension-window (slots 97–108) cost correction replicated when computing savings.
- [ ] Multiplier-1.0 regression gate passes for both tiers before sweeping.
