# 03 — Revenue-Stacking Formulation (extension of the existing MILP)

Written as a delta against the existing Pyomo model (`optimisation.py`,
`constraints.py`). Everything from Paper 1 is retained unchanged: job scheduling
(`ut_ks`, JobCompletion), the SOS2 piecewise CPU→power map, UPS energy balance,
thermal dynamics, TES balance, power balance, temperature bounds. The extension
adds market commitments on top of the physical schedule and changes the objective
from cost minimisation to net-cost (profit) minimisation.

## 1. Sets and indices

| Symbol | Meaning | Code mapping |
|---|---|---|
| `s ∈ T_ext` | 15-min slots, 1…108 (96 + 12 recovery) | `m.TEXT_SLOTS` (unchanged) |
| `b ∈ B` | EFA blocks (4 h = 16 slots), 6 per day | new set; `S_b` = slots in block b |
| `h ∈ H` | settlement periods (30 min = 2 slots), 48 per day | new set (BR granularity) |
| `j ∈ J↓` | turn-down (demand-reduction) products: {DC-L, DM, DR-L, BR+, PQR, SR} | new |
| `j ∈ J↑` | turn-up products: {DC-H, DR-H, BR−, NQR, LCM} | new |
| `ω ∈ Ω` | scenarios (Stage-3 stochastic variant only) | new |

Commitment granularity follows each product's procurement: response (DC/DM/DR)
per EFA block b; Balancing Reserve per settlement period h; QR/SR per their
auction windows (parameterise in `market_parameters.py`, don't hard-code);
BM/DFS/DSO actions per settlement period mapped to slots. Below, `r[j,b]` is
shorthand for "commitment at product j's own granularity".

## 2. New decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `r[j,b] ≥ 0` | MW | capacity committed to product j in block b |
| `x_bm_off[s], x_bm_bid[s] ≥ 0` | MW | BM offer (turn-down) / bid (turn-up) energy delivered in slot s |
| `x_dfs_dn[s], x_dfs_up[s] ≥ 0` | MW | DFS turn-down / turn-up delivery (event slots only; bi-directional since Apr 2026) |
| `x_dso[w] ≥ 0` | MW | DSO availability committed in tendered window w |
| `C_CM ≥ 0` | MW | Capacity Market commitment (outer-loop scalar, fixed per run) |

Everything else (grid power, UPS, TES, temperatures, job schedule) is the existing
variable set defining the **baseline trajectory** `P(s)`:

```
P(s) = p_grid_it_kw[s] + p_chiller_hvac_w[s]/1000 + p_chiller_tes_w[s]/1000
       + nominal_overhead_addition + p_ups_ch_kw[s]          # as in objective_rule()
```

## 3. Headroom / footroom constraints (the stacking core)

Commitments must be simultaneously deliverable. For every slot s in block b:

**(R1) Turn-down headroom** — total committed reduction plus any BM/DFS/DSO
delivery must fit inside what the facility can actually shed at s:

```
Σ_{j∈J↓} r[j,b] + x_bm_off[s] + x_dfs_dn[s] + x_dso[w(s)] + C_CM·1{stress test window}
      ≤  H↓(s)
```

**(R2) Turn-up footroom** — symmetric, and additionally capped by the grid
connection:

```
Σ_{j∈J↑} r[j,b] + x_bm_bid[s] + x_dfs_up[s]  ≤  H↑(s),
P(s) + Σ_{j∈J↑} r[j,b] + x_bm_bid[s] + x_dfs_up[s]  ≤  P_conn   # import-capacity cap (new parameter)
```

`H↓(s)`, `H↑(s)` are the *instantaneous* deliverable headrooms implied by the
physical model. Two ways to define them — this is the key formulation choice:

### Variant A — embedded delivery duplicates (exact, heavier)

For each product j with sustain duration `τ_j`, add a *delivery copy* of the state
trajectory: if called at any slot s, the facility follows a perturbed trajectory
for `τ_j` slots delivering `r[j,b]` MW, then recovers, while all physical
constraints hold. This is exactly the structure of `build_duration_model()` in
`flexibility_duration.py` (fix the baseline, add `add_power_change_constraints`,
check feasibility) — but embedded for *every possible call time*, which multiplies
model size by O(|calls| × τ). Tractable for a single product, explosive for a
portfolio. Use only for spot-validation of Variant B.

### Variant B — envelope-based deliverability (the recommended, paper-linking route)

Paper 1's flexibility assessment already computes, for the cost-optimal baseline,
the maximum feasible duration `τ*(s, ΔP)` for any start slot and deviation — i.e.
a **magnitude–duration envelope** `F(s)`: the set of (ΔP, τ) pairs deliverable
from slot s. Invert it to `ΔP_max(s, τ)` and impose:

```
(R1′)  Σ_{j∈J↓} r[j,b] · 1{τ_j ≤ τ} + (BM/DFS/DSO terms)  ≤  ΔP↓_max(s, τ)   ∀ τ ∈ {τ_j}, ∀ s ∈ S_b
(R2′)  analogous for J↑ with ΔP↑_max(s, τ)
```

i.e. nested-duration constraints: everything that must sustain ≥ τ fits inside the
τ-envelope. Plus first-order *component* headrooms evaluated directly on the
baseline variables (linear, endogenous — these keep the commitment decisions
coupled to the schedule instead of a fixed pre-computed envelope):

```
(R1a) IT headroom:      ΔP_IT↓(s) ≤ shed-able flexible load at s (from ut_ks with delay room left) mapped through the PW power curve
(R1b) UPS headroom:     ΔP_UPS↓(s) ≤ p_max_disch_kw − p_ups_disch_kw[s]
(R1c) Cooling headroom: ΔP_cool↓(s) ≤ p_chiller_hvac_w[s]/1000 (TES can substitute, bounded by R4)
      H↓(s) = ΔP_IT↓(s) + ΔP_UPS↓(s) + ΔP_cool↓(s)
```

**Two-loop scheme (proposed algorithm, and a contribution in itself):**
1. Solve the stacking MILP with linear headrooms (R1a–c) + energy reservations (R3–R5).
2. Certify the resulting commitments with the Paper-1 duration machinery
   (Variant A feasibility checks at sampled call times).
3. If a commitment fails certification, add a cut (tighten `ΔP_max(s,τ)`) and
   re-solve. Converges in a few iterations in practice; report the certification
   gap. This makes "duration-certified market offers" the headline method.

## 4. Energy-reservation constraints (why fast products are energy-capped)

Committed response must be backed by stored energy for its sustain duration at
*every* slot of the block, on top of the resilience floor:

```
(R3)  e_ups_kwh[s] ≥ e_min_kwh + Σ_{j∈J↓} α_ups[j,b] · r[j,b] · τ_j / η_disch      ∀ s ∈ S_b
(R4)  e_tes_kwh[s] ≥ E_TES_min + Σ_{j∈J↓} α_tes[j,b] · r_th[j,b] · τ_j / η_TES,dis  ∀ s ∈ S_b
(R5)  e_ups_kwh[s] ≤ e_max_kwh − Σ_{j∈J↑} α_ups[j,b] · r[j,b] · τ_j · η_ch          ∀ s ∈ S_b   (room to absorb)
```

where `α_asset[j,b] ∈ [0,1]` are (optional) explicit asset-allocation shares with
`Σ_asset α_asset[j,b] = 1` — making *which asset backs which product* a decision
variable and directly answering RQ2 (asset attribution). For DR, additionally
model expected throughput (continuous cycling) as an energy drain term calibrated
from historical frequency distributions — this is what should make the optimiser
reject DR for the resilience-floored UPS.

Note (R3)–(R5) make Paper 1's terminal conditions (`e_ups(end) = e_start`,
TES cycling) *product-aware*: recovery after a call is priced, not assumed free.

## 5. Objective

```
min   Σ_s Δt · P(s) · π_DA(s)/1000                                  # energy purchase (existing)
    + Σ_s Δt · P(s) · π_DUoS(s)/1000                                # residual time-varying network charge
    − Σ_b Σ_j π_av[j,b] · r[j,b] · 4                                 # availability revenue (£/MW/h × 4 h)
    − Σ_b Σ_j φ[j] · π_util[j,b] · r[j,b] · E[energy called]         # expected utilisation (calibrated call factors φ)
    − Σ_s Δt · ( x_bm_off[s] · π_off(s) · κ_acc − x_bm_bid[s] · π_bid(s) · κ_acc )   # BM with acceptance haircut κ
    − Σ_{s∈events} Δt · ( x_dfs_dn[s] · π_DFS,dn + x_dfs_up[s] · π_DFS,up )
    − Σ_w x_dso[w] · π_DSO,av · hrs(w)  −  utilisation term
    − C_CM · δ_derate · π_CM / 365                                   # CM layer, daily share
```

Availability prices `π_av[j,b]` are exogenous historical EAC clearing prices —
the facility is a price-taker (defensible at 10 MW; state it). Delivered service
energy also reduces/increases the DA purchase; keep accounting consistent
(decide: settle deviations at utilisation price, doc 04 §5).

## 6. Uncertainty extension (Stage 3)

Two-stage stochastic program: first stage = commitments `r[j,b]`, BM/DSO
positions, and the *purchased* DA schedule; second stage (scenario ω) = physical
dispatch, utilisation calls, DFS events, price realisations. Objective becomes
`E_ω[net cost] + λ·CVaR_95[net cost]`; sweep λ for a risk–return frontier.
Scenario set: bootstrap historical days (prices × frequency-events × call
patterns), reduced to 10–20 scenarios by clustering. The deterministic model is
the λ=0, |Ω|=1 special case, so this is an additive code change.

## 7. Model statistics & tractability

Existing model: ~108 slots, ~3.5 k continuous vars, ~220 binaries (UPS z_ch/z_disch),
SOS2 ×108 — solves in seconds with SCIP. The stacking layer adds ~10 products × 6
blocks ≈ 60 commitment vars + O(10³) linear constraints: negligible. Annual runs =
365 independent daily solves (embarrassingly parallel — reuse the
`sensitivity_sweep.py` ProcessPool pattern). The stochastic variant multiplies the
physical block by |Ω|; keep |Ω| ≤ 20 and relax UPS binaries to continuous where the
min-charge thresholds aren't binding (validate on sample days).

## 8. Implementation sketch

```
rev_stack/model/
  market_parameters.py   # products: {name, direction, τ_j, speed class, min MW, price series column}
  price_data.py          # loaders for DA, EAC, BM, DFS, DSO, CM series → per-slot arrays
  stack_constraints.py   # R1–R5 + connection cap (imports & reuses constraints.py untouched)
  stack_objective.py     # §5
  run_day.py             # one-day co-optimisation (mirrors run_single_optimization)
  run_year.py            # rolling annual harness (mirrors sensitivity_sweep.py structure)
  certify.py             # Variant-A certification calls into flexibility_duration.py
```
