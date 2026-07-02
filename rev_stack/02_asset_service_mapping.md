# 02 — Which Assets Should Provide Which Services?

Maps the facility's flexibility assets (the four modelled in Paper 1, plus standby
generation as a proposed addition) onto the GB service catalogue (doc 01), and
fixes the modelling shortlist. The physical characteristics below are the model's
own parameters (`inputs/parameters_optimisation.py`), scaled ×10 for the 10 MW
case (scaling table in doc 04 §3). Service specs verified July 2026 (sources in
doc 01); EPRI's DCFlex taxonomy (compute assets / balance-of-plant / power
assets) maps cleanly onto our IT / cooling+TES / UPS+genset split and is worth
citing as the international framing.

## 1. Asset characterisation (what each asset can physically sell)

| Asset (10 MW scale) | Power capability | Energy capability | Speed | Direction | Key restriction |
|---|---|---|---|---|---|
| **IT workload shifting** (4 tranches, 0.5/1/2/3 h max delay) | up to ~7–8 MW turn-down of the flexible share at peak (depends on `flexible_load` profile & the P∝CPU^1.32 curve) | hours (limited by tranche delay windows + mandatory recovery/payback) | seconds–minutes with orchestration; assume ≥ 2 min conservatively | strong turn-down; weak turn-up (can only advance work that exists — Paper 1's asymmetry result) | payback: deferred work *must* execute within its window → post-event rebound the market position must absorb |
| **UPS battery** | 2.7 MW ch / 27 MW disch (nameplate ×10) — but *contracted* MW capped by energy, not power | 6 MWh nominal; usable = (SoC floor ↔ 100 %) ⇒ 3 MWh above the 50 % resilience floor | inverter-speed (< 1 s) — grid-interactive UPS | symmetric (discharge = turn-down; charge = turn-up, capped at 2.7 MW) | resilience floor: SoC ≥ 50 % at all times ⇒ e.g. max 12 MW of DC-L for the 15 min sustain, less after netting round-trip losses (η_ch 0.82, η_dis 0.92) |
| **TES + chiller headroom** | ±3 MW_th ⇒ ±0.6 MW_e at COP 5; chiller headroom 4 MW_e cap | 10 MWh_th ≈ 2 MWh_e equivalent | minutes (valve/pump); assume ≥ 2 min | symmetric-ish: pre-cool = turn-up, discharge TES = turn-down of chiller power | must keep cold-aisle 18–23 °C; TES cycling constraint; its main role is **extending the duration** of IT-led reductions (cooling follows IT heat with a lag) |
| **Overhead loads** (531 kW at 10 MW scale) | none | — | — | — | fixed; keep as constant |
| **Standby generation** (proposed addition, not in Paper 1) | typically 1:1 with IT load ⇒ ~10–12 MW | tank-limited, effectively many hours | ≤ 2 min start (diesel), ~10 min to full | turn-down of *net* grid demand (behind-the-meter generation) | MCPD/environmental permit run-hour caps (~50–500 h/yr ⚠ verify per permit type), fuel cost ~£200–300/MWh, air-quality constraints in urban zones; reputational/ESG considerations for diesel |

**Recommendation on gensets:** include as an *optional scenario*, not the base
case. It is the honest engineering reality (every DC has them, and they dominate
US demand-response programmes), but diesel run-hour limits and ESG policies mean
many operators won't dispatch them for revenue. A with/without-genset comparison
is a strong result in itself (hypothesis: gensets add CM + Slow Reserve value but
don't displace the UPS/IT stack).

## 2. Asset × service suitability matrix

✓✓ = primary fit, ✓ = feasible/secondary, ○ = marginal, ✗ = infeasible.

| Service | IT shift | UPS | TES/cooling | Genset | Binding reason |
|---|---|---|---|---|---|
| DA arbitrage | ✓✓ | ✓ | ✓ | ✗ (fuel > price) | IT shift is the volume; UPS/TES small but free riders on the same optimisation |
| Dynamic Containment low (DC-L) | ○ (fast capping unproven at scale) | ✓✓ | ✗ (too slow) | ✗ | ≤1 s response ⇒ inverter assets only |
| Dynamic Containment high (DC-H) | ✗ (can't absorb) | ✓ (charge headroom only 2.7 MW) | ○ | ✗ | turn-up needs absorption capacity |
| Dynamic Moderation | ○ | ✓✓ | ✗ | ✗ | 10 s response |
| Dynamic Regulation | ✗ | ○ (throughput kills resilience floor) | ✗ | ✗ | continuous cycling |
| Balancing Reserve (±) | ✓ | ✓ | ✓ | ○ | delivered via BM ⇒ needs BMU/VLP registration; per-settlement-period product |
| Quick Reserve (PQR/NQR) | ○ (1-min TTFD is tight for orchestration — test assumption) | ✓✓ | ✓ | ○ | full delivery ≤ 1 min, 1-s metering; non-BM route live since Sep 2025 |
| Slow Reserve / STOR | ✓✓ | ○ (duration) | ✓ (duration extender) | ✓✓ | 15-min TTFD suits schedulers; 15-s metering |
| BM offers (turn-down) | ✓✓ | ✓ | ✓ | ✓ | pay-as-bid energy; OBP has raised small-unit dispatch rates |
| BM bids (turn-up) | ○ (advance-only) | ✓ (charge) | ✓✓ (pre-cool) | ✗ | Paper 1 asymmetry |
| DFS turn-down | ✓✓ | ○ (metering-point netting) | ✓ | ✗ (low-carbon eligibility rules) | 30-min events; Self-Nominated baseline (Apr 2026) suits I&C |
| DFS turn-up (new, Apr 2026) | ○ (advance-only) | ✓ (charge) | ✓✓ (pre-cool) | ✗ | zonal, bi-directional DFS gives an LCM-like product GB-wide |
| Capacity Market | ✓ (proven DSR) | ✓ | ○ | ✓✓ | annual obligation; needs guaranteed deliverability |
| DSO Peak Reduction | ✓✓ | ✓ | ✓ | ✓ | winter-evening windows |
| LCM (turn-up, Scotland) | ○ | ✓ | ✓✓ | ✗ | absorb surplus wind |

## 3. The two structural insights the matrix encodes

1. **Speed–duration complementarity.** The UPS is power-rich/energy-poor (27 MW
   vs 3 MWh usable) — worthless for hours-long products, ideal for sub-second,
   15-min-sustain response. IT shifting is the mirror image: MW-limited by the
   flexible workload share but sustains for hours (Paper 1: 100 kW for 6.8 h at
   1 MW scale). TES is neither fast nor big but *couples* to IT: it extends and
   smooths IT-led reductions and provides most of the credible turn-up. The
   optimal stack should therefore not be one service — it is a **layered
   portfolio sorted by response speed**, and the formulation must let different
   MW of different assets hold different products simultaneously.
2. **The rebound is a market position.** Deferred IT work must run later
   (JobCompletion constraints); a naive service commitment ignores that the
   payback energy is bought at post-event prices. Co-optimising commitments *with*
   the energy schedule prices the rebound correctly — this is the main
   methodological advantage over per-service valuation papers, and worth a
   dedicated result figure (value of co-optimisation vs sequential/greedy
   stacking).

## 4. Stacking & exclusivity rules to encode

| Rule | Encoding in the model |
|---|---|
| Same MW cannot provide two NESO services in the same settlement period | shared headroom/footroom constraint: Σ(turn-down commitments) ≤ available headroom at every t; same for turn-up (§03 eq. R1–R2). No binaries needed |
| Response (DC/DM/DR) requires reserved *energy* as well as MW | energy-reservation constraints on UPS SoC and TES level (§03 eq. R3–R5) |
| BR requires BM registration; BM and wholesale positions interact | model facility as a single VLP/BMU; BM modelled as system-price arbitrage with acceptance haircut (doc 04 §5) |
| DFS Primacy process (introduced Apr 2026) governs conflicts with other services | on DFS event periods, DFS delivery MW excluded from headroom offered to other services; encode Primacy as strict exclusivity (conservative) |
| CM is stackable with everything except during stress events/tests | standing deliverability constraint sized by `C_CM`; no revenue interaction otherwise |
| DSO vs NESO primacy | availability windows constrain headroom like any other commitment; assume no simultaneous conflicting dispatch (state assumption) |
| UPS resilience is non-negotiable | hard SoC floor; **sensitivity: floor ∈ {30, 50, 70} %** to price the cost of conservatism — a result operators actually want |

## 5. Shortlist decision

- **Co-optimised daily (endogenous):** DA arbitrage, DC-L, DC-H, DM, DR, BR±,
  QR (PQR/NQR), SR, BM (haircut model), DFS turn-down & turn-up (event days),
  DSO windows (in-zone scenario).
- **Annual outer layer:** CM committed MW (scalar sweep).
- **Scenario switches:** genset on/off, Scotland/LCM siting, DSO in-zone/out.
- **Excluded:** intraday continuous, reactive/inertia/restoration, TRIAD
  (abolished) — justifications in doc 01 §11.

A modelling-fidelity note on speed classes: assume IT orchestration can deliver
within 2 min (Slow Reserve, BM, DFS comfortably; Quick Reserve only if a
frequency-/signal-triggered power-capping layer is assumed — make that an explicit
sensitivity, citing the Oracle Phoenix 25 % peak-hour reduction and Google
carbon-aware load-shaping as evidence such control loops exist in production).
