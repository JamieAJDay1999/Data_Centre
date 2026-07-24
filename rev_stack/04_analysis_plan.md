# 04 — Analysis Plan for Paper 2

## 1. Research questions

- **RQ1 (portfolio):** What is the value-maximal, deliverability-certified service
  portfolio for a 10 MW flexible DC in post-reform GB markets, and how does its
  composition vary by EFA block, day type and season?
- **RQ2 (attribution):** How does portfolio value decompose across assets (IT
  shifting, UPS, TES/cooling, optional gensets)? What is each asset's *marginal*
  value in the stack (leave-one-out), as opposed to standalone value — i.e. how
  large are the synergies (e.g. TES extending IT-led reduction durations)?
- **RQ3 (co-optimisation value):** How much value is lost by common heuristics —
  single-service dedication, greedy sequential stacking, ignoring the rebound —
  relative to full co-optimisation?
- **RQ4 (robustness):** How stable is the optimal stack under price-regime shifts
  (FR saturation, spread widening), utilisation-frequency assumptions, forecast
  error, and the UPS resilience floor?
- **RQ5 (policy):** What does facility flexibility buy beyond revenue — CM
  contribution, and acceptable firm-connection-capacity reduction under
  connections reform?

## 2. Positioning & novelty check

Done properly in [05_novelty_and_literature.md](05_novelty_and_literature.md)
(web-verified July 2026). One-line version: **no published study co-optimises a
whole-facility DC (IT + UPS + thermal) across a real market portfolio, and
nothing at all covers the post-2024 GB product set (EAC response, BR/QR/SR,
bi-directional DFS) for data centres; NESO itself excludes DC flexibility from
FES for lack of evidence.** The certification angle (deliverable-for-duration
offers) answers the feasibility gap the newest US work (Fan & Zhao 2026) raises
for a single service, generalised to a portfolio. Also deliberately correct the
stale literature: TRIAD-avoidance value no longer exists (TCR 2023), so older GB
DC-flexibility valuations are obsolete.

## 3. Case study: scaling 1 MW → 10 MW

Scale by replication (×10 racks) so the thermal model's per-rack physics is
untouched; recompute lumped capacitances/conductances via the existing helper
functions rather than multiplying end results.

| Parameter | 1 MW (current) | 10 MW case | Note |
|---|---|---|---|
| IT power (idle–max) | 166.7–1,000 kW | 1.667–10 MW | `n_racks` 100→1,000 |
| Overhead | 53.1 kW | 531 kW | keep 7 % rule |
| UPS energy / charge / discharge | 600 kWh / 270 kW / 2,700 kW | 6 MWh / 2.7 MW / 27 MW | check 27 MW is nameplate-plausible (10–15 min bridge sizing); *contracted* MW will be energy-capped anyway |
| UPS SoC floor | 50 % | 50 % (sensitivity 30/70 %) | resilience policy |
| TES / chiller | 1 MWh_th, 300 kW_th / 400 kW_e | 10 MWh_th, 3 MW_th / 4 MW_e | |
| Grid connection `P_conn` | n/a (new) | 12 MW firm (sensitivity: non-firm) | new parameter; enables RQ5 |
| Genset (scenario) | n/a | 12 MW, 2-min start, £250/MWh fuel, run-hour cap per MCPD/permit (set from the case-study permit class) | optional |

Why 10 MW: clears every product's 1 MW minimum standalone (no aggregator margin to
model; DFS is now 0.1 MW anyway), matches a typical GB colocation facility, small
enough to stay price-taker. 100 MW variant = pure ×10 sensitivity run (flag where
price-taker weakens: DC-L requirement is roughly 1–1.5 GW total, so 100 MW of UPS
response is no longer marginal — check requirement volumes in the NESO EAC results
when parameterising).

Workload inputs: reuse `load_profiles.csv` shape and the `dc_utilisation` real
traces; sensitivity on flexible share (Paper 1 already found this a top-2 driver).

## 4. Data assembly (Stage 0, the long pole — start immediately)

| Series | Source | Resolution | Use |
|---|---|---|---|
| DA prices (N2EX/EPEX GB), 2024–25 (and H1 2026) | Nord Pool / EPEX via [Elexon Insights](https://bmrs.elexon.co.uk/) (MID data) | HH/hourly | arbitrage price `π_DA` |
| EAC auction results: DC-L/H, DM, DR (+ QR/SR as migrated) | [NESO Data Portal](https://www.neso.energy/data-portal) — daily auction results incl. requirement volumes | EFA block | `π_av[j,b]` + market-depth check |
| Balancing Reserve clearing prices | NESO Data Portal | settlement period | `π_av[BR,h]` |
| Quick/Slow Reserve auction results | NESO Data Portal (PQR/NQR from Dec 2024 / Sep 2025; SR from launch) | window | `π_av` ; STOR history as SR backfill |
| System prices & BM acceptances for comparable DSR/BESS units | Elexon Insights (BOALF/DISBSAD); Modo research for benchmark cross-checks | HH | BM haircut κ_acc calibration |
| Grid frequency (1 s) | NESO Data Portal historic frequency data | 1 s → throughput factors | DR/DM/DC utilisation & UPS cycling |
| DFS: events, accepted volumes & prices, evolved-design rules | [NESO DFS](https://www.neso.energy/industry-information/balancing-services/demand-flexibility-service-dfs) + Data Portal | event | DFS layer (turn-down & turn-up from Apr 2026) |
| CM clearing prices & DSR derating | EMR Delivery Body registers; [Drax tracker](https://energy.drax.com/intelligence/capacity-market-auction-results/) for quick reference | annual | CM layer (T-4 £60–65/kW/yr recent; T-1 £20/kW/yr 2025/26) |
| DSO tenders (windows, availability/utilisation prices) | [Piclo Flex](https://www.piclo.com/) public tenders; DNO Ofgem procurement reports | window | DSO scenario |
| LCM bids/dispatch | NESO Data Portal / Piclo | event | Scottish scenario |
| DUoS red/amber/green bands (EHV site, case-study DNO) | DNO charging statement | ToU | residual network adder |

Deliverable: one tidy parquet/CSV pack + a `price_data.py` loader with a data
dictionary; every series carries its source URL and retrieval date (reviewers will
ask). Rules/prices were spot-verified July 2026 (doc 01); this stage turns those
into citation-grade, versioned inputs.

## 5. Modelling decisions to lock early (with recommendations)

1. **BM representation:** arbitrage against historical accepted prices with an
   acceptance haircut κ (calibrated from comparable units), not a dispatch
   simulation. Simple, defensible, sensitivity on κ. (Full BM simulation = future
   work.)
2. **Utilisation of response/reserve:** expected-energy factors φ[j] from
   historical frequency data / activation statistics, not scenario simulation, in
   Stages 1–2; scenario calls only in Stage 3.
3. **Settlement of delivered flexibility:** delivered energy deviations settle at
   the product's utilisation price; facility otherwise balanced. State clearly.
4. **Foresight:** Stage 2 uses perfect foresight day-ahead (upper bound, standard
   practice); Stage 3 quantifies the foresight premium with persistence/naïve
   forecasts. Report both — the gap is itself a result.
5. **Baseline integrity:** market baselines (DFS/DSO) assume the regulator-approved
   baseline equals our model's counterfactual `P(s)`. Note gaming literature;
   out of scope.

## 6. Experimental programme

**Stage 1 — Mechanics & representative days (first results, ~small compute).**
Cluster the two price years into 8–12 representative day types (k-medoids on
joint DA + DC-L + BR price shapes, seasonal stratification). Solve the
deterministic co-optimisation per day type. Outputs: optimal `r[j,b]` allocation
patterns; per-service and per-asset revenue; sanity checks against published
benchmarks (Modo GB BESS index: **~£73k/MW/yr** for a 2-h system over the 12
months to Apr 2026, monthly range ~£59–77k in 2025, FR share ~20–33 % — our UPS
slice should be consistent per usable MW after the resilience-floor haircut).

**Stage 2 — Annual simulation (headline numbers).**
365 daily solves × 2 years, rolling state (UPS/TES end-of-day carryover, IT
recovery spillover via the existing 12-slot extension). CM outer sweep over
`C_CM`. Deliverables: annual revenue stack (£/yr and £/MW/yr) with cost baseline
from Paper 1's nominal case; stack composition heatmaps (product × EFA block ×
season); duration-certification pass rates.

**Stage 3 — Uncertainty & risk.**
Two-stage stochastic with 10–20 scenarios; CVaR frontier (λ sweep); foresight
premium; utilisation-shock stress (φ ×2, ×5).

**Stage 4 — Sensitivities & scenarios (each = one bar in a tornado chart).**
- FR saturation: EAC prices ×{0.5, 1, 2}; spread widening: DA volatility ×{1, 1.5, 2}
  (reuse the mean-preserving-spread machinery from
  `sensitivity_analysis/sensitivity_sweep.py`).
- Flexible workload share ×{0.5, 1, 1.5}; UPS SoC floor {30, 50, 70} %.
- Genset on/off; DSO in-zone/out; Scotland siting (LCM on).
- Scale: 10 vs 100 MW; asset sizing: UPS energy ×{1, 2, 4} (would you *build*
  bigger UPS for revenue? — capex break-even calc), TES ×{0.5, 1, 2}.
- Connection: `P_conn` ∈ {10.5, 12, 14} MW and a non-firm-connection case
  (curtailment windows) → RQ5.

**Benchmark heuristics for RQ3:** (a) energy-only (Paper 1 as-is), (b) single
best service dedication per asset, (c) greedy sequential (commit FR first, then
arbitrage residual), (d) full co-optimisation, (e) co-optimisation without
certification (count delivery failures — the "phantom flexibility" a naive model
would sell). Value ladder a→d is the co-optimisation result; d vs e is the
certification result.

## 7. Expected headline figures (design the paper backwards)

1. Annual revenue stack bar: baseline energy cost → + arbitrage → + FR → +
   reserve → + BM → + CM → + DFS/DSO, per configuration (the "money chart").
2. Stack composition heatmap: committed MW by product × EFA block × season.
3. Asset attribution: leave-one-out marginal values + synergy matrix (RQ2).
4. Value-of-co-optimisation ladder (RQ3) incl. phantom-flexibility count.
5. Magnitude–duration envelope (from Paper 1) overlaid with product (τ_j, MW)
   requirement boxes — the visual bridge between the two papers.
6. Risk–return frontier (CVaR) + tornado chart of sensitivities.
7. Firm-capacity-reduction vs cost curve (connections-reform policy figure).

## 8. Paper outline (target: Applied Energy / IEEE TSG / Advances in Applied Energy)

1. Introduction — DC growth, GB reform wave (REMA settled, EAC, MHHS, connections
   queue), gap: whole-facility stacking with certified deliverability.
2. GB market framework for demand-side stacking (condensed doc 01 + stacking
   rules table — a citable reference table in its own right).
3. Methodology — recap Paper 1 model (brief, cite); stacking formulation
   (doc 03); certification loop; uncertainty treatment.
4. Case study & data — 10 MW facility, 2024–25 GB prices.
5. Results — figures §7, Stages 1–4.
6. Discussion — operator playbook, market-design observations (product
   granularity vs DC capabilities, resilience-floor economics, saturation),
   connections/policy angle, limitations (price-taker, baseline integrity,
   perfect foresight bound).
7. Conclusion.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| FR prices so saturated the stack is boring ("arbitrage is 90 % of value") | that *is* a finding — it mirrors the documented BESS evolution (FR 80 %→20–33 % of revenue); emphasise QR/SR/BR/DFS-2026 (new, unsaturated, unquantified in literature) + RQ4 regime sweeps |
| Reserve products too new → thin price history | backfill with STOR/legacy analogues, clearly labelled; scenario prices |
| Market rules mis-stated (fast-moving) | rules spot-verified July 2026 with dated primary sources (doc 01); keep a rules-version appendix; re-verify at write-up |
| Scooped by fast-moving US arXiv work or NESO's own DC innovation project | monthly arXiv sweep; track NESO "Options for Optimising GB Data Centres" outputs; the GB-products + whole-facility + certification triple remains defensible (doc 05 §4) |
| MILP runtime for stochastic stage | binaries→continuous relaxation where valid; parallel days; SCIP → HiGHS/Gurobi if needed |
| Reviewer: "why no intraday/real-time?" | scoped as extension; rolling-horizon Stage 2 captures re-optimisation flavour |
| Double-counting revenues (classic stacking-paper flaw) | headroom + energy-reservation constraints by construction; certification loop; explicitly demonstrate no simultaneous double commitment in a results appendix |

## 10. Work plan (indicative, ~9 months to submission)

| Months | Work |
|---|---|
| 1–2 | Stage 0 data pack; verify all ⚠ items; freeze product/rule table |
| 2–4 | Implement `rev_stack/model/` (doc 03 §8); Stage 1 representative days; internal review of mechanics |
| 4–6 | Stage 2 annual runs + CM sweep; certification loop; benchmark validation |
| 6–7 | Stage 3 stochastic/CVaR; Stage 4 sensitivity programme |
| 7–9 | Figures, writing, co-author review, submission |

**First concrete task:** build `market_parameters.py` with the product table
(names, directions, τ_j, speed class, min MW, price source) — it forces every
⚠ verification and is the single source of truth for both the model and the
paper's Table 1.
