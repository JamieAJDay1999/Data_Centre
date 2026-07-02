# Paper 2 — Revenue Stacking for a Flexible Data Centre in the GB Electricity Markets

This folder contains the full plan for the second paper, building directly on the
whole-facility flexibility model in this repository (Paper 1: *Characterisation and
Quantification of Data Centre Flexibility for Power System Support*). Paper 1's
conclusion explicitly names revenue stacking as the next step; this plan turns that
into a concrete research design.

## The core idea

Paper 1 produces two things this paper monetises:

1. A **cost-optimal baseline schedule** for the integrated IT + UPS + TES/cooling
   facility (`optimisation.py`).
2. A **duration-aware flexibility envelope** — for any start time and power
   deviation, the maximum feasible duration (`flexibility_duration.py`).

Paper 2 reframes the envelope as a **tradeable product set**: the facility
co-optimises its energy purchase (wholesale arbitrage) with capacity commitments
into NESO frequency-response and reserve markets, the Balancing Mechanism, the
Capacity Market, and DSO flexibility services — subject to the same physical model,
plus delivery-guarantee constraints that ensure every committed MW is deliverable
for the product's required duration from any point in the committed window.

**Headline research question:** *What is the value-maximal, deliverable portfolio of
market services for a 10 MW flexible data centre in post-reform GB electricity
markets, and how does the optimal stack decompose across the facility's assets?*

The honest answer to "what is the ideal stack?" is that it is **time-varying and
price-regime-dependent**, so the paper's contribution is not a single fixed stack
but (a) a co-optimisation formulation that *finds* the optimal stack endogenously
per EFA block and per season, and (b) an analysis of the structure of that optimum
(which assets back which services, when, and why) plus its robustness. A likely
result, stated up-front as a hypothesis to test: **UPS → dynamic frequency response
(energy-limited, power-rich); IT workload shifting → wholesale/BM arbitrage, DFS
and Capacity Market; TES/cooling → duration extension of IT-led reductions and
downward (turn-up) flexibility; optional standby generation → Capacity Market and
Slow Reserve.** Given the documented saturation of GB frequency-response markets
(FR fell from ~80 % to ~20–33 % of battery revenues by 2025), expect an
arbitrage/BM-led stack with response as a minority share — mirroring the BESS
market's own evolution, and itself a publishable calibration point.

## Documents

| File | Contents |
|---|---|
| [01_gb_market_landscape.md](01_gb_market_landscape.md) | Catalogue of every GB market/service a DC could plausibly access — web-verified July 2026 with sources — covering technical requirements, remuneration, and the 2023–26 reforms that matter (EAC, Balancing/Quick/Slow Reserve, the evolved bi-directional DFS, P415, MHHS, TCR, REMA outcome, TM04+/connections reform). |
| [02_asset_service_mapping.md](02_asset_service_mapping.md) | Asset × service suitability matrix for the four modelled assets (+ standby generation as a proposed fifth), stacking/exclusivity rules incl. the DFS Primacy process, and the shortlist of services to model explicitly vs. treat exogenously vs. exclude. |
| [03_mathematical_formulation.md](03_mathematical_formulation.md) | The revenue-stacking MILP: sets, variables, constraints and objective, written as an extension of the existing Pyomo model (`optimisation.py` / `constraints.py`), with two delivery-guarantee variants (embedded worst-case call vs. precomputed envelope from Paper 1) and a certification loop. |
| [04_analysis_plan.md](04_analysis_plan.md) | Research questions, scaling to 10 MW, data requirements with live sources, experimental design (representative days → annual rolling horizon → stochastic/CVaR), sensitivity programme, expected results/figures, paper outline, risks, and work plan. |
| [05_novelty_and_literature.md](05_novelty_and_literature.md) | Literature review (BESS stacking, DC demand response, the 2024–26 AI-load wave, industry/policy activity), a precise gap statement, the four contributions, an honest risk assessment, and venue/framing recommendation. |

## Recommended starting scope (decision summary)

- **Facility scale:** 10 MW IT load (parameters scaled ×10 from the 1 MW case;
  scaling table in doc 04). This clears the 1 MW minimum-entry threshold for every
  NESO product on its own, without an aggregator, while staying below the size
  where the DC would distort local prices — and a 100 MW case is a one-line
  sensitivity.
- **Services modelled endogenously (co-optimised):** day-ahead wholesale arbitrage,
  Dynamic Containment (low & high), Dynamic Moderation, Dynamic Regulation,
  Balancing Reserve (±), Quick Reserve (positive & negative), Slow Reserve,
  Balancing Mechanism bids/offers, Demand Flexibility Service events (turn-down
  *and* turn-up under the April 2026 evolved design), DSO availability windows.
- **Services layered exogenously (fixed £/kW/yr, feasibility-checked):** Capacity
  Market (DSR CMU), because commitment is annual not daily.
- **Excluded, with justification recorded:** intraday continuous trading (data
  burden; note as extension), reactive power, inertia/stability, restoration/black
  start, TRIAD avoidance (abolished by the Targeted Charging Review).

## Novelty (one paragraph; full assessment in doc 05)

No published study co-optimises a whole-facility DC (IT + UPS + cooling/TES)
across a real market portfolio, and nothing covers the post-2024 GB product set
for data centres at all. NESO itself excludes DC flexibility from its Future
Energy Scenarios for lack of evidence, while ~50 GW of data centres queue for
connections and Ofgem designs faster connections for flexible DCs — so the paper
lands into an explicitly acknowledged evidence gap with a strong policy audience.
The certification loop (duration-certified offers from Paper 1's envelope) is the
methodological differentiator against both BESS-stacking and US DC-regulation
literature.

## Immediate next steps

1. Assemble the 2024–26 price dataset (doc 04 §4 — sources linked there).
2. Build `market_parameters.py` (the product table = the paper's Table 1).
3. Implement Stage 1 of the formulation (deterministic, single day, availability-only)
   as `rev_stack/model/` reusing `constraints.py` unchanged where possible.
4. Re-verify Slow Reserve status and check outputs of NESO's "Options for
   Optimising GB Data Centres" innovation project before the writing stage.
