# 01 — The GB Market Landscape for Data-Centre Flexibility (post-2023 reforms)

Everything a GB data centre could sell flexibility into, verified against NESO /
Elexon / Ofgem / GOV.UK / industry sources as of **July 2026** (retrieval date
2026-07-02; source links at each section — re-check service terms at
implementation time, several products are still evolving).

Direction convention (matches Paper 1): **upward flexibility = demand turn-down**
(helps the system when short of generation / low frequency); **downward
flexibility = demand turn-up** (absorbs surplus / high frequency).

---

## 1. Wholesale energy markets

### 1.1 Day-ahead (DA) auctions
- Hourly/half-hourly auctions via N2EX / EPEX Spot GB. For a load, "arbitrage"
  means shifting purchase volume between periods — exactly what `optimisation.py`
  already does against a synthetic ToU tariff. Paper 2 replaces
  `generate_tariff()` with historical DA clearing prices.
- **Access:** via supplier (flex/pass-through contract) or independently. BSC
  Modification **P415** (implemented **7 Nov 2024**) lets **Virtual Lead Parties**
  trade customer flexibility in the wholesale market without being the supplier,
  creating the "Virtual Trading Party" route
  ([Elexon P415](https://www.elexon.co.uk/bsc/mod-proposal/p415/),
  [Elexon news](https://www.elexon.co.uk/2024/11/07/elexon-implements-landmark-rule-change-to-drive-growth-in-flexibility-and-progress-to-net-zero/)).
- **Modelling:** exogenous price series `π_DA(t)`; existing objective structure
  carries over.

### 1.2 Intraday continuous / within-day
- Exclude from the core model (continuous-trading microstructure is a paper in
  itself); the rolling-horizon annual simulation captures re-optimisation flavour.
  Note as extension.

### 1.3 Imbalance exposure
- Deviations settle at the system price (Elexon). Assume the facility is balanced
  apart from instructed/paid flexibility; state this assumption.

## 2. NESO frequency response — Dynamic services via the EAC

All three dynamic services are procured **day-ahead in co-optimised auctions on
the Enduring Auction Capability (EAC)** platform, in 4-hour EFA blocks, split
into low (under-frequency = demand turn-down) and high (over-frequency = demand
turn-up) variants, **minimum 1 MW**, availability-paid £/MW/h, pay-as-clear;
clearing prices can go negative
([NESO EAC](https://www.neso.energy/industry-information/balancing-services/enduring-auction-capability-eac),
[NESO Dynamic services](https://www.neso.energy/industry-information/balancing-services/frequency-response-services/dynamic-services-dcdmdr)).
The EAC co-optimises across services and lets a unit split MW across products per
block — institutionally, the market already does what our optimiser will do
internally.

| Service | Speed | Delivery duration | Character | DC-asset fit |
|---|---|---|---|---|
| **Dynamic Containment (DC-L/DC-H)** | 0.5 s initiate, **1 s full delivery** | **15 min at full output** (energy-limited assets) | post-fault containment; low expected throughput | grid-interactive UPS — the flagship UPS product |
| **Dynamic Moderation (DM)** | 0.5 s initiate, full ≤ 10 s | 15–30 min class | pre-fault, moderate throughput | UPS; possibly fast IT power capping |
| **Dynamic Regulation (DR)** | 2 s initiate, ~10 min to full | up to **60 min**; continuous tracking | high energy throughput | poor fit for a resilience-floored UPS; include so the optimiser can reject it |

([NESO DC service doc](https://www.neso.energy/document/173206/download),
[DC guidance](https://www.neso.energy/document/175296/download))

- **Feasibility precedent:** Microsoft + Eaton run grid-interactive
  ("EnergyAware") UPS at the Dublin data centre providing fast frequency response
  in EirGrid's DS3 market via Enel X — commercial proof the UPS asset class works
  ([DCD](https://www.datacenterdynamics.com/en/news/microsoft-teams-up-with-eaton-for-grid-interactive-ups-at-dublin-data-center/),
  [Eaton/Microsoft whitepaper](https://www.eaton.com/content/dam/eaton/markets/data-center/eaton-microsoft-grid-interactive-whitepaper-wp153031en.pdf)).
- **Saturation is the defining price fact:** GB frequency response went from
  ~80–87 % of BESS revenue (2020–22) to ~20–33 % now; FR revenues for a typical
  BESS were ~£23k/MW/yr in Nov 2025. Ancillary markets are small vs wholesale
  and saturate once a competitive fleet qualifies
  ([Modo Nov 2025](https://modoenergy.com/research/en/me-bess-gb-battery-energy-storage-revenues-november-2025-balancing-mechanism-gas-wind),
  [Rabobank](https://www.rabobank.com/knowledge/d011469493-backup-power-for-europe-part-2-the-uk-s-bess-leadership-and-evolving-revenue-stacks)).
  The model must be calibrated to 2024–26 prices, not 2021–22 folklore; and the
  paper should expect **arbitrage/BM-led stacks with FR as seasoning** — matching
  the current BESS benchmark (~£73k/MW/yr for a 2-h system, 12 months to Apr 2026,
  [Modo](https://modoenergy.com/research/en/how-does-battery-energy-storage-make-money)).

## 3. NESO reserve — the 2024–25 reform family

NESO's Markets Roadmap replaced legacy STOR/Fast Reserve with a new family,
migrating into the EAC:

| Service | Status | Speed / duration | Procurement & payment | Notes |
|---|---|---|---|---|
| **Balancing Reserve (BR±)** | live **March 2024** | delivered via BM instructions | day-ahead pay-as-clear **per settlement period / service window**, £/MW/h availability | requires BM unit (BMU or VLP); positive & negative variants ([NESO BR](https://www.neso.energy/industry-information/balancing-services/reserve-services/balancing-reserve), [service terms](https://www.neso.energy/document/369846/download)) |
| **Quick Reserve (PQR/NQR)** | Phase 1 (BM) live **Nov/Dec 2024**; Phase 2 (non-BM) first auction **2 Sep 2025** | **full delivery ≤ 1 min** (TTFD), 1-s metering | day-ahead auction, availability + utilisation | min **1 MW**; positive and negative variants ([NESO QR](https://www.neso.energy/industry-information/balancing-services/reserve-services/quick-reserve), [Ofgem Phase 1 decision](https://www.ofgem.gov.uk/sites/default/files/2024-10/Quick%20Reserve_phase_1_decision.pdf)) |
| **Slow Reserve (SR)** | design v2.1 July 2025; EAC integration underway (Dec 2025 industry update); replaces STOR | **TTFD 15 min**, 15-s metering, sustained hour-scale delivery | daily auction, min **1 MW** | the natural product for orchestrated IT workload deferral ([design proposal](https://www.neso.energy/document/366106/download), [NESO SR](https://www.neso.energy/industry-information/balancing-services/reserve-services/slow-reserve)) |
| **STOR** (legacy) | being phased out | 20 min response, ≥ 2 h duration | day-ahead auction | use only for historical backfill of SR prices |

- Reserve is **exclusive with response for the same MW in the same period** but
  stackable across periods and across different MW of the same facility — handled
  with shared-headroom constraints, not binaries.
- Note BR's settlement-period granularity vs the EFA-block granularity of
  response: the formulation indexes commitments accordingly (doc 03 §1).

## 4. Balancing Mechanism (BM)

- Pay-as-bid bids/offers per settlement period; for a load, an *offer* =
  turn-down (paid £/MWh), a *bid* = turn-up. No availability payment.
- **Access:** secondary BMU or **VLP** (Wider Access, min 1 MW). NESO's **Open
  Balancing Platform** (live Dec 2023, bulk dispatch of small/battery units) has
  materially raised small-unit dispatch rates — BM is now the *largest single
  revenue line* for GB batteries in several 2025 months
  ([Modo Oct 2025](https://modoenergy.com/research/en/battery-energy-storage-revenues-gb-october-2025-record-balancing-mechanism-dispatch-rates)).
- **Modelling:** arbitrage against historical accepted prices with an acceptance
  haircut κ calibrated from comparable units (doc 04 §5). Full dispatch
  simulation = future work.

## 5. Demand Flexibility Service (DFS) — now year-round and bi-directional

The DFS has evolved well beyond the winter-crisis service in older literature
([NESO DFS](https://www.neso.energy/industry-information/balancing-services/demand-flexibility-service-dfs),
[NESO announcement](https://www.neso.energy/neso-announces-shakeup-new-look-demand-flexibility-service)):

- **Ofgem approved the evolved design (March 2026)** with a derogation for
  **year-round use until 31 March 2027**.
- From **April 2026**: **bi-directional** (demand turn-up *and* turn-down),
  **zonal procurement**, eligibility threshold reduced to **0.1 MW**, a
  **Primacy process** (managing conflicts with other services), and a
  **Self-Nominated baseline** option aimed at I&C assets.
- Winter 2024/25 scale: ~11 GWh bid, 5.4 GWh accepted, ~£0.94 M paid — small
  money, but the 2026 redesign (zonal + turn-up + I&C baselines) makes it a
  genuine option for a DC, including for absorbing surplus (the LCM direction)
  outside Scotland.
- Sits inside NESO's **Demand-side Flexibility Routes to Market Review**
  (Dec 2025) — cite as evidence the demand-side product set is actively being
  rationalised ([review](https://www.neso.energy/industry-information/flexibility/demand-side-flexibility-routes-market-review)).
- **Modelling:** event-based utilisation revenue with the Primacy rules encoded
  as headroom exclusivity on event periods.

## 6. Capacity Market (CM)

- £/kW/yr for derated capacity via T-4/T-1 auctions; proven DSR CMUs participate
  (2.9 GW derated DSR in the T-1 2026/27 prequal). Recent clearing prices:
  **T-4: £63/kW/yr (DY 2026/27), £65 (DY 2027/28), £60 (DY 2028/29)**;
  **T-1: £20/kW/yr (DY 2025/26)** — a five-year T-1 low
  ([Drax auction tracker](https://energy.drax.com/intelligence/capacity-market-auction-results/),
  [Argus](https://www.argusmedia.com/en/news-and-insights/latest-market-news/2664901-uk-t-1-capacity-market-auction-clears-at-five-year-low),
  [DESNZ parameters](https://www.gov.uk/government/publications/capacity-market-auction-parameters-letter-from-desnz-to-neso-february-2026/final-auction-parameters-t-1-and-t-4-capacity-market-auctions)).
- Obligations: deliver during System Stress Events (rare) + satisfy testing; DSR
  derating factors apply (look up current factor in the auction guidelines when
  parameterising).
- **Modelling:** annual **exogenous layer** — commit `C_CM` MW, add a standing
  deliverability constraint (reuse the Paper-1 envelope), add revenue
  `C_CM × derating × price`, sweep the scalar `C_CM` in an outer loop.

## 7. Distribution-level (DSO) flexibility

- All six GB DSO groups run regular tenders (mostly via Piclo Flex) using the
  **common product set agreed across DSOs in 2023** (≥ 80 % of tendered volume
  through common products since summer 2024), with **availability + utilisation
  payments** and converging baselining/settlement methodologies
  ([ENA Open Networks](https://www.energynetworks.org/industry/flexibility-services),
  [ENWL procurement report 2025](https://www.enwl.co.uk/globalassets/future-energy/flexibility-hub/document-library/ofgem-dfp-reporting/distribution-flexibility-services-procurement-report--2025.pdf),
  [Piclo](https://www.piclo.com/)).
- Zonal: value exists only in constrained zones — treat as a scenario
  (in-zone vs out). Windows are typically winter evening peaks, colliding with
  high wholesale prices; the co-optimiser resolves the trade-off (good figure).
- Primacy/conflict rules between DSO and NESO services are being formalised
  (see DFS Primacy above); encode as no-simultaneous-conflicting-dispatch.

## 8. Local Constraint Market (LCM)

- NESO's B6-boundary constraint product (launched 2023, **extended to January
  2027**): pays **demand turn-up** in Scotland to absorb constrained wind;
  asset-level metering allowed; strong provider growth through 2025 (7× Q1→Q2)
  ([NESO LCM](https://www.neso.energy/industry-information/balancing-services/local-constraint-market),
  [extension news](https://www.neso.energy/news/local-constraint-market-extended-january-2027)).
- Turn-up = advance IT load + pre-cool + charge UPS — the direction Paper 1
  found constrained but non-zero. Include as a **Scottish-siting scenario**;
  note NESO's "Demand for Constraints" work as the enduring successor signal.

## 9. Network charges — mostly *removed* as a revenue stream (important negative result)

- **TRIAD avoidance is dead:** the Targeted Charging Review moved demand TNUoS
  residual to fixed banded charges (April 2023). Peak-avoidance revenue that
  dominated older GB DC-flexibility papers **no longer exists** — say so
  explicitly; much of the literature is stale on this.
- **BSUoS** is a fixed volumetric charge (2023): not avoidable by shifting.
- **DUoS** red/amber/green bands persist for HV/EHV forward-looking charges:
  keep a small time-varying adder (DNO-specific; take the relevant charging
  statement for the case-study zone).

## 10. The reform backdrop (framing for the introduction)

1. **REMA settled (10 July 2025):** zonal pricing rejected; **reformed national
   pricing** retained, with a Delivery Plan due end-2025 (SSEP, CSNP, connections
   and network-charging reform)
   ([Norton Rose summary](https://www.nortonrosefulbright.com/en/knowledge/publications/4399413b/rema-summer-update-no-to-zonal-pricing-yes-to-reformed-national-pricing)).
   Consequence: no zonal energy-price uncertainty in the core model; locational
   value flows via DSO/LCM/zonal-DFS products.
2. **EAC + reserve reform:** co-optimised day-ahead procurement of response (and
   progressively reserve) — the institutional justification for a
   *co-optimisation* formulation.
3. **MHHS:** market-wide half-hourly settlement rollout makes HH price exposure
   the default for all demand.
4. **OBP + Wider Access + P415:** small/aggregated DSR dispatchable and
   wholesale-enabled.
5. **Connections reform is the policy hook.** The GB demand queue tripled to
   **125 GW by June 2025, of which ~140 data centres ≈ 50 GW** (> GB peak demand)
   ([NESO](https://www.neso.energy/neso-implements-electricity-grid-connection-reforms-unlock-investment-great-britain),
   [Hogan Lovells](https://www.hoganlovells.com/en/publications/connecting-data-centres-to-the-electricity-networks)).
   Ofgem's **TM04+** (approved April 2025) makes access merit-based (Gate 2 offers
   through 2026); DESNZ is consulting on **accelerating connections for strategic
   demand**; the **Connections Accelerator Service** (pilot Dec 2025) explicitly
   supports flexible alternatives (non-firm, ramped, profiled connections); and
   Ofgem's **Flex Technical Taskforce**, alongside the government's **AI Energy
   Council**, is developing arrangements where **data centres that commit to
   operate flexibly connect sooner**
   ([GOV.UK consultation](https://www.gov.uk/government/consultations/accelerating-electricity-network-connections-for-strategic-demand/accelerating-electricity-network-connections-for-strategic-demand-accessible-webpage),
   [NatLawReview](https://natlawreview.com/article/new-race-power-what-ofgems-grid-reform-means-data-centre-development-great-britain)).
   → Paper 2 should quantify flexibility as a *connection accelerant*, not just a
   revenue line (doc 04, RQ5).
6. **The evidence gap is officially acknowledged:** NESO's FES 2025 projects DC
   demand 7.6 → 33 TWh by 2035 but **explicitly excludes DC demand flexibility
   for lack of evidence** ([ECA commentary](https://www.eca-uk.com/2025/09/23/is-there-a-data-centre-sized-hole-in-flexibility-forecasts/));
   NESO runs an innovation project, *Options for Optimising GB Data Centres*
   ([NESO innovation](https://www.neso.energy/about/innovation/our-innovation-projects/options-optimising-gb-data-centres))
   — check its outputs before submission and position against them.

## 11. Excluded services (record the justification)

| Service | Why excluded |
|---|---|
| FFR / legacy firm response | closed, replaced by DC/DM/DR |
| Mandatory FR, Obligatory Reactive Power | generator licence obligations |
| Inertia / stability pathfinders | bespoke tenders; grid-forming UPS is plausible future work — mention |
| Restoration / black start | bespoke; on-site gen too small |
| TRIAD | abolished (TCR) — state explicitly |
| Intraday continuous | scope; rolling horizon proxies it |
