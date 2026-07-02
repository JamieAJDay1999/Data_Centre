# 05 — Novelty Assessment: Is This a Good Research Direction?

**Short verdict: yes — this is a genuinely novel and well-timed contribution,
provided it is framed carefully against a fast-moving field.** The specific
combination — *whole-facility* data-centre model (IT + UPS + cooling/TES),
*post-reform GB* product set, *duration-certified* portfolio bidding — does not
exist in the literature as of July 2026. The main risks are pace (US-focused
arXiv papers are appearing monthly) and framing (it must not read as "BESS
revenue stacking with extra steps"). Both are manageable; details below.

## 1. What the adjacent literature actually covers

### 1.1 BESS revenue stacking (mature, the methodological neighbour)
- Revenue stacking of batteries across energy + ancillary markets is
  well-studied, including GB-specific MILP dispatch against DC/DM/DR and
  wholesale (e.g. [behind-the-meter BESS stacking, EPSR 2022](https://www.sciencedirect.com/science/article/pii/S0378779622004825);
  [UK market mechanism as a tool for BESS dispatch, Renewable Energy 2024](https://www.sciencedirect.com/science/article/abs/pii/S0960148124010255);
  [stacked energy+reserve revenues with FR modelling, Applied Energy 2023](https://www.sciencedirect.com/science/article/abs/pii/S0306261923010851)).
- Industry analytics (Modo Energy) publish GB BESS benchmarks monthly
  (~£73k/MW/yr for 2-h systems, 12 months to Apr 2026; FR share collapsed from
  ~80 % in 2022 to ~20–33 %).
- **What they don't do:** a battery is one asset with one state variable. None of
  this literature handles a *portfolio of heterogeneous assets behind one meter*
  where the "storage" includes deferrable computation with job-completion
  (rebound) constraints and a thermal plant with temperature-bounded dynamics.
  The deliverability question — is a committed MW actually sustainable for the
  product duration given coupled IT/thermal/UPS states? — is trivial for a
  battery (energy ÷ power) and genuinely hard for a DC. That difficulty is
  exactly what Paper 1 solved, and Paper 2 monetises.

### 1.2 Data-centre demand response / flexibility (large, but not this)
- Classic DC demand-response literature (Wierman, Liu, Rahman et al., mostly
  2013–2019, US markets) covers workload shifting for price response and single
  ancillary products; geo-distributed load balancing; PJM regulation with
  server-level control.
- Recent wave (2024–26), driven by AI load growth:
  - [Fan & Zhao, arXiv Feb 2026](https://arxiv.org/abs/2602.01508): day-ahead
    co-optimisation of geo-distributed workload + *frequency regulation capacity*
    with chance/VaR queue constraints on an IEEE 68-bus system. Closest recent
    methodological neighbour. **Single service (regulation), no UPS/cooling/TES,
    synthetic system, not GB.** Notably, its motivating gap — "most existing
    methods treat workload scheduling and regulation capacity bidding separately,
    resulting in potentially infeasible commitments" — is precisely our
    certification argument, generalised by us to a *portfolio* of products.
  - [Chen & Zheng, arXiv Apr 2026](https://arxiv.org/abs/2604.05376): AI-DC
    flexibility in capacity-expansion planning (3–21 % system-cost reduction).
    Planning-level, generic grid — complementary, not competing.
  - [Grid frequency stability support potential of DCs, arXiv Oct 2025](https://arxiv.org/html/2510.01050v1):
    physical potential quantification, not market revenue.
  - [Energy storage dispatch/design for DC grid services, Energy 2025](https://www.sciencedirect.com/science/article/pii/S0360544225001537):
    ESS-centric, not whole-facility market portfolio.
  - An older near-miss: [multi-stage DC scheduling with chilled-water storage in
    energy + regulation markets, arXiv 2020](https://arxiv.org/pdf/2007.09770) —
    two US products, no UPS, no duration certification.
- Reviews (e.g. *Data centres as a source of flexibility for power systems*,
  Energy Reports 2025) explicitly flag UPS participation in frequency response
  and market-integration analysis as underdeveloped — i.e. the reviews are
  *asking for* this paper.

### 1.3 Industry/policy activity (validation, and the audience)
- **EPRI DCFlex** (40+ members incl. Google, Meta, Microsoft, PJM) is field-testing
  exactly our asset taxonomy (compute / balance-of-plant / power assets incl.
  UPS) — US-centric, demonstration-oriented, no open co-optimisation model
  ([ITIF overview](https://itif.org/publications/2025/11/24/united-states-needs-data-centers-data-centers-need-energy-but-that-is-not-necessarily-a-problem/)).
- **Microsoft/Eaton Dublin**: grid-interactive UPS delivering fast frequency
  response in EirGrid's market via Enel X — commercial proof of the UPS asset
  class ([DCD](https://www.datacenterdynamics.com/en/news/microsoft-teams-up-with-eaton-for-grid-interactive-ups-at-dublin-data-center/)).
- **GB policy pull**: NESO FES 2025 *excludes* DC flexibility for lack of
  evidence; NESO runs an innovation project on optimising GB DCs; Ofgem's Flex
  Technical Taskforce + the AI Energy Council are designing
  faster-connections-for-flexible-DCs; ~50 GW of DCs sit in the connection queue.
  A peer-reviewed GB quantification lands directly into that evidence gap
  ([ECA](https://www.eca-uk.com/2025/09/23/is-there-a-data-centre-sized-hole-in-flexibility-forecasts/),
  [GOV.UK consultation](https://www.gov.uk/government/consultations/accelerating-electricity-network-connections-for-strategic-demand/accelerating-electricity-network-connections-for-strategic-demand-accessible-webpage)).

## 2. The gap, stated precisely

> No published study co-optimises a physically-validated whole-facility data
> centre model — deferrable IT workload with completion/rebound constraints,
> resilience-constrained UPS, and temperature-bounded cooling with thermal
> storage — across a *portfolio* of products in any real market, let alone the
> **post-2024 GB product set** (EAC dynamic response, Balancing/Quick/Slow
> Reserve, BM via OBP, bi-directional zonal DFS, Capacity Market, DSO services).
> Existing GB data-centre valuations predate the reform wave and typically
> monetise TRIAD avoidance, which no longer exists. Moreover, no DC-flexibility
> study guarantees that stacked commitments are *deliverable for each product's
> required duration from every callable state* — the "phantom flexibility"
> problem that single-asset battery models don't face and single-service DC
> models only note in passing.

## 3. Contributions (as they should appear in the paper)

1. **First post-reform GB revenue-stacking formulation for a whole-facility data
   centre**, co-optimising energy arbitrage with response, reserve, BM, DFS, CM
   and DSO commitments under shared-headroom and energy-reservation constraints —
   with a citable, dated codification of the 2024–26 GB demand-side product set
   and its stacking rules (useful to practitioners in its own right).
2. **Duration-certified market offers**: a certification loop that converts
   Paper 1's magnitude–duration flexibility envelope into feasible bid sets, and
   quantifies the phantom-flexibility error of uncertified stacking (bridging the
   feasibility gap Fan & Zhao identify, but across a heterogeneous portfolio).
3. **Asset–service attribution**: marginal (leave-one-out) values and synergy
   quantification for IT shifting, UPS, TES/cooling and optional gensets — the
   first such decomposition for a DC in real market prices; directly answers
   "which assets should participate in what".
4. **Policy quantification for the connections debate**: the revenue stack
   re-expressed as the firm-capacity reduction a flexible DC could accept at
   given cost — evidence for the NESO FES gap and the Ofgem/AI Energy Council
   flexible-connections agenda.

## 4. Honest risk assessment

| Risk | Severity | Handling |
|---|---|---|
| A similar US/China arXiv paper appears before submission | Medium-high (monthly cadence in this space) | GB product set + whole-facility physics + certification is a defensible triple; move fast (9-month plan, doc 04 §10); check arXiv monthly and the NESO innovation project outputs |
| "It's just BESS stacking applied to a DC" (reviewer) | Medium | lead with what batteries don't have: rebound constraints, thermal coupling, resilience floor, certification; include the phantom-flexibility result (d vs e in doc 04 §6) as the headline methodological figure |
| FR saturation makes the stack arbitrage-dominated and "boring" | Low | that *is* a publishable finding (matches the BESS market's own evolution); the QR/SR/BR/DFS-2026 products are new enough that quantifying them at all is a first |
| Market rules churn between now and publication | Medium | date-stamp the rules table; parameterise products in `market_parameters.py`; frame method as product-agnostic |
| Estimated revenues rest on price-taker + perfect-foresight assumptions | Medium | report the foresight premium explicitly (Stage 3); benchmark the UPS slice against Modo BESS indices for sanity |

## 5. Venue and framing recommendation

- **Venue:** Applied Energy or Advances in Applied Energy (systems + market +
  policy mix; Paper 1's audience), IEEE Transactions on Smart Grid if the
  certification/optimisation angle is led. Given the policy salience, Applied
  Energy with a strong policy-implications section is the better fit.
- **Framing:** lead with the GB evidence gap (NESO excludes DC flexibility from
  FES for lack of evidence; 50 GW of DCs queueing) → "here is the missing
  evidence, produced with a certified-deliverability method." That framing makes
  the paper's value robust even if the revenue numbers themselves shift with
  market conditions.
