# Paper methodology amendment plan

## 1. Scope

This plan covers the revisions needed to align `paper/main_new.tex` with the implemented annual rolling-horizon model and the completed Scenario 2 sensitivity design.

It intentionally does **not** attempt to:

- replace numerical results, tables, or result-dependent claims;
- revise the Scenario 3 flexibility-duration case study;
- rewrite the discussion or conclusion; or
- decide how any new numerical findings should ultimately be interpreted.

Those items should be handled after the modelling description has been corrected and the final result set has been agreed.

The main editorial principle should be that the paper describes the model that was actually run. Where the paper and code currently differ, the discrepancy must either be resolved in the code first or explicitly recorded as an assumption in the revised method.

## 2. Overall assessment

The title, research motivation, system architecture, IT-workload background, and most of the literature review remain suitable. The substantial revision begins in the nomenclature and methodology, then continues through the workflow, Scenario 1, Scenario 2, and sensitivity-design subsections.

| Current paper element | Revision level | Required action |
|---|---:|---|
| Title | None expected | Retain unless the final results motivate a narrower title. |
| Highlights | Moderate | Replace the methodological highlight with the annual rolling-horizon contribution. Leave result-dependent and Scenario 3 highlights for the later results revision. |
| Abstract | Moderate | Rewrite the methods description around the year-long, state-linked rolling horizon and signed observed prices. Leave result values and Scenario 3 findings as placeholders for later. |
| Introduction and contributions | Minor | Preserve the motivation and background; update the contribution statement and any “day-ahead/single-day” wording. |
| Architecture and IT-workload background | Little or none | Retain, subject to terminology consistency. |
| Literature review | Little or none | Retain. An optional short gap statement on multi-day state continuity can be added only if supported by appropriate literature. |
| Nomenclature | Major | Add day, timestamp, horizon, state-boundary, workload-cohort, backlog, and piecewise-linear symbols; remove obsolete cyclic and extension-only definitions. |
| IT workload formulation | Major | Replace the current fixed 24-hour/extension formulation with timestamped cohorts, absolute deadlines, carried backlog, and core/look-ahead commitment. |
| IT power approximation | Major | Replace the SOS2 description with the implemented four-segment, non-uniform DLOG formulation and document its approximation error. |
| UPS formulation | Major | Replace two charge/discharge binaries and the daily cyclic equality with the implemented single-mode binary, boundary-state recursion, and inter-day state handoff. |
| Cooling and TES formulation | Major | Replace explicit Euler equations with the implemented implicit discretisation; remove TES daily cycling and add inter-day state handoff. |
| Workflow figure | Major | Redraw as a 365-day rolling process rather than one isolated 24-hour solve with a three-hour extension. |
| Electricity-price input description | Major | Remove the synthetic 24-hour table and document the actual 2025 signed price series, temporal expansion, and year-end look-ahead data. |
| Scenario 1 definition | Major | Describe the implemented annual baseline accurately and resolve the cold-aisle setpoint discrepancy before final wording. |
| Scenario 2 definition | Major | Recast as an annual rolling-horizon optimisation with committed daily decisions and continuous physical/workload state. |
| Scenario 3 | Deferred | Do not revise its detailed method or claims yet; preserve a clearly marked placeholder and ensure earlier notation does not preclude its later revision. |
| Sensitivity design | Major | Replace the old one-at-a-time daily design with the 12 representative weeks and the completed capacity/flexibility multiplier cases. |
| Results, discussion, conclusion | Deferred | Update only after the methodological revision and final result decisions. |

## 3. Modelling decisions to resolve before rewriting equations

These are not merely editorial issues. Each changes what can validly be claimed in the paper.

### 3.1 Baseline cold-aisle control

The paper currently defines the baseline as maintaining a fixed cold-aisle temperature of 22.5 °C. The annual baseline implementation instead allows cooling to optimise the cold-aisle temperature within the same temperature bounds used by the optimised case. A configuration value for a 22.5 °C baseline setpoint exists but is not enforced.

Before revising Scenario 1, choose one of the following:

1. enforce 22.5 °C in the baseline code and rerun the baseline; or
2. retain the implemented temperature flexibility and describe Scenario 1 as a storage-disabled, workload-at-arrival benchmark rather than a fixed-setpoint baseline.

The first option is closer to the current paper and gives a clearer contrast between conventional and flexible operation. The second avoids a rerun but changes the interpretation of savings because the baseline already contains cooling flexibility. This decision should be made before any final baseline or Scenario 2 text is drafted.

### 3.2 Thermal temperature limit

The paper states a cold-aisle range of 18–22.5 °C, while the active configuration uses 18–23 °C. Select and apply one limit consistently in the code, parameter table, equations, and narrative.

### 3.3 Cooling lower-bound constraint

The implementation imposes a lower bound on cooling equal to the contemporaneous IT electrical load. This constraint is not shown in the paper. Confirm its intended physical interpretation and units. If it is retained, add and justify it explicitly; if it is redundant or overly restrictive, remove it and revalidate before documenting the model.

### 3.4 Storage operating assumptions

The code uses one mode binary for each storage technology and does not impose minimum charge or discharge powers. The paper uses two binaries and minimum-power constraints. The central formulation should follow the active code unless the omitted restrictions represent genuine equipment requirements.

Also confirm that the intended central assumptions are:

- no daily terminal/cyclic energy equality;
- no explicit UPS or TES degradation/throughput cost; and
- no terminal-energy credit in the central case.

These assumptions materially affect inter-day arbitrage and should be stated rather than left implicit.

### 3.5 Price-series identity and provenance

The code reads the `IMRP_Amount` series, including negative values, and expands it to quarter-hour resolution. The paper currently calls the input a synthetic day-ahead price. Before revising the data subsection, confirm the precise market product, source, currency, time convention, preprocessing, and licence/citation. Do not describe the data as day-ahead unless that is verified.

### 3.6 End-of-year workload and energy treatment

The daily optimisation uses a three-hour look-ahead, including after 31 December, but annual accounting commits only intervals belonging to the target year. Define the intended terminal policy for:

- workload arriving near the end of 31 December whose deadline falls in the next year;
- the cost of processing that workload after midnight; and
- final UPS, TES, and temperature states.

The paper needs an explicit rule, even if the chosen rule is to allow unvalued terminal states and to report year-end outstanding workload separately.

### 3.7 Software and solver record

The paper states that SCIP and SOS2 are used. The annual implementation uses the HiGHS interface and a DLOG piecewise-linear representation. Capture the exact Python, Pyomo, HiGHS, operating-system, and processor details from the final computational environment before the paper is finalised.

## 4. Detailed section-by-section amendments

Line references below refer to the current `paper/main_new.tex` and may move during editing.

### 4.1 Title, highlights, keywords, and abstract

#### Title (`main_new.tex`, approximately line 58)

No methodological change is required. “Characterisation and Quantification” remains compatible with the annual rolling analysis.

#### Highlights (approximately lines 74–78)

Replace the present method-oriented highlight with one that identifies:

- a year-long rolling-horizon MILP;
- quarter-hour resolution;
- continuity of physical states and deadline-constrained workload between local days; and
- explicit treatment of signed electricity prices.

Do not update result magnitudes or the Scenario 3 highlight in this pass. Mark them for the later results revision.

#### Abstract (approximately lines 99–101)

Revise the method portion to state that the integrated IT, UPS, thermal, cooling, and TES model is solved sequentially across the full calendar year. The description should distinguish:

- the local-day commitment window;
- the three-hour advisory look-ahead;
- physical-state and workload-backlog handoff;
- operation across 23-, 24-, and 25-hour local days; and
- the use of signed observed prices.

Avoid describing the model as one independent daily optimisation. Remove or postpone any statement that price volatility was the dominant sensitivity unless that sensitivity is rerun under the new design. Retain placeholders for all numerical results and Scenario 3 findings.

#### Keywords

Consider adding “rolling-horizon optimisation” and “demand response” if space permits. This is optional and should not displace more central terms.

### 4.2 Introduction and contributions

#### General motivation (approximately lines 109–253)

Retain. The decarbonisation, data-centre growth, flexibility, and integrated-resource motivation remains valid and is not dependent on the revised computational procedure.

Perform only a terminology pass to remove any sentence that assumes:

- an isolated representative day;
- exclusively positive prices; or
- a day-ahead price product that has not been verified.

#### Contribution statement (approximately lines 254–258)

Revise “least-cost model considering day-ahead electricity prices” to identify the actual contribution: an annual, state-linked rolling-horizon formulation using signed prices.

The contribution list should explicitly include:

1. integration of deadline-constrained IT workload, UPS, cooling thermal mass, and TES;
2. preservation of storage, temperature, and unfinished-workload states between daily optimisations;
3. calendar-aware quarter-hour operation across daylight-saving transitions; and
4. a reduced but seasonally distributed sensitivity design using representative monthly weeks.

Do not include new numerical findings yet.

#### Paper structure paragraph

Update only if section headings are reorganised as proposed below. Keep the Scenario 3 description deliberately broad until that study is revised.

### 4.3 Architecture and background sections

#### Data-centre architecture (approximately line 260 onward)

Retain the physical architecture and explanatory figure. Check only that equipment labels match the revised equations: UPS behind-the-meter supply, chiller electric consumption, TES thermal charging/discharging, and constant facility overhead.

#### IT-workload background (approximately line 272 onward)

Retain. The explanation of interactive, batch, and flexible workload remains useful.

Where appropriate, standardise terminology so that a “cohort” means a quantity of flexible CPU-hours sharing an arrival timestamp and deadline. Avoid calling the tranche allocation fractions optimisation decisions if they are supplied as fixed inputs.

### 4.4 Literature review

The current review can remain substantially unchanged.

An optional closing sentence may state that the present work differs by retaining physical and workload states across a full-year sequence of local-day optimisations. Add this only after checking that the cited literature supports the claimed gap; otherwise present it narrowly as a feature of this work rather than a claim that no prior work does so.

No wholesale literature-review rewrite is warranted solely because the computational implementation changed.

### 4.5 Nomenclature

The nomenclature must be revised before the equations so that time and state indexing is unambiguous.

Add:

- local-day index \(d\);
- absolute quarter-hour timestamp index \(\tau\);
- daily core/commitment set \(\mathcal{C}_d\);
- daily look-ahead set \(\mathcal{L}_d\);
- full optimisation set \(\mathcal{H}_d=\mathcal{C}_d\cup\mathcal{L}_d\);
- control-interval and state-boundary indices, with \(N_d+1\) state boundaries for \(N_d\) intervals;
- workload-cohort index \(c\);
- cohort arrival \(a_c\), absolute deadline \(\delta_c\), initial CPU-hours \(W_c\), remaining workload, and processing rate;
- the fixed flexibility/deadline tranche shares;
- grid import and its upper bound;
- piecewise-linear breakpoints, weights, and segment notation if the DLOG construction is shown;
- initial and transferred UPS, TES, IT, rack, cold-aisle, and hot-aisle states; and
- sensitivity multipliers if they are used repeatedly in the design equations.

Remove or redefine:

- the fixed \(\mathcal{T}=\{1,\ldots,96\}\) assumption;
- the fixed \(\mathcal{T}_{ext}=\{1,\ldots,108\}\) definition;
- two-binary charge/discharge symbols if the one-binary implementation is retained;
- daily cyclic-state notation; and
- any notation treating the look-ahead IT load as an incremental quantity relative to a baseline.

Rename the price parameter according to the verified source rather than presuming that it is a day-ahead tariff.

### 4.6 Methodology overview

At the start of the methodology section (approximately line 298), add a compact overview of the hierarchy:

1. construct a continuous UTC quarter-hour timeline with Europe/London local-day labels;
2. create fixed deadline cohorts from the flexible workload;
3. solve one MILP for the current local day plus 12 look-ahead intervals;
4. commit only current-day controls and costs;
5. pass the terminal committed physical states and unfinished cohort workload to the next day; and
6. repeat through all 365 local dates.

This overview will prevent each component subsection from having to repeat the rolling-horizon logic.

### 4.7 IT workload formulation

The current formulation (approximately lines 444–488) requires a full replacement.

#### Fixed workload composition

Clarify that the interactive/inflexible and flexible shares, and the division of flexible demand among deadline tranches, are input data. They are not chosen by the optimiser.

For the sensitivity cases, define how the flexibility multiplier transfers CPU-hours between the inflexible and flexible pools while preserving the total workload in every original arrival interval. State that the flexible share is capped by the total available workload.

#### Cohort formulation

Define a cohort \(c\) by:

- an absolute arrival time;
- an absolute completion deadline;
- a fixed CPU-hour requirement; and
- an allowed processing window.

Introduce a processing-rate decision variable only for intervals between arrival and deadline. Require each cohort whose deadline lies within the current optimisation horizon to be completed. For cohorts extending beyond the horizon, carry the remaining CPU-hours to the next solve rather than forcing premature completion.

The aggregate IT-capacity constraint must include:

- inflexible workload processed on arrival;
- processing of newly arrived flexible cohorts; and
- processing of cohorts carried from preceding days.

Add an explicit workload conservation/backlog update. The prose should explain that a cohort’s absolute deadline is preserved when it crosses midnight.

#### Core and look-ahead treatment

Delete the current explanation that the three-hour extension isolates only workload shifted from the previous day. The implementation models the complete facility and complete workload over the look-ahead.

Delete the piecewise definition that subtracts baseline IT power during the extension. IT power is the full IT demand in every optimised interval. The look-ahead is advisory: only controls and processing in the core local day are committed, and future intervals are reoptimised on the next day.

### 4.8 IT electrical-power model

Retain the underlying nonlinear relation between utilisation and IT power, including the exponent, idle power, and maximum power, if these parameter values remain unchanged.

Replace the SOS2 implementation description with:

- four piecewise-linear segments;
- five non-uniform breakpoints generated by the implemented breakpoint rule;
- a DLOG representation in Pyomo; and
- a statement that the resulting interpolation overestimates the convex analytic curve.

Report the approximation verification in either this subsection or an appendix: maximum and mean absolute error over the utilisation range, together with the reason for preferring the compact representation. If the eight- or sixteen-segment sensitivity is later adopted for the central run, update these details accordingly.

Explain that the baseline has fixed CPU processing and therefore evaluates the same piecewise curve directly without creating the IT-load binary decisions used in the optimised formulation.

### 4.9 UPS formulation

Replace the UPS equations (approximately lines 489–521) with the implemented boundary-state formulation.

The revised equations should show:

- one binary mode variable per interval;
- charging bounded by the mode binary;
- discharging bounded by its complement;
- charge and discharge efficiencies;
- interval-duration scaling;
- energy and state-of-charge limits;
- the effective discharge limit imposed by the IT load as well as the equipment rating; and
- grid-supplied IT power as total IT demand minus UPS discharge plus UPS charge, with no electricity export.

Remove minimum charge/discharge constraints unless they are reinstated in code.

Remove the daily cyclic equality. State that the first-day UPS state is fixed and every subsequent day begins from the preceding day’s committed terminal state.

If throughput costs and terminal values remain zero in the central case, say so briefly in the assumptions or parameter table rather than implying that storage degradation has been represented.

### 4.10 Cooling, thermal states, and TES

The current thermal equations (approximately lines 522–564) require a full discretisation correction.

#### Thermal-state equations

Replace “forward/explicit Euler” with “backward/implicit Euler.” Define temperatures at interval boundaries and heat/electrical flows over intervals. Write the recursions using the next-boundary temperatures on the right-hand side, matching the implemented model.

Explain that the implicit scheme preserves linearity while avoiding the numerical instability observed with the explicit quarter-hour discretisation. Keep this explanation short in the paper; detailed validation can be placed in supplementary material.

Add the cooling lower-bound constraint only if it is confirmed under Section 3.3 above.

Show that the initial IT, rack, cold-aisle, and hot-aisle temperatures are fixed on 1 January and then handed from one committed day to the next. Remove any implication that thermal states are independently reset each day.

#### TES equations

Use one charge-mode binary, with mutually exclusive thermal charge and discharge, energy limits, efficiencies, and standing loss as implemented.

Remove the daily cyclic TES equality. Add the same inter-day state-handoff rule used for the UPS.

Use the confirmed cold-aisle bounds consistently throughout this subsection and the parameter tables.

### 4.11 Whole-facility balance and objective

Create a short subsection that collects equations currently dispersed across the component descriptions.

Define total grid import as the sum of:

- grid-supplied IT power after UPS action;
- chiller electricity serving contemporaneous cooling;
- chiller electricity charging TES;
- constant auxiliary/facility demand; and
- any other term actually active in the implementation.

Add the implemented structural grid-import upper bound and explain how it is derived from equipment limits. If lower scenario-specific import caps can be configured but are not used centrally, mention them only as optional constraints.

State that electricity prices may be negative and that export is not permitted. The central objective should be the signed-price cost of the full optimisation horizon. Explain separately that annual reported settlement cost is accumulated only over committed core intervals, so look-ahead intervals are never counted twice.

Optional storage throughput costs and terminal credits need not clutter the central objective if they are zero. They can be listed as available extensions in supplementary material.

### 4.12 Annual rolling-horizon procedure

Add a dedicated subsection and preferably a short algorithm box or pseudocode.

For every local date \(d\):

1. obtain that date’s 92, 96, or 100 quarter-hour core intervals;
2. append 12 quarter-hour look-ahead intervals;
3. load the previous committed physical states and unfinished cohorts;
4. add new workload cohorts arriving in the horizon;
5. solve the MILP subject to the stated time limit and optimality-gap policy;
6. validate termination, feasibility, exclusivity, and workload conservation;
7. commit only core decisions;
8. calculate core settlement cost using the original signed price; and
9. pass the committed boundary state and remaining cohorts onward.

Explain that a continuous UTC backbone prevents duplicated or missing physical intervals, while Europe/London labels determine the local-day commitment windows and correctly retain daylight-saving days.

Document the year-end look-ahead source and the terminal policy chosen under Section 3.6.

### 4.13 Computational implementation

Replace the present SCIP/SOS2 and “all solves optimal in 5–10 seconds” text.

Record:

- exact software and solver versions;
- the DLOG representation and central segment count;
- the time limit, requested MIP gap, and maximum accepted gap;
- the rule for accepting or rejecting non-optimal incumbent solutions;
- the standard-horizon binary count;
- checkpoint/resume behaviour; and
- automated checks for objective reconciliation, state continuity, workload conservation, and simultaneous storage operation.

Runtime statistics and counts of non-optimal horizons belong in the results or computational-performance subsection, not in the method plan.

### 4.14 Case-study introduction and workflow figure

The case-study introduction (approximately line 565) should distinguish:

- the annual Scenario 1 benchmark;
- the annual Scenario 2 integrated optimisation; and
- the later, separate Scenario 3 flexibility-duration experiment.

Redraw the workflow figure (approximately line 580) as a loop across local dates. It should show data preparation, cohort creation, daily core plus look-ahead solve, core commitment, state/backlog transfer, and annual aggregation.

Do not depict the look-ahead as an independent extension used only for prior-day shifted work.

### 4.15 Electricity-price data

Remove the synthetic 24-hour price table (approximately line 600). Replace it with a concise data subsection covering:

- source and market product;
- calendar coverage;
- currency and units;
- original temporal resolution;
- conversion/expansion to quarter-hour intervals;
- treatment of negative prices;
- timezone and daylight-saving alignment; and
- the additional data used for the final look-ahead.

Any descriptive annual price statistics or plots should be added later with the results, not embedded in the core method.

The signed series must remain the basis for reported settlement costs. Any price-floor or constant-shift formulation should be described only as a solver diagnostic or sensitivity, not as the central economic objective.

### 4.16 Scenario 1

Rewrite the Scenario 1 subsection (approximately line 615) as an annual benchmark using the same annual timeline and exogenous demand inputs as Scenario 2.

The definitive description should state that:

- workload is executed at its arrival time;
- UPS charging and discharging are disabled;
- TES charging and discharging are disabled;
- auxiliary demand is included identically to Scenario 2;
- physical temperature states still evolve continuously between days; and
- the cooling-control policy is whatever is selected under Section 3.1.

Avoid calling all flexible workload “inflexible” if the implementation simply removes its temporal rescheduling while retaining the original accounting labels. “Workload-at-arrival” is clearer.

### 4.17 Scenario 2

Rewrite the Scenario 2 subsection (approximately line 630) around the annual rolling process, not a single optimised day.

It should identify the controllable decisions:

- flexible-cohort execution times within deadlines;
- UPS charge/discharge;
- cooling/temperature trajectory;
- TES charge/discharge; and
- associated grid import.

It should also state what is fixed:

- workload arrivals and total CPU-hours;
- cohort deadline shares;
- equipment capacities and efficiencies;
- auxiliary load;
- the signed price series; and
- the initial state on 1 January.

The subsection should refer to the common equations rather than restating them. Its main purpose is to define which controls are enabled relative to Scenario 1.

### 4.18 Scenario 3

Leave the detailed Scenario 3 methodology unchanged for now, except where symbol changes elsewhere would make it uncompilable or internally inconsistent.

Insert an editorial marker that the flexibility-duration experiment, its assumptions, and its result claims require a separate revision after the relevant runs are complete. Do not attempt to infer new Scenario 3 behaviour from the annual Scenario 2 model.

### 4.19 Sensitivity-analysis design

Replace the existing sensitivity-design subsection (approximately line 693) with the implemented representative-week design.

#### Sampling frame

State that one complete Monday-to-Sunday week is selected for each calendar month, yielding 12 weeks and 84 scored local days per case. No warm-up days are used. Each monthly week is solved as an independent run initialised from the common default physical state.

Explain the representative-week selection method:

- candidate weeks are described by monthly price characteristics including mean, standard deviation, minimum, maximum, and negative-price share;
- characteristics are standardised before computing distance;
- the joint set of 12 weeks is constrained to reproduce the annual mean price and negative-price share within the specified tolerances; and
- among feasible combinations, the selection minimises aggregate monthly characteristic distance.

This makes clear that the weeks are selected systematically rather than manually.

#### Parameter cases

Define the completed cases:

- central setting;
- workload-flexibility multipliers of 0.50, 0.75, 1.25, and 1.50;
- UPS energy-capacity multipliers of 0.50, 0.75, 1.25, and 1.50; and
- TES energy-capacity multipliers of 0.50, 0.75, 1.25, and 1.50.

Together with the common baseline, this gives 13 optimised parameter settings plus the baseline comparator.

State that UPS and TES **energy capacities** are scaled while their charge/discharge power ratings remain fixed. Initial stored energy should be scaled to preserve the initial state-of-charge fraction. Define the workload-flexibility multiplier as a reallocation between flexible and inflexible workload that preserves total CPU-hours.

Remove the present price-volatility sensitivity from the claimed completed design unless it is rerun and integrated consistently.

#### Aggregation

Define monthly or annual aggregation by interval-count weights, so 23- and 25-hour daylight-saving days are represented correctly. Calculate percentage savings as a ratio of summed costs rather than as an average of weekly percentages.

State the limitation introduced by omitting warm-up days: each sampled week begins from a common default physical state rather than the state generated by the preceding calendar days. Consequently, the sampled analysis is suitable for comparative response curves across parameter settings, but it is not automatically an unbiased substitute for a full-year absolute saving. Any full-year validation requirement should be decided before the sensitivity results are presented.

## 5. Proposed revised method structure

A clearer structure would be:

1. **Methodology**
   1. System boundary and temporal indexing
   2. IT workload inputs and cohort formulation
   3. IT electrical-power approximation
   4. UPS model
   5. Cooling thermal model and TES
   6. Whole-facility power balance and objective
   7. Annual rolling-horizon algorithm
   8. Computational implementation and validation
2. **Case-study design**
   1. Data, calendar, prices, and initial conditions
   2. Scenario 1: workload-at-arrival benchmark
   3. Scenario 2: integrated annual optimisation
   4. Scenario 3: flexibility duration — deferred revision
   5. Representative-week sensitivity design

This separates the reusable mathematical model from the scenario-specific controls and avoids embedding the rolling algorithm piecemeal in component equations.

## 6. Equation replacement checklist

Before treating the method as complete, verify that the revised paper includes and matches the code for:

- variable-length daily core and fixed 12-interval look-ahead sets;
- cohort arrival, absolute deadline, completion, capacity, and backlog equations;
- full IT power in both core and look-ahead intervals;
- DLOG piecewise-linear IT power;
- one-binary UPS exclusivity and boundary-state recursion;
- one-binary TES exclusivity and boundary-state recursion;
- implicit thermal-state recursions;
- facility grid-import balance and cap;
- signed-price objective;
- inter-day state-handoff equations;
- core-only commitment and cost accounting; and
- sensitivity scaling and weighted aggregation.

Explicitly delete or supersede:

- fixed 96-interval daily indexing;
- the extension-only incremental IT-power equation;
- SOS2 implementation claims;
- UPS and TES daily cyclic equalities;
- forward-Euler thermal equations;
- two-binary storage formulations if not reinstated;
- synthetic day-ahead price assumptions; and
- independent-day language.

## 7. Figures and tables affected before the results revision

### Retain with minor checks

- data-centre architecture figure;
- workload classification material;
- physical equipment-parameter tables, once bounds and ratings are reconciled.

### Replace or substantially revise

- nomenclature table;
- optimisation workflow figure;
- synthetic hourly price table;
- parameter table entries for temperature bounds and initial states;
- solver/formulation table or prose; and
- sensitivity-design table.

### Add if space permits

- a compact rolling-horizon timeline showing core, look-ahead, commitment, and handoff;
- pseudocode for the daily loop; and
- a small table distinguishing Scenario 1 from Scenario 2 by enabled control.

The rolling timeline and scenario-control table would likely communicate more efficiently than adding further prose.

## 8. Recommended writing order

1. Resolve the seven modelling decisions in Section 3.
2. Freeze a central configuration and record exact solver/software versions.
3. Rewrite nomenclature and temporal indexing.
4. Rewrite workload, IT power, UPS, thermal/TES, objective, and rolling-horizon equations.
5. Rewrite the data, Scenario 1, Scenario 2, and sensitivity-design subsections.
6. Redraw the workflow and rolling-timeline figures.
7. Update the abstract, highlights, and contribution paragraph.
8. Perform a consistency audit against the frozen configuration and code.
9. Only then replace results, discussion, Scenario 3, and conclusion material.

## 9. Definition of done for this revision stage

The pre-results revision is complete when:

- every central equation maps to an active implementation constraint;
- all active central constraints with material physical or economic consequences are disclosed;
- the paper no longer describes isolated daily solves or daily state resets;
- workload carried across midnight is defined mathematically;
- DST and year-end policies are explicit;
- baseline controls are accurately distinguished from optimised controls;
- price provenance and signed-price treatment are correct;
- solver and piecewise-linear claims match the final environment;
- the representative-week sensitivity design is reproducible from the text; and
- all result-dependent sentences and Scenario 3 details are visibly reserved for the later revision rather than silently left as current claims.

