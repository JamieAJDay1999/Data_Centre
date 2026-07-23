# Annual Optimisation and Flexibility Analysis

Actions arising from Jamie–Meysam catch-up, 17 July 2026

**Purpose:** Prepare a robust 365-day rolling-horizon model and a focused parameter-sensitivity analysis.

**Working lead:** Jamie Day

**Review support:** Meysam Qadrdan; Zahra for rolling-horizon implementation

**Status:** Proposed plan; target dates are suggested and can be reset at the next supervision meeting.

> **Immediate focus**  
> Make the 365-day optimisation physically consistent before expanding the analysis: resolve negative-price behaviour, link daily state variables, and close any end-of-horizon workload loophole.

## 1. Desired outcome

A reproducible annual simulation that behaves physically under negative electricity prices, passes state consistently between daily horizons, supports controlled parameter sweeps, and feeds a representative-day flexibility envelope without changing the existing Stage 3 visualisation unnecessarily.

- The model can run all 365 days without discontinuities at day boundaries.

- Negative-price cases cannot create value through non-physical energy consumption or dissipation.

- Annual cost is reported using the original price series, regardless of any transformed series used inside the optimisation.

- Sensitivity results are compared with one documented central scenario using a small, consistent KPI set.

## 2. Priority workstreams

### A. Negative-price treatment and physical behaviour

Treat the proposed upward price shift as a hypothesis to test, not an assumed equivalence. Adding a constant to every price preserves the schedule only if total grid energy is fixed across all feasible schedules. Battery losses, cooling choices, storage losses and workload recovery may make total energy schedule-dependent.

- Audit the battery formulation: confirm the binary charge/discharge constraint is active in every run and prevents simultaneous charging and discharging.

- Audit cooling and thermal storage for any route that can increase consumption or dissipate energy solely to exploit a negative price.

- Run the same test days under three price treatments: original prices, prices capped at zero, and prices shifted upward so the minimum is zero.

- Compare schedules, grid energy, equipment states and objective values. Record the conditions under which the transformed-price schedule is or is not invariant.

- For every reported result, calculate true operating cost outside the optimiser as the time-step sum of original price multiplied by site electricity consumption.

### B. Rolling-horizon state continuity

Run each daily problem over approximately 24 hours plus the existing 3-hour recovery window, retain the first 24 hours as the reported day, and initialise the next day from the relevant terminal states.

- Carry forward battery state of charge, thermal-storage energy and IT-room temperature.

- Carry any deferred workload that remains due within the next day's first three hours; represent it consistently so it cannot disappear at midnight.

- Confirm that workload delay remains bounded to three hours and that the terminal window does not encourage systematic shifting beyond the reported day.

- Create boundary checks showing that each next-day initial state equals the previous day's retained terminal state to numerical tolerance.

- Discuss implementation details with Zahra, who is using a similar rolling-horizon approach.

### C. Central scenario and parameter sweep

Define one central set of assumptions, then vary the flexible/inflexible workload split, battery capacity and thermal-storage capacity over agreed ranges. Use the central scenario for the main narrative and report other cases as deviations from it.

- Document the central values and the reason each is representative.

- Use a staged sweep first (one parameter at a time) to expose trends and coding issues; add a full factorial sweep only if interactions are important and the output remains interpretable.

- Aggregate annual operating cost directly across all time steps.

- Aggregate flexibility using a small set of magnitude and duration metrics rather than producing figures for every time step.

### D. Representative day and Stage 3 outputs

Select a representative day from the annual central-scenario results using a documented method. Rerun the existing Stage 3 flexibility-envelope/heat-map analysis for that day and retain the current chart family unless the revised model requires a change.

- Define the representative-day selection rule using normalised distance to annual central values (for example demand, price, temperature and workload), with extreme days retained separately if needed.

- Check that the selected day is not distorted by a day-boundary state or deferred-workload carryover.

- Use sensitivity tables or compact summary plots for parameter comparisons; avoid duplicating the full Stage 3 chart set for every scenario.

## 3. Action register

| **ID** | **Action and completion test**                                                                                                                                       | **Owner**      | **Proposed target** |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|---------------------|
| **A1** | Trace every negative-price-sensitive term and constraint in the objective and energy balances. Completion: short audit note plus confirmed battery binary behaviour. | Jamie          | 31 Jul 2026         |
| **A2** | Implement and compare original, capped and shifted price cases on selected negative-price days. Completion: schedule/KPI comparison and conclusion on invariance.    | Jamie          | 31 Jul 2026         |
| **A3** | Add true-cost post-processing using original prices at every time step. Completion: reconciled annual and daily cost outputs.                                        | Jamie          | 31 Jul 2026         |
| **B1** | Implement daily state hand-off for battery SoC, TES energy and IT-room temperature. Completion: automated boundary residual checks pass.                             | Jamie          | 7 Aug 2026          |
| **B2** | Verify deferred-workload carryover and three-hour service limit across midnight. Completion: workload energy balance closes on boundary test cases.                  | Jamie          | 7 Aug 2026          |
| **B3** | Compare rolling-horizon implementation with Zahra's approach and document any changes.                                                                               | Jamie / Zahra  | Early Aug           |
| **C1** | Define central scenario and parameter ranges for workload flexibility, battery capacity and TES capacity.                                                            | Jamie + Meysam | Next review         |
| **C2** | Run staged annual sensitivity analysis and calculate agreed KPIs. Completion: tidy scenario-results dataset.                                                         | Jamie          | 14 Aug 2026         |
| **D1** | Select and justify the representative day; rerun the existing Stage 3 envelope and heat-map outputs.                                                                 | Jamie          | 21 Aug 2026         |
| **E1** | Prepare code for joint review: comments, configuration block, run instructions, state definitions and output dictionary.                                             | Jamie          | Before code review  |
| **E2** | Review revised code and identify worthwhile additional features.                                                                                                     | Meysam + Jamie | After conference    |
| **E3** | Join the gas/electricity network visualisation meeting and exchange ideas with the software developer.                                                               | Meysam / Jamie | When scheduled      |

## 4. KPI set for annual comparisons

| **KPI**                   | **Use**                                                               | **Minimum reporting form**                            |
|---------------------------|-----------------------------------------------------------------------|-------------------------------------------------------|
| **Annual operating cost** | Primary economic outcome; recalculate with original time-step prices. | £/year and % change from central case                 |
| **Flexibility magnitude** | How much import can increase or decrease.                             | MW or kW; % change from central case                  |
| **Flexibility duration**  | How long the response can be sustained.                               | Minutes/hours at defined power levels                 |
| **State continuity**      | Confirms rolling-horizon consistency.                                 | Daily boundary residuals for SoC, TES and temperature |
| **Workload service**      | Checks that deferral is completed within the 3-hour limit.            | Deferred/processed energy balance and maximum delay   |

## 5. Validation and acceptance checks

- Negative-price sanity: no simultaneous battery charging/discharging; no cooling or storage loop consumes energy without a physical service.

- Price-treatment comparison: differences among original, capped and shifted cases are quantified and explained.

- Cost reconciliation: post-processed cost equals the time-step calculation using the original price series.

- State hand-off: SoC, TES energy and IT-room temperature match across each retained day boundary within tolerance.

- Workload conservation: all accepted workload is processed; deferred energy neither disappears nor is duplicated at midnight.

- Terminal behaviour: no systematic rise in deferral near the end of the retained 24-hour horizon.

- Annual reproducibility: the full run completes from one configuration with logged assumptions, seed/version information where relevant, and tidy outputs.

## 6. Decisions required at the next review

- Negative prices: retain original prices with strengthened physical constraints, or use a transformed series only if schedule equivalence is demonstrated.

- Central scenario: confirm the baseline workload-flexibility share and battery/TES capacities.

- Sensitivity design: confirm one-at-a-time sweeps versus a limited factorial design.

- Representative day: agree the variables and weighting used in the selection metric, and whether separate extreme-price days should also be shown.

- Flexibility aggregation: agree the exact magnitude/duration KPIs and time windows used for annual comparison.

## 7. Recommended execution order

| **Step** | **Focus**              | **Exit condition**                                             |
|----------|------------------------|----------------------------------------------------------------|
| **1**    | **Model integrity**    | Resolve negative-price behaviour and true-cost calculation.    |
| **2**    | **Temporal integrity** | Link daily states and workload carryover; pass boundary tests. |
| **3**    | **Baseline lock**      | Agree the central scenario and KPI definitions.                |
| **4**    | **Annual analysis**    | Run parameter sweeps and aggregate outcomes.                   |
| **5**    | **Communication**      | Choose the representative day and refresh Stage 3 outputs.     |

*Source: meeting transcript, “Meysam, Jamie Catch up”, 17 July 2026.*
