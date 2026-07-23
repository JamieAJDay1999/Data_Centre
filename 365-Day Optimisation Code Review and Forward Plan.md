# 365-Day Optimisation: Code Review and Forward Plan

**Prepared:** 17 July 2026  
**Scope:** The current code and stored outputs required to run the 2025 annual data-centre nominal and optimised cases.  
**Primary recommendation:** Retain the fast 24-hour plus look-ahead optimisation structure, but rebuild it as a genuinely sequential rolling-horizon controller with explicit physical state, workload backlog, timestamp, price, and checkpoint semantics.

## 1. Executive conclusion

The repository can currently execute and aggregate a large set of daily optimisation cases, and the stored 2025 run contains results for 363 standard days. However, this is not yet a physically continuous 365-day optimisation.

The current annual command runs each date as an independent 27-hour problem. Every date receives a fresh parameter object; physical states are reset or allowed to be selected by the optimiser; flexible workload scheduled after midnight is not carried into the next daily solve; and the 23-hour and 25-hour daylight-saving days are excluded. The annual result is therefore best described as an **independent-day price sensitivity analysis using observed 2025 prices**, rather than a state-linked annual simulation.

This distinction matters. Battery energy, thermal storage energy, room temperatures, and deferred IT work are all intertemporal quantities. If they are reset at midnight, the annual schedule and cost no longer represent one feasible operating trajectory.

The best route forward is an incremental rolling-horizon refactor rather than a single monolithic annual MILP:

1. define unambiguous state and action indices;
2. pass physical states and outstanding workload between days;
3. use every source timestamp exactly once, including daylight-saving transitions;
4. make the original signed price series the primary settlement series and remove negative-price degeneracies from the physical model;
5. count only committed intervals in annual cost and energy totals;
6. add invariant tests, solver-quality checks, and fingerprinted checkpoints; and
7. run annual sensitivity cases as separate sequential chains that can be parallelised across scenarios.

The existing independent-day runner should be retained, renamed, and used for price-day screening and representative-day analysis. It should not be discarded, because it is already useful and its stored outputs provide a valuable regression baseline.

## 2. What currently runs

### 2.1 Current entry point

The current full available-day command is:

```powershell
python run_imrp_year_sample.py --sample-days all --floor-negative-prices
```

The main orchestration path is:

1. `run_imrp_year_sample.py:68-142` loads observed IMRP prices and constructs a 27-hour price vector for each eligible date.
2. `run_imrp_year_sample.py:145-175` turns dates into independent sensitivity-style cases.
3. `run_imrp_year_sample.py:216-250` monkey-patches the existing nominal, optimisation, and flexibility modules with the observed tariff and case-specific paths.
4. `run_imrp_year_sample.py:259-283` loops through dates serially and resumes a case whenever `tier1_result.json` already exists.
5. `sensitivity_sweep.py:296-341` runs Scenario 1 nominal operation and Scenario 2 optimisation for each date.
6. `run_imrp_year_sample.py:291-350` merges daily JSON results and calculates distribution statistics.
7. `run_imrp_year_sample.py:438-580` writes CSV, JSON, and HTML reports.

Each model uses:

- 96 retained quarter-hour slots representing 24 hours;
- 12 additional quarter-hour slots representing a three-hour look-ahead;
- a generic 108-row load profile for the day and extension;
- a 96-row shiftability profile;
- observed hourly prices repeated across four quarter-hour model slots; and
- SCIP through Pyomo.

### 2.2 Current stored annual result

The existing artifacts show:

- 363 completed daily cases;
- 363 unique dates;
- 26 cases where negative prices were clipped to GBP 0/MWh;
- no missing daily saving values;
- approximately 1,075 seconds of summed recorded case runtime, or about 17.9 minutes;
- total reported nominal cost of GBP 546,292.86;
- total reported optimised cost of GBP 519,482.12; and
- a ratio-of-sums saving of approximately 4.908%.

These totals are useful diagnostics, but they inherit the independent-day, boundary-state, daylight-saving, mixed-price-treatment, and extension-accounting limitations described below. They should not yet be used as the final annual result.

### 2.3 What the current workflow already does well

- It validates required price columns, duplicate date-period keys, nulls, and daily settlement-period sequences.
- It uses actual next-day prices for the three-hour extension.
- It isolates each case's inputs and outputs in a separate directory.
- It supports resume, report-only, dry-run, force, and fail-fast modes.
- It records the dates and counts of negative-price adjustments.
- It has successfully exercised the model across nearly a full year of different price profiles.
- Daily cases are fast enough that model correctness, not raw runtime, is currently the main bottleneck.

These are strong foundations for a proper rolling runner.

## 3. Critical findings from the code audit

## 3.1 The current annual run is 363 independent days

`run_imrp_year_sample.py:265-277` invokes each date as a separate case. `sensitivity_sweep.py:236-255` constructs a fresh `ModelParameters` object for every case. No previous-day result or state is accepted by the next model.

Consequences:

- battery energy is not continuous across midnight;
- thermal-storage energy is not continuous across midnight;
- IT, rack, cold-aisle, and hot-aisle temperatures are not continuous;
- workload deferred after midnight is not represented in the next optimisation; and
- every date can use a different, mutually incompatible physical trajectory.

This is the highest-priority issue because all subsequent annual KPIs depend on a feasible stitched operating trajectory.

## 3.2 Initial physical states are reset or freely chosen

The default battery starts at 600 kWh in `inputs/parameters_optimisation.py:109-121`. The battery balance uses that fixed start value in `constraints.py:52-59`, and `constraints.py:65` forces the battery back to the same value at the end of the full 27-hour horizon.

The thermal boundary is more serious. `constraints.py:76-80` applies lower bounds rather than equalities to the first-slot temperatures and TES energy. The optimiser can therefore choose a favourable initial state for free, subject only to the variable bounds.

The stored 363-day outputs confirm that this happens:

- first-slot IT temperature ranges from 30.4797 to 49.4015 degC, despite a nominal initial value of 28.5 degC;
- first-slot rack temperature ranges from 26.9227 to 39.7369 degC, despite a nominal initial value of 26 degC;
- first-slot hot-aisle temperature reaches 40 degC; and
- first-slot TES energy ranges from 500 to 932.5 kWh, despite a nominal initial value of 500 kWh.

The annual model must instead receive one explicit boundary state and fix every initial state by equality.

## 3.3 State and action indexing is inconsistent

Battery energy at slot `s` includes the charge and discharge action at slot `s` (`constraints.py:56-59`). By contrast, TES and thermal state at slot `t` are updated using actions and conditions from slot `t-1` (`constraints.py:95-108`).

This creates two problems:

1. the correct state to carry at the end of the first 96 slots differs between assets; and
2. final-slot TES charge/discharge actions do not enter a subsequent TES state equation, even though they are costed.

The safe fix is to use boundary states indexed `0..N` and actions indexed `0..N-1`:

```text
state[t + 1] = transition(state[t], action[t], disturbance[t])
```

The initial state is then `state[0]`, the state after the retained day is `state[96]`, and the state after a 108-slot horizon is `state[108]`. This removes the current off-by-one ambiguity.

## 3.4 Daily cyclic constraints conflict with rolling operation

`constraints.py:65` forces UPS energy at the end of the 27-hour horizon back to its starting value. `constraints.py:115-116` does the equivalent for TES energy.

These constraints prevent genuine cross-day storage behaviour and make core-day decisions depend on an artificial requirement to recover the state during the three-hour extension. Temperatures have no equivalent terminal value, so the model has inconsistent horizon-end incentives across assets.

For rolling operation:

- remove daily hard cyclic equalities;
- pass the retained boundary state into the next solve;
- apply a terminal value, soft target, or reserve constraint only at the end of the look-ahead; and
- apply a clearly documented year-end treatment to the last horizon.

## 3.5 Flexible workload is not carried across midnight

The optimised model creates job arrivals only for the first 96 slots, but allows them to execute as late as slot 108 (`optimisation.py:70-73`, `constraints.py:4-10`). The extension includes generic next-day inflexible load but not next-day flexible arrivals. The next daily case then starts again with a fresh workload profile.

This means a day-D job scheduled after midnight can overlap next-day work without both being subject to the same CPU-capacity constraint. The annual runner does not read `load_profiles_opt.csv` or `shiftability_profile_opt.csv` back as a next-day state; those files are produced for later flexibility analysis, not for annual continuity.

The correct rolling state must therefore include outstanding workload cohorts. Each cohort should record at least:

- cohort or arrival identifier;
- remaining CPU-hours;
- absolute deadline timestamp;
- shiftability tranche or service class; and
- any processing already committed in an overlap period.

The next solve must include both newly arriving work and carried outstanding work.

## 3.6 The two daylight-saving days are omitted

`run_imrp_year_sample.py:105-124` accepts only dates with periods exactly `1..24`. The 23-hour spring transition and 25-hour autumn transition are excluded, leaving 363 dates.

A true annual chain cannot skip those intervals because the state before the gap would be passed directly to a non-adjacent physical time.

Recommended treatment:

- build a continuous, timezone-aware source timeline;
- preferably optimise on a monotonic UTC interval index;
- retain local date and settlement-period labels for reporting; and
- segment the continuous series into rolling solve windows rather than assuming every local date contains 96 quarter-hours.

The acceptance condition is not merely 365 labels. Every source interval must be used exactly once in the committed annual series, without duplicates or gaps.

## 3.7 Negative prices are currently handled by a mixed recovery rule

`run_imrp_year_sample.py:130-138` can clip negative prices to zero. The stored annual artifacts contain 337 cases solved with original non-negative horizons and 26 recovered cases solved after clipping negative values. This produces a mixed annual price policy.

The original-price failures exposed physical and modelling weaknesses rather than proving that signed prices are invalid. In particular:

- UPS charging and discharging are mutually exclusive (`constraints.py:60-64`), which is good;
- TES charging and discharging are not mutually exclusive;
- cooling and storage losses can create controllable consumption; and
- free or weak terminal states can reward horizon-end energy use.

The stored outputs contain 15 intervals across 4-5 October with simultaneous TES charging and discharging, with overlap as high as 300 kW. These cases occurred at zero prices after clipping, so the degeneracy is not limited to strictly negative prices.

Adding a constant to all prices is not generally schedule-preserving. If energy use is decision-dependent, the transformed objective is:

```text
sum(price[t] * energy[t]) + constant * sum(energy[t])
```

The additional term changes the optimum unless total energy is invariant across every feasible schedule. In this model, total energy varies with IT power, cooling, storage losses, and terminal states.

Recommended policy:

- retain the original signed price as the settlement price;
- make the physical model bounded and non-degenerate under signed and zero prices;
- add TES charge/discharge exclusivity;
- consider battery/TES throughput or degradation cost if deliberate cycling is outside the intended operating policy;
- add explicit import-capacity limits if they exist physically;
- calculate realised cost on the original settlement price for every committed interval; and
- treat zero-floor and constant-shift cases as transparent sensitivities, not as silently mixed recovery rules.

If an alternative optimisation price is used, store both `optimisation_price` and `settlement_price` and report the counterfactual realised cost separately.

## 3.8 First-slot cooling uses a noncausal averaging constraint

`constraints.py:83-88` forces first-slot HVAC and TES chiller power to equal their average power over all later slots. This is a horizon-wide boundary workaround, not a physical initial condition. It makes the first action depend on the length and content of the future look-ahead.

The state-index refactor should remove this relation. Interval-zero controls should use the same physical equations as every other interval, starting from a supplied state at boundary zero.

## 3.9 Annual cost accounting is not a stitched time-series total

`sensitivity_sweep.py:259-276` reports each optimised daily cost as:

```text
optimised core cost + (optimised extension cost - nominal extension cost)
```

The next day is then solved independently over the overlapping hours. Component costs in `sensitivity_sweep.py:279-293` cover only the first 96 slots, so they do not reconcile with the headline cost that includes an extension adjustment.

In a corrected rolling run:

- look-ahead intervals inform the decision but are not counted when first forecast;
- only committed intervals are appended to the annual result;
- an interval is counted when it becomes part of the committed core;
- the final tail is handled once under an explicit year-end rule; and
- annual total cost equals the direct sum of committed grid import multiplied by original price and duration.

## 3.10 Checkpoints do not prove compatibility

The current resume check tests only whether `tier1_result.json` exists (`run_imrp_year_sample.py:273-277`). It does not verify:

- input-data hash;
- price treatment;
- resolved parameter values;
- code version;
- solver version and options;
- checkpoint schema; or
- previous-day state.

For independent cases this can silently reuse stale results. For a rolling chain it is more dangerous because a changed or corrupt day invalidates every downstream state.

The rolling runner should use atomic checkpoint writes and include a deterministic fingerprint. Each daily checkpoint should include the hash of its predecessor so that invalidation propagates safely.

## 3.11 Solver quality and reproducibility are under-specified

`optimisation.py:525-547` hardcodes SCIP and accepts time-limit or `other` outcomes without recording a reliable incumbent count, optimality gap, or complete status. The nominal path accepts only an optimal termination. No time limit, gap, thread count, random seed, or solver log policy is exposed in the annual CLI.

The current default shell cannot find SCIP, although the stored results were produced in another environment. `requirements.txt` uses lower-bound package requirements and the metadata does not record the Python environment, solver version, git revision, input hash, or resolved parameter set.

The canonical IMRP input currently has SHA256:

```text
3BAC942112CF76638F696387E8EDAF57D376BF5C1F60648C46AB2A9ADB39904C
```

Future runs should record this automatically rather than relying on a value written into a plan.

## 3.12 The annual load assumption is static

The current annual wrapper changes the observed tariff by date but copies the same load and shiftability profiles into every case (`sensitivity_sweep.py:194-233`). This can be valid for a controlled price-year experiment, but it is not an annual demand trace.

The study must choose and state one of two interpretations:

1. **Controlled tariff experiment:** repeat a representative daily data-centre workload across the year to isolate the effect of observed price variation.
2. **Annual operations experiment:** supply date-specific workload, ambient conditions, and any other seasonal disturbances.

The first is a reasonable primary paper scenario if it is labelled accurately. The second is more realistic but requires additional defensible data.

## 4. Recommended target architecture

## 4.1 Core data structures

Introduce typed, serialisable structures rather than passing module globals and loosely structured dictionaries.

### `ModelConfig`

Contains all resolved technical parameters:

- timestep and look-ahead;
- UPS and TES capacities, efficiencies, and power limits;
- temperature limits and thermal parameters;
- flexible-load tranches and deadlines;
- terminal-value policy;
- negative-price and degradation policy; and
- solver configuration.

### `OperationalState`

Represents the physical state at an interval boundary:

- UPS energy;
- TES energy;
- IT temperature;
- rack temperature;
- cold-aisle temperature;
- hot-aisle temperature; and
- previous control values if ramp constraints require them.

### `WorkloadState`

Represents work not completed before the current boundary:

- remaining CPU-hours by cohort;
- absolute deadline;
- service/shiftability class; and
- optional committed overlap processing.

### `HorizonInputs`

Contains timestamp-aligned exogenous information:

- monotonic timestamps;
- settlement price;
- optional optimisation price;
- inflexible CPU load;
- new flexible-work arrivals;
- ambient temperature or other disturbances; and
- initial physical and workload states.

### `HorizonResult`

Contains:

- all actions and states over the solve horizon;
- the committed slice;
- boundary state after the committed slice;
- remaining workload at that boundary;
- objective and ex-post settlement cost;
- solver status, bound, incumbent, gap, and runtime; and
- conservation and feasibility audit values.

## 4.2 Clean model API

Replace module-level path redirection and tariff monkey-patching with functions similar to:

```python
def solve_horizon(
    config: ModelConfig,
    inputs: HorizonInputs,
    solver: SolverConfig,
) -> HorizonResult:
    ...
```

File writing and chart generation should remain outside the mathematical model. This will make the model testable in memory and allow independent annual scenarios to execute safely in separate processes.

## 4.3 Rolling annual orchestration

The annual runner should:

1. load and validate one continuous annual source timeline;
2. create the declared initial physical and workload state;
3. construct the current core plus look-ahead forecast;
4. solve the horizon;
5. validate the solution and conservation audits;
6. append only the committed intervals to the annual output;
7. extract the committed boundary physical state and outstanding workload;
8. write an atomic checkpoint containing the new state and fingerprints;
9. advance to the next boundary; and
10. repeat until every annual interval is committed exactly once.

Days must be sequential within one scenario. Different complete scenario chains can be run in parallel.

## 5. Possible paths forward

| Path | Description | Advantages | Disadvantages | Recommendation |
|---|---|---|---|---|
| A. Correct rolling MPC | Refactor the daily model into a state-linked, timestamp-based 24-hour core plus configurable look-ahead | Reuses the fast daily MILP; physically defensible; restartable; supports annual sensitivities | Requires careful state, backlog, terminal, and DST work | **Recommended primary path** |
| B. Transitional independent-day study | Retain the current runner, fix obvious within-day degeneracies, apply one price policy, and label results as independent-day price sensitivity | Fastest route to a transparent interim result; preserves existing outputs | Cannot answer questions about annual physical operation or state continuity | Keep as a secondary analysis, not the final annual model |
| C. Monolithic annual MILP | Solve all 35,040 quarter-hour intervals in one model | Perfect global foresight; no daily handoff logic | Very large MILP; unrealistic foresight; difficult debugging/checkpointing; likely unnecessary | Do not use as the first implementation |
| D. Multi-day block optimisation | Solve 48-72-hour or weekly blocks with overlap and pass states between blocks | Easier continuity than daily; useful benchmark for look-ahead bias | Still needs state/backlog/DST semantics; larger solve | Use as a validation benchmark or fallback |

### Why Path A is best

The current daily solves are already quick. A properly structured rolling controller preserves that computational advantage while providing a defensible physical trajectory. It also matches how operational scheduling would work: optimise with a forecast, implement a limited portion, update state, and solve again.

A monolithic year should be used only for simplified benchmark windows or if later evidence shows that short look-ahead materially biases results.

## 6. Detailed implementation plan

## Phase 0 - Freeze and document the current baseline

**Purpose:** Preserve the existing 363-case result as a regression reference without presenting it as the final annual model.

### Tasks

- Rename its interpretation in documentation to "independent standard-day price analysis".
- Record hashes for the IMRP data, load profile, shiftability profile, parameter file, and current result CSV.
- Record the current environment and solver used to produce the stored outputs, if recoverable.
- Add a compact baseline manifest containing the 363 daily headline results.
- Document the two excluded daylight-saving dates and the 26 price-adjusted cases.
- Preserve the current GBP 546,292.86 nominal and GBP 519,482.12 optimised totals only as regression anchors.

### Acceptance criteria

- The baseline can be regenerated or its provenance limitations are explicit.
- No future rolling result silently overwrites the independent-day result.
- Reports and filenames clearly distinguish independent-day and rolling-year outputs.

## Phase 1 - Normalise state/action indexing and boundary conditions

**Purpose:** Remove free initial state, off-by-one state handoff, missing final transitions, and noncausal first-slot controls.

### Tasks

- Change storage and thermal state sets to boundaries `0..N`.
- Change control/action sets to intervals `0..N-1`.
- Apply every transition as `state[t+1] = f(state[t], action[t])`.
- Fix all initial physical states exactly to `OperationalState` values.
- Remove first-slot "equal to future average" chiller constraints.
- Parameterise or remove daily UPS and TES cyclic terminal equalities.
- Ensure the final action affects the final state.
- Make units explicit in variable names and validation: kW, W, kWh, degC, hours.

### Tests

- One-step UPS charge and discharge balance tests.
- One-step TES charge and discharge balance tests.
- One-step thermal transition tests.
- Initial-state equality tests for every physical state.
- Final-action perturbation test proving that the final state changes correctly.
- No-free-energy test showing that changing the initial TES parameter changes state zero exactly.

### Acceptance criteria

- Every state has one unambiguous physical timestamp.
- All first states match supplied values within tolerance.
- Every action appears in exactly one state transition.
- The retained boundary state can be extracted without asset-specific index rules.

## Phase 2 - Make zero and negative prices physically safe

**Purpose:** Solve all observed price horizons without clipping as a numerical recovery measure.

### Tasks

- Add mutually exclusive TES charge and discharge modes, using binary variables or a validated net-flow formulation.
- Audit all controllable electricity consumption for finite physical upper bounds.
- Decide whether deliberate battery/TES cycling is physically allowed.
- If it is not intended, add a small evidence-based degradation/throughput cost, cycle constraint, or operating rule.
- Add explicit import-capacity constraints if supported by the facility assumptions.
- Separate settlement price from optional optimisation price in the data model.
- Export an LP/MPS and solver log for any signed-price failure before applying a price transformation.
- Compare original, zero-floor, and constant-shift schedules on the 26 affected horizons.

### Tests

- Boundedness on all 26 previously failing horizons.
- No simultaneous UPS or TES charge/discharge above numerical tolerance.
- No unexplained cooling or storage consumption at zero and negative prices.
- Solver objective equals ex-post settlement cost when original price is used.
- Transformed-price cases report true cost using original prices and clearly show schedule differences.

### Decision gate

- Use signed original prices as the primary case if the corrected model solves and behaviour is physically defensible.
- If market participation rules prevent the data centre from receiving negative prices, use an explicit tariff contract rule rather than an unexplained mathematical cap.
- Keep price flooring and shifting as sensitivity cases with uniform application across the whole year.

## Phase 3 - Build a continuous annual timeline including DST

**Purpose:** Ensure every physical source interval is committed once.

### Tasks

- Convert source date and settlement-period fields into a validated continuous timestamp index.
- Store both UTC and Europe/London labels.
- Preserve the 23-hour and 25-hour local dates instead of dropping them.
- Generalise horizon slicing to timestamps rather than fixed local-day row counts.
- Expand hourly prices to quarter-hour resolution through a single tested function.
- Decide how the final look-ahead obtains prices beyond 31 December, such as using actual early-2026 data already present in the dataset.

### Tests

- No duplicate or missing source interval.
- Spring and autumn transition fixtures retain the correct number and ordering of intervals.
- Expansion preserves hourly values and energy-weighted cost.
- The committed 2025 series contains the expected total duration.

### Acceptance criteria

- All 365 calendar dates are represented in reporting.
- Every source interval is used exactly once in the committed optimisation timeline.
- No physical state jumps across a skipped daylight-saving date.

## Phase 4 - Implement workload backlog and deadline carry-over

**Purpose:** Make flexible IT scheduling feasible and conserved across midnight.

### Tasks

- Represent work as timestamped arrival cohorts with remaining CPU-hours and absolute deadlines.
- Include next-day flexible arrivals in the look-ahead forecast.
- Commit only work processed during the retained core.
- Recalculate remaining work at the boundary and pass it into the next horizon.
- Enforce CPU-capacity constraints across carried work, new flexible arrivals, and inflexible demand.
- Decide whether overlap actions are forecasts only or explicitly committed; do not mix both semantics.
- Store a cohort-level or aggregate audit that can prove conservation without creating excessive output volume.

### Recommended semantics

Use receding-horizon backlog semantics:

- the look-ahead schedule is advisory;
- only the core actions are committed;
- unprocessed work remains in the boundary backlog;
- the next solve may reschedule that work, but cannot change its arrival or deadline; and
- missed deadlines are infeasible unless an explicit penalised slack variable is included for diagnostics.

### Tests

- A job arriving at 23:45 with a three-hour deadline is processed exactly once.
- Carried work and new next-day work share the same CPU-capacity constraint.
- Total arrivals equal total completions plus final outstanding work.
- No cohort is processed before arrival or after its deadline.
- A deliberately infeasible workload produces a clear diagnostic rather than silent loss.

## Phase 5 - Create the dedicated rolling-year runner

**Purpose:** Replace sensitivity-module monkey-patching with a production annual API.

### Tasks

- Add a dedicated rolling runner, for example `run_rolling_year.py`.
- Keep `run_imrp_year_sample.py` as the independent-day screening runner.
- Inject paths, tariff arrays, configuration, state, and solver settings directly.
- Run dates or interval blocks sequentially within a scenario.
- Write only committed timestep results to the stitched annual output.
- Save the physical and workload boundary state after every committed block.
- Add atomic checkpoint writes through temporary file plus rename.
- Add checkpoint schema validation and deterministic fingerprints.
- Propagate invalidation downstream when a predecessor state changes.
- Support targeted restart from a selected boundary.
- Add a `--validate-only` mode that checks timeline, config, input hashes, and solver availability without solving.

### Checkpoint contents

- scenario/config hash;
- code revision and dirty-worktree indicator;
- input hashes;
- current boundary timestamp;
- predecessor checkpoint hash;
- physical state;
- workload state;
- solver outcome and quality metrics;
- committed cost and energy audit; and
- schema version.

### Acceptance criteria

- A clean run and a resumed run produce identical stitched outputs.
- Corrupt or incompatible checkpoints are rejected and rerun safely.
- Changing one upstream state invalidates all dependent downstream checkpoints.
- No module-level path or tariff monkey-patching remains in the rolling path.

## Phase 6 - Define baseline and annual cost accounting

**Purpose:** Produce interpretable and reconcilable annual savings.

### Tasks

- Define the Scenario 1 baseline precisely. At present, workload and storage are restricted, but cooling is still optimised; it is therefore not a completely uncontrolled nominal case.
- Run baseline and optimised cases as separate continuous chains with identical exogenous inputs and declared initial states.
- Count each committed interval once.
- Calculate cost on original settlement prices.
- Calculate component costs from the same committed intervals as the headline total.
- Add energy, workload, storage-throughput, and temperature KPIs.
- Define the final-year boundary treatment for storage and outstanding workload.

### Required annual outputs

- stitched quarter-hour result CSV or Parquet;
- annual baseline and optimised costs;
- ratio-of-sums annual saving percentage;
- daily saving distribution statistics as secondary evidence;
- monthly and seasonal cost/energy totals;
- component cost reconciliation;
- UPS and TES throughput and cycles;
- temperature limit and duration statistics;
- workload delayed, delay duration, and deadline compliance;
- negative-price import and cost statistics;
- solver-quality table;
- state-continuity audit; and
- workload-conservation audit.

### Acceptance criteria

- Annual total equals the direct timestep sum.
- Component totals reconcile to headline total within numerical tolerance.
- Baseline minus optimised cost equals reported saving.
- Every daily/monthly/seasonal subtotal reconciles to the annual total.

## Phase 7 - Verification ladder before the full annual run

Run progressively larger tests. Do not begin with a full-year solve after the refactor.

### Level 1: Pure unit tests

- price expansion and DST timestamp construction;
- state transitions and units;
- backlog transitions and deadlines;
- terminal policy calculations;
- checkpoint hashing and schema validation; and
- annual aggregation and component reconciliation.

### Level 2: Two-day synthetic boundary test

- price step across midnight;
- a late flexible job crossing midnight;
- non-default UPS, TES, and thermal states;
- exact handoff from day one to day two; and
- cost counted once across the boundary.

### Level 3: Three-day signed-price test

- negative price before midnight;
- zero price after midnight;
- storage and cooling degeneracy checks;
- original-price cost reconciliation; and
- solver-status and gap capture.

### Level 4: DST transition weeks

- one week around the spring transition;
- one week around the autumn transition;
- continuous timestamps and state; and
- no duplicated or omitted work.

### Level 5: Rolling versus longer-horizon benchmark

- compare a 24+3-hour rolling schedule with a 48-72-hour monolithic solve on selected windows;
- repeat with 3, 6, 12, and 24 hours of look-ahead; and
- quantify committed cost and schedule sensitivity.

### Level 6: 30-day cold and resumed runs

- benchmark model build, solver, post-processing, and I/O time;
- verify restart equality; and
- inspect checkpoint invalidation.

### Level 7: Central 2025 annual chain

- run one baseline chain and one optimised chain;
- complete all coverage, continuity, conservation, cost, and solver audits; and
- freeze this as the central annual result before launching sensitivities.

## Phase 8 - Annual sensitivity and scenario analysis

**Purpose:** Answer the reviewer questions without creating an unmanageable figure set.

### Central scenario

Define one defensible central configuration containing:

- flexible-work share;
- UPS capacity and power limits;
- TES capacity and power limits;
- temperature limits;
- signed-price policy;
- look-ahead length;
- terminal-value policy; and
- load-profile interpretation.

### Screening sequence

1. Use the retained independent-day runner to screen broad parameter ranges cheaply.
2. Run annual one-at-a-time chains for the most important parameters.
3. Examine selected interactions, especially UPS x TES and flexible share x storage.
4. Use a small number of stress scenarios for negative prices and high-price years.
5. Select representative days only after annual results exist.

### Priority annual sensitivities

- flexible/inflexible workload share;
- UPS energy capacity;
- UPS charge/discharge power;
- TES energy capacity;
- TES charge/discharge power;
- cold-aisle temperature limit;
- look-ahead length;
- price treatment;
- terminal-value policy; and
- representative versus date-varying workload.

### Execution strategy

- Keep days sequential within each annual scenario.
- Run different annual scenarios in parallel processes.
- Allocate solver threads so scenario-level parallelism does not oversubscribe the machine.
- Cache the baseline chain where the baseline is unaffected by an optimised-asset parameter.
- Use deterministic scenario IDs derived from resolved configuration hashes.
- Estimate runtime and disk space before each sweep. The current per-day output tree is about 25.6 MB for 363 cases, so retaining every debug CSV for many annual scenarios will scale poorly.

### Reporting strategy

- Use annual totals and a small KPI table for all scenarios.
- Show monthly or seasonal variation for the central case.
- Use tornado or response plots for one-at-a-time sensitivities.
- Use a limited heat map for the most important two-parameter interaction.
- Retain the existing Stage 3 flexibility visualisation for a central scenario and a representative day.
- Select the representative day through a declared distance-to-central-metrics method, not visual preference.

## Phase 9 - Reproducibility, documentation, and release gate

### Tasks

- Add a locked environment definition or reproducible Conda environment.
- Document SCIP installation and supported solver alternatives.
- Expose solver name, time limit, MIP gap, threads, and seed in configuration.
- Record Python, package, and solver versions in run metadata.
- Record git commit and dirty status.
- Record hashes for all canonical inputs.
- Decide whether small canonical inputs should be versioned or managed through DVC/external storage.
- Keep bulk outputs ignored, but commit schemas, compact summaries, and paper-critical result tables.
- Update `README.md` with the independent-day and rolling-year workflows.
- Make report wording mode-aware so a full run is not called a sample trial.

### Final release gate

The annual result is ready for use in the paper only when:

- every 2025 interval is represented once;
- all adjacent physical boundary states match;
- workload is conserved and deadlines are respected;
- signed-price operation is bounded and physically defensible;
- no simultaneous UPS/TES charging and discharging occurs beyond tolerance;
- annual and component costs reconcile;
- every accepted solve has a feasible incumbent and recorded gap/status;
- clean and resumed runs are identical;
- the central result is stable to a reasonable increase in look-ahead; and
- the report states the load, price, terminal, and baseline assumptions explicitly.

## 7. Recommended order of work

### Immediate correctness work

1. Preserve and relabel the current 363-day result.
2. Refactor state indices and exact initial conditions.
3. Remove noncausal first-slot and hard daily cyclic boundary rules.
4. Add TES mode exclusivity and signed-price regression tests.
5. Introduce explicit workload backlog.
6. Build the continuous DST-aware timeline.

### Rolling integration work

7. Add the clean horizon API.
8. Implement the sequential runner and chained checkpoints.
9. Stitch committed intervals and reconcile costs.
10. Validate on two-day, signed-price, DST, and longer-horizon fixtures.

### Research analysis work

11. Freeze the central annual baseline and optimised result.
12. Define annual KPIs and representative-day selection.
13. Run screened annual sensitivities.
14. Generate paper tables, plots, and the Stage 3 representative-day analysis.

## 8. What should not be done

- Do not describe the present 363 independent cases as a continuous 365-day optimisation.
- Do not patch the next day merely by assigning `e_start_kwh` from one stored row; thermal indexing and deferred workload would remain inconsistent.
- Do not keep the first-slot temperature/TES inequalities as rolling initial conditions.
- Do not add a constant to prices and assume the schedule is unchanged without proving total energy invariance.
- Do not mix clipped and original prices within one headline annual result.
- Do not count look-ahead intervals as realised cost and then count the same physical intervals again the next day.
- Do not parallelise days inside one state-linked annual scenario.
- Do not launch a large annual sensitivity sweep until the central rolling chain passes the continuity and conservation gates.
- Do not begin with a monolithic 35,040-step MILP unless smaller benchmark evidence shows that rolling look-ahead is inadequate.

## 9. Best practical path

The recommended practical path is:

1. **Keep the current runner** as an explicitly named independent-day screening tool.
2. **Refactor the mathematical model once** around boundary states `0..N` and actions `0..N-1`.
3. **Add explicit physical and workload carry state** rather than trying to infer continuity from output CSVs.
4. **Use original signed prices** after adding the missing physical constraints, with price flooring and shifting retained as sensitivities.
5. **Build a dedicated sequential rolling runner** with timestamp slicing and chained checkpoints.
6. **Validate against short monolithic windows** to select a defensible look-ahead.
7. **Run the central annual case first**, then parallelise complete annual sensitivity chains.

This path requires more work than passing a few end-of-day values into the existing wrapper, but it addresses the root modelling problems and produces an annual result that is feasible, reproducible, and defensible in peer review.

## 10. First implementation milestone

The first meaningful milestone should be a reproducible two-day rolling demonstration, not a new annual run.

It is complete when:

- every physical state is fixed at the first boundary;
- the day-two initial state exactly matches the day-one committed boundary state;
- a flexible job can cross midnight without being lost or duplicated;
- original negative prices solve without simultaneous TES cycling;
- only committed intervals contribute to realised cost;
- component costs reconcile to total cost;
- a cold run and resumed run are identical; and
- the same two-day case can be compared with a monolithic 48-hour benchmark.

Once this milestone passes, extending the runner over the full timestamp sequence is primarily orchestration and verification rather than another modelling redesign.
