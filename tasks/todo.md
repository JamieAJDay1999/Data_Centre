# Price Scenario Sensitivity

- [x] Inspect current sensitivity analysis and price input flow.
- [x] Add a replaceable price-scenario input layer with dummy representative days.
- [x] Replace price-volatility multiplier cases with price-day scenario cases in the sweep.
- [x] Keep flex share, TES, UPS, and cold-aisle cases on the existing OAT methodology.
- [x] Update plotting to label price scenarios without implying a multiplier.
- [x] Verify with compile and dry-run checks.

## Notes

- Existing `rev_stack` working-tree changes are unrelated and were left untouched.

## Results

- Added named dummy price-day scenarios under `inputs/price_scenarios_dummy.csv`.
- Tier 1/Tier 2 sensitivity cases now use `price_day` scenarios instead of price-volatility multipliers.
- Other sensitivity variables still use the existing OAT multiplier/absolute-temperature setup.
- Verified with `py_compile`, price loader checks, Tier 1/Tier 2 dry-runs, and a synthetic plotting smoke test.

# IMRP Rolling-Year Plot

- [x] Confirm the CSV date range and rolling-year boundary.
- [x] Add a reusable plotting script under `plotting_and_saving/`.
- [x] Generate a standard PNG under `static/images/`.
- [x] Verify all annual legend labels, the partial final year, and rendered layout.

## Results

- Created `plotting_and_saving/plot_imrp_actuals_by_year.py`.
- Generated `static/images/imrp_actuals_by_rolling_year.png` from all 87,960 observations.
- Verified 11 legend entries: ten complete rolling years and a partial 2026–27 series.
- Visually checked the rendered PNG for readable axes, lines, and legend placement.

# IMRP EDA and Year-Selection Report

- [x] Confirm the dataset definition, units, coverage, and relationship to the paper's day-ahead price input.
- [x] Audit completeness, duplicates, daylight-saving periods, and numerical integrity.
- [x] Analyse long-run regimes, annual distributions, seasonality, intraday profiles, ramps, negative prices, and extremes.
- [x] Compare calendar-year and first-timestep-anchored selection strategies using explicit representativeness scores.
- [x] Recommend a primary year and sensitivity design for the paper revision.
- [x] Generate a self-contained HTML report with embedded annotated figures and reproducible methodology.
- [x] Render and visually verify the complete report.

## Acceptance criteria

- The report uses all observations in `static/data/imrp_actuals.csv` and states any exclusions.
- Every selection recommendation is tied to a transparent criterion rather than visual judgement alone.
- The HTML opens without network access and contains its figures as embedded data.
- The report distinguishes a one-year study horizon from the paper's current 24-hour illustrative profile.

## Results

- Audited all 87,960 hourly observations spanning 30 June 2016 to 12 July 2026: no missing dates, null source fields, or duplicate date-period keys.
- Confirmed 3,645 normal 24-period days plus ten 23-period and ten 25-period daylight-saving days.
- Compared nine complete calendar years (2017–2025) and ten complete 30-June-anchored windows.
- Selected calendar year 2025 as the recent-regime representative; 2024 ranks second, while 2022 is a distinct high-price stress regime.
- Identified reproducible 2025 daily profiles for typical, high-volatility, negative-price, and high-mean-price cases.
- Generated `reports/imrp_eda_year_selection_report.html` with seven embedded annotated figures, two detailed tables, methodology, limitations, and paper-revision guidance.
- Verified the report in a browser at desktop resolution with zero console errors and no external image, script, or stylesheet dependencies.

# IMRP 2025 Model Trial Wrapper

- [x] Identify reusable Scenario 1, Scenario 2, checkpoint, and aggregation functions.
- [x] Select 50 evenly spaced standard 24-period days from calendar year 2025.
- [x] Supply the actual following-day first three prices to the model extension horizon.
- [x] Wrap the existing nominal and optimised cost workflows without duplicating model code.
- [x] Add restartable per-day execution and failure capture.
- [x] Aggregate daily results into median, IQR, 5th–95th percentile, and component-level summaries.
- [x] Remove flexibility-duration execution, aggregation, CLI staging, and report content.
- [x] Verify the wrapper with compile checks, pure-function assertions, and a solver-free dry run.

## Acceptance criteria

- Default execution targets 50 evenly spaced dates; `--sample-days all` expands to all 363 standard days.
- The two daylight-saving dates are excluded from fixed 24-hour runs but remain documented.
- Scenario outputs are isolated by date and existing completed dates are resumed unless `--force` is used.
- Default execution calls only Scenario 1 nominal calculation and Scenario 2 optimisation; `--report-only` rebuilds outputs without a solver.
- No 50-day solver run is launched during implementation verification.
- Summary CSV, JSON metadata, HTML, and a sample manifest are written to discoverable project paths.

## Results

- Added `sensitivity_analysis/run_imrp_year_sample.py`, reusing
  `sensitivity_analysis.sensitivity_sweep.run_tier1_case()` rather than
  duplicating the nominal and optimisation model logic.
- Confirmed 363 eligible 2025 days and selected 50 unique dates spanning 1 January to 31 December at 7–9 day intervals.
- Excluded only 30 March (23 periods) and 26 October (25 periods) from fixed 24-hour runs.
- Verified the 108 model steps contain the selected day's 96 quarter-hours followed by 12 quarter-hours from the actual next day.
- Added per-date resume checkpoints, failure tracebacks, component savings, and aggregate median/IQR/P05/P95 outputs.
- Added a self-contained result figure covering seasonal savings and component contributions; it is embedded directly in the HTML report after results exist.
- Generated the solver-free manifest and report under `reports/imrp_annual_sample/` without launching a new 50-day trial; one existing Scenario 1/2 checkpoint is included and will be resumed.
- Passed compile, sampling, horizon-boundary, cost-only execution, aggregation, embedded-figure, CLI, metadata, and report-only checks.
- Environment note: the existing pipeline requires SCIP, which is not currently available on `PATH`; the wrapper stops before model execution until that prerequisite is restored.

# IMRP 2025 Full-Year Failure Recovery

- [x] Inventory failed dates, tracebacks, partial outputs, and completed checkpoints.
- [x] Determine whether failures arise from inputs, model feasibility, result parsing, or solver behaviour.
- [x] Add a per-horizon option that clips negative prices to zero without modifying the source CSV.
- [x] Retry only dates without a completed Scenario 1/2 checkpoint.
- [x] Verify all 363 fixed-horizon days and regenerate the full-year report.

## Acceptance criteria

- The cause of all 26 reported failures is evidenced from stored tracebacks or solver outputs.
- Existing successful dates are not rerun.
- The combined report records that only recovered dates use the zero price floor.
- Retried dates either complete or retain a precise, date-specific failure record.
- Full-year aggregate artifacts accurately reflect the final checkpoint state.

## Results

- Confirmed that all 26 failed 27-hour horizons contained negative prices, while none of the 337 successful horizons did.
- Added `--floor-negative-prices`, which clips prices to GBP 0/MWh only after each selected day plus its three-hour extension has been loaded into memory; `static/data/imrp_actuals.csv` remains unchanged.
- Preserved and resumed all 337 completed checkpoints, then successfully recovered only the 26 missing dates.
- Verified 363 checkpoints, 363 unique daily result rows, 26 price-adjusted rows, and zero remaining recorded failures.
- Regenerated the combined CSV, distribution summary, metadata, manifest, date audit, and self-contained HTML report under `reports/imrp_annual_sample/`.
- Recorded the original negative-price count, clipped-price count, and applied floor in the combined results so the recovery treatment is explicit for paper reporting.

# 365-Day Optimisation Forward Plan

- [x] Identify the current full-year entry points and required runtime dependencies.
- [x] Trace the complete daily nominal and optimised calculation path.
- [x] Audit rolling-horizon state continuity, workload carry-over, negative prices, and annual aggregation.
- [x] Review configuration, checkpointing, testing, observability, and sensitivity-sweep support.
- [x] Compare viable implementation paths and recommend a phased route forward.
- [x] Write and validate a detailed Markdown plan in the project root.

## Acceptance criteria

- Every current-code claim is tied to a concrete file, function, or stored run artifact.
- The plan separates correctness-critical work from performance, analysis, and presentation improvements.
- Alternative paths include trade-offs, prerequisites, risks, and decision gates.
- Each phase has specific deliverables and testable completion criteria.

## Results

- Created `365-Day Optimisation Code Review and Forward Plan.md` in the project root.
- Confirmed that the current annual workflow solves 363 independent 27-hour cases rather than one state-linked year.
- Verified stored-output evidence of free first-slot state selection and simultaneous TES charging/discharging.
- Recommended a sequential rolling-MPC architecture with boundary-indexed physical states, workload backlog, DST-complete timestamps, signed-price safeguards, and chained checkpoints.
- Included transitional and monolithic alternatives, phased implementation tasks, decision gates, validation fixtures, annual KPIs, and sensitivity execution guidance.
- Validated the Markdown structure, ASCII encoding, file status, and whitespace with repository checks.

# Rolling-Year Optimisation Implementation

- [x] Add explicit serialisable physical and workload boundary states.
- [x] Build a continuous UTC/local 2025 quarter-hour timeline including both DST days.
- [x] Implement a boundary-indexed horizon MILP with TES/UPS mode exclusivity.
- [x] Carry physical state and deferred workload between sequential daily solves.
- [x] Add original-price settlement-cost accounting and committed-only outputs.
- [x] Add fingerprinted daily checkpoints and deterministic resume behaviour.
- [x] Add a `rolling_optimisation/run_rolling_year.py` CLI while retaining the
  independent-day runner.
- [x] Verify two-day handoff, signed prices, DST coverage, workload conservation, and cost reconciliation.
- [x] Document commands, outputs, solver behaviour, and known first-version limitations.

## Acceptance criteria

- A scenario is solved as sequential linked daily horizons, never as independent days.
- Every committed 2025 quarter-hour appears exactly once, including clock-change dates.
- All physical boundary states pass exactly from one daily solve to the next.
- Flexible workload is neither lost nor duplicated at a day boundary.
- Original signed prices are accepted and realised cost is calculated from committed grid import.
- Clean and resumed short runs produce numerically identical stitched outputs.

## Results

- Added `rolling_optimisation/` and its `run_rolling_year.py` entry point.
- Validated all 35,040 quarter-hours and 365 local dates, including 92- and
  100-interval clock-change days.
- Replaced the unstable 15-minute explicit thermal step with a linear
  implicit-Euler step so exact physical initial states are feasible.
- Passed a linked two-day signed-price run and deterministic checkpoint resume:
  zero state residual, zero UPS/TES overlap, workload residual below
  3e-15 CPU-hours, and settlement-cost reconciliation within 2e-12 GBP.
- Solved both daylight-saving days and the difficult 4 October negative-price
  case; time-limited incumbents retain their solver bound and recorded gap.
- Added three default unit/data tests plus an opt-in solver integration test.
