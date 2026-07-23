# Negative-price MILP reformulation

**Implemented:** 23 July 2026  
**Scope:** Linked rolling-year model only (`rolling_optimisation/`)  
**Solver comparison:** Deliberately deferred at Jamie's request.

## Decision

The central case continues to optimise against the original signed settlement
price. Price flooring and shifting are available only as explicit sensitivity
settings. Settlement cost is always calculated from the original signed price
column, so transformed-price runs cannot silently become headline results.

The completed pre-change 2025 run established that negative-price horizons were
the runtime bottleneck: 28 of 365 horizons used 85.3% of solver time, and all
three time-limit terminations occurred on negative-price days.

## Active formulation changes

### IT power curve

The previous model used a uniform 10-segment convex-combination formulation
with ten one-hot binaries per interval. The active optimised model uses:

- four non-uniform segments;
- breakpoints generated as `linspace(0, 1, 5) ** 1.5`, concentrating resolution
  where the `CPU ** 1.32` curve has greatest curvature; and
- Pyomo's logarithmic disaggregated convex-combination (`DLOG`) representation,
  requiring two IT-curve binaries per interval.

The new piecewise-linear curve overestimates the underlying convex analytical
curve by at most **6.241 kW**, or **0.624% of maximum IT power**. The former
10-segment curve had a maximum error of 4.061 kW. This modest loss of
approximation accuracy is explicit and must be included in the paper's model
sensitivity checks. Eight- and sixteen-segment formulations remain selectable
from the command line.

For baseline mode, CPU utilisation is fixed. Its IT power is therefore
interpolated directly and no IT-curve binaries are created.

### Storage modes and bounds

- UPS charging/discharging now uses one binary mode variable per interval
  instead of two.
- TES charging/discharging now uses one binary mode variable per interval
  instead of two.
- UPS discharge is explicitly bounded by maximum IT power rather than the
  looser 2,700 kW equipment value.
- IT grid power, UPS charging and storage flows have explicit tight bounds.
- Grid import has an equipment-derived structural upper bound of 1,723.095 kW.
- A lower contractual grid-import limit and a higher UPS reserve can be supplied
  through configuration when evidence is available.

A standard 108-interval optimised horizon now has **432 binaries**, down from
approximately **1,512**. The baseline horizon has no binary variables.

### Physical operating costs and terminal values

The objective now supports:

- UPS throughput cost in GBP/kWh;
- TES throughput cost in GBP/kWh-thermal;
- terminal UPS energy value;
- terminal TES energy value;
- a configurable UPS reserve; and
- a configurable grid-import limit.

All new economic values default to zero. No defensible degradation or terminal
valuation data exists in the repository, so inserting assumed values into the
central case would be less valid than retaining the documented zero-cost
assumption. These parameters should be calibrated before the paper's final
central run and then tested as sensitivities.

### Solver quality and end-of-year workload

- HiGHS is asked not to load a solution until a finite incumbent has been
  confirmed.
- Every result records whether a feasible incumbent exists, its relative gap,
  whether it meets the requested gap, whether it meets the separately declared
  maximum accepted gap, its quality label, and the model's binary count.
- The annual summary lists dates exceeding the accepted gap.
- `--fail-on-gap-exceeded` prevents an unacceptable horizon from being
  checkpointed.
- Console output uses `optimal`, `time_limit_gap_accepted`, or
  `time_limit_gap_exceeded` rather than describing every incumbent as solved.
- Work remaining at a committed day boundary is checked against the complete
  three-hour look-ahead plan. The annual summary now distinguishes boundary
  backlog from work genuinely unserved after the planned look-ahead. This
  explains the expected 31 December carryover without treating it as lost work.
- Checkpoint/fingerprint schema is now version 2, preventing old and revised
  formulation checkpoints from being mixed.

## Price treatments

`--price-treatment signed` is the default and the only recommended central
case. The alternatives are:

- `floor_zero`: optimise with negative prices set to zero;
- `shift_year_min`: add the absolute annual minimum to every optimisation
  price.

Both retain the original settlement column for external cost calculation.
They remain counterfactual sensitivities because prior controlled tests proved
that they change energy use and dispatch.

## Timing benchmark

`run_rolling_formulation_benchmark.py` reuses the completed old run's opening
physical states and workload backlogs. This isolates formulation performance:
the date, signed prices, opening state and workload state are identical. The
revised cases below used HiGHS, a 60-second limit and a requested 0.1% gap.

| Date | Old runtime | Revised runtime | Revised gap | Result |
|---|---:|---:|---:|---|
| 5 April | 25.14 s | 1.32 s | 0.0028% | Optimal |
| 25 May | 300.32 s | 60.19 s | 2.2278% | 60 s limit; above 1% acceptance |
| 22 June | 133.62 s | 3.24 s | 0.0049% | Optimal |
| 23 June | 163.03 s | 3.35 s | 0.0909% | Optimal |
| 5 August | 173.49 s | 4.41 s | 0.0494% | Optimal |
| 7 September | 300.37 s | 7.35 s | 0.0996% | Optimal |
| 4 October | 300.23 s | 60.20 s | 0.7107% | 60 s limit; accepted |
| 5 October | 280.97 s | 4.66 s | 0.0569% | Optimal |

Across these eight difficult horizons, recorded runtime fell from 1,677.17 to
144.73 seconds, an 11.6-fold reduction. For the six cases that reached the
requested solver gap, runtime fell from 1,076.62 to 24.33 seconds, a 44.2-fold
reduction.

The extreme 25 May case was also run for 300 seconds. Its gap was 0.593%,
compared with 3.216% for the old formulation at the same limit. It therefore
meets the new 1% acceptance threshold, although it does not meet the requested
0.1% solver gap.

The generated detailed results are in
`reports/rolling_formulation_benchmark.csv`.

## Correctness tests run

```powershell
python -m pytest -q
python -m pytest -q -m integration
```

Results:

- 5 unit tests passed;
- 2 integration tests passed, covering signed linked horizons and optional
  physical-cost/limit reconciliation;
- state continuity, workload conservation, signed settlement cost, storage
  exclusivity, feasible-incumbent reporting, binary count and curve-error
  checks passed;
- a two-day optimised runner smoke test completed with both horizons optimal,
  no accepted-gap failures and zero workload unserved after look-ahead;
- a two-day baseline smoke test solved in 0.17 seconds total with no binary
  variables in the model.

`pytest.ini` now limits discovery to `tests/`; previously, a script named
`Archive/dc_diagram_test.py` aborted repository-wide test collection because
its optional `diagrams` dependency was absent.

## How to run

Central signed-price run with strict checkpoint quality:

```powershell
python run_rolling_year.py --year 2025 --mode optimised `
  --scenario-id 2025_optimised_reformulated `
  --fail-on-gap-exceeded
```

Repeatable hard-day benchmark:

```powershell
python run_rolling_formulation_benchmark.py --time-limit 60
```

Higher-resolution IT-curve sensitivity:

```powershell
python run_rolling_year.py --year 2025 --mode optimised `
  --scenario-id 2025_optimised_dlog8 `
  --it-power-segments 8 --it-power-representation DLOG `
  --it-power-breakpoint-exponent 1.5
```

Physical-cost example, only after values have been justified:

```powershell
python run_rolling_year.py --year 2025 --mode optimised `
  --scenario-id calibrated_physical_costs `
  --ups-throughput-cost-gbp-per-kwh <value> `
  --tes-throughput-cost-gbp-per-kwh-th <value> `
  --terminal-ups-value-gbp-per-kwh <value> `
  --terminal-tes-value-gbp-per-kwh-th <value>
```

## Required next validation

Do not overwrite the completed old chain. Run the revised annual case under a
new scenario ID. Before using it in the paper:

1. compare annual cost, energy and schedules under 4-, 8- and former 10-segment
   curve formulations;
2. calibrate or explicitly retain zero degradation and terminal values;
3. run with `--fail-on-gap-exceeded`;
4. confirm that the final boundary backlog is fully served within the planned
   look-ahead; and
5. report both the requested 0.1% solver gap and the separate 1% maximum
   acceptance policy.
