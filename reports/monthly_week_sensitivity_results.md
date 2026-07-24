# Monthly representative-week sensitivity results

Run date: 24 July 2026

## Scope

The analysis uses twelve selected Monday-Sunday weeks, one per month, with no
warm-up days. Each monthly week starts from the case-specific default state.
The sampled baseline and thirteen optimised parameter settings cover 84 scored
days per case.

## Results

| Case | Estimated annual cost (GBP) | Saving vs sampled baseline (GBP) | Saving (%) | Change from central (percentage points) |
|---|---:|---:|---:|---:|
| Central | 514,617.73 | 37,032.70 | 6.713 | 0.000 |
| Flexible workload 0.5x | 525,881.91 | 25,768.52 | 4.671 | -2.042 |
| Flexible workload 0.75x | 520,049.61 | 31,600.82 | 5.728 | -0.985 |
| Flexible workload 1.25x | 509,782.02 | 41,868.41 | 7.590 | +0.877 |
| Flexible workload 1.5x | 505,373.40 | 46,277.03 | 8.389 | +1.676 |
| UPS capacity 0.5x | 518,545.18 | 33,105.25 | 6.001 | -0.712 |
| UPS capacity 0.75x | 516,566.35 | 35,084.09 | 6.360 | -0.353 |
| UPS capacity 1.25x | 512,715.26 | 38,935.17 | 7.058 | +0.345 |
| UPS capacity 1.5x | 510,906.42 | 40,744.01 | 7.386 | +0.673 |
| TES capacity 0.5x | 516,512.56 | 35,137.87 | 6.370 | -0.343 |
| TES capacity 0.75x | 515,490.18 | 36,160.25 | 6.555 | -0.158 |
| TES capacity 1.25x | 514,104.00 | 37,546.43 | 6.806 | +0.093 |
| TES capacity 1.5x | 513,762.19 | 37,888.24 | 6.868 | +0.155 |

The sampled ranking is:

1. flexible workload share;
2. UPS energy capacity;
3. TES energy capacity.

The endpoint spans are 3.718 percentage points for flexible workload, 1.385
percentage points for UPS capacity, and 0.499 percentage points for TES
capacity. All three five-point response curves are monotonic. The incremental
saving from each successive 0.25x increase declines:

- flexible workload: +1.057, +0.985, +0.877 and +0.799 percentage points;
- UPS capacity: +0.359, +0.353, +0.345 and +0.328 percentage points;
- TES capacity: +0.185, +0.158, +0.093 and +0.062 percentage points.

This confirms diminishing marginal returns, strongest for TES and flexible
workload. The UPS response is comparatively close to linear over the tested
range.

## Solver quality

All cases completed and no horizon exceeded the configured 1% accepted-gap
threshold. Every horizon was optimal except 21 January in the 1.5x TES case,
which reached the 300-second limit with a feasible incumbent and a 0.996% gap.
All workload, state, cost-reconciliation, and charge/discharge-overlap audits
passed for that horizon.

The five-case batch took 845 seconds (14.1 minutes) of wall-clock time. The 1.5x
TES case accounted for 407 seconds because of the single time-limited horizon.
The six intermediate cases took a further 650 seconds (10.8 minutes); all six
cases were fully optimal.

## Full-year validation

| Metric | Sampled | Full year | Error |
|---|---:|---:|---:|
| Baseline cost (GBP) | 551,650.43 | 550,503.59 | +0.208% |
| Baseline energy (kWh) | 6,717,040 | 6,717,862 | -0.012% |
| Central optimised cost (GBP) | 514,617.73 | 522,707.23 | -1.548% |
| Central optimised energy (kWh) | 6,742,829 | 6,794,102 | -0.755% |
| Central saving (%) | 6.713 | 5.049 | +1.664 percentage points |

The price-based week selection reproduces baseline cost and energy well, but it
does not reproduce the full-year optimised saving closely enough for the
sampled percentages to be presented as validated annual estimates. The
independent monthly state resets and the concentration of flexibility value in
particular operating regimes are both plausible contributors; this experiment
does not isolate their individual effects.

The endpoint results are therefore suitable as a screening result for the
direction and relative importance of parameters. The absolute annualised saving
percentages require either a bias correction supported by additional validation
or selected full-year endpoint runs before being used as annual claims in the
paper.

## Machine-readable outputs

- `static/data/monthly_week_sensitivity/2025/sensitivity_comparison.csv`
- `static/data/monthly_week_sensitivity/2025/sampling_validation.csv`
- `static/data/monthly_week_sensitivity/2025/remaining_cases_batch_summary.json`
- `static/data/monthly_week_sensitivity/2025/intermediate_cases_batch_summary.json`
