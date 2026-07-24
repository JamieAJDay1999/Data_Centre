# Monthly representative-week sensitivity analysis

## Purpose

This analysis screens the annual cost result against three physical parameters:

- UPS energy capacity;
- TES energy capacity; and
- flexible workload share.

It uses one complete Monday-Sunday week from each calendar month. The twelve
weeks contain 84 scored days, approximately 23% of the year. Each week is a
separate seven-day rolling chain and starts from the case-specific default
physical state. **No warm-up days are used.** State and outstanding workload are
linked within a selected week but not between months.

The baseline is run once because changing storage capacity or flexible workload
share does not affect the no-flexibility baseline.

## Representative-week selection

Every complete Monday-Sunday week contained within a month is a candidate. For
each candidate and its month, the code calculates:

- mean settlement price;
- settlement-price standard deviation;
- minimum settlement price;
- maximum settlement price; and
- share of intervals with a negative settlement price.

Each characteristic is standardised using its variation across all candidate
weeks in the year. This produces a root-mean-square standardised distance for
every candidate.

Selecting the lowest-distance week in every month independently can
systematically under-sample negative prices because they occur in short
clusters. The twelve weeks are therefore selected jointly. The algorithm finds
the combination whose:

- annualised negative-price share is within 0.25 percentage points of the full
  year's share; and
- annualised mean price is within GBP 0.50/MWh of the full-year mean.

Among combinations meeting both conditions, it chooses the lowest sum of
monthly distance scores. If no combination meets both tolerances, it minimizes
the combined normalized deviations before considering the distance score. This
preserves monthly representativeness without omitting the price regime known to
affect model behaviour or biasing annual cost through a systematically high or
low sampled mean price.

The selection is made once from the unmodified central input data and is reused
for every case. The following audit files are created:

- `selected_weeks.csv`;
- `week_selection_candidates.csv`.

If an output folder already contains a different selection, the runner stops
instead of mixing results based on different sampled weeks.

## Sensitivity cases

| Case | Mode | UPS multiplier | TES multiplier | Flexible-workload multiplier |
|---|---|---:|---:|---:|
| `baseline` | baseline | 1.0 | 1.0 | 1.0 |
| `central` | optimised | 1.0 | 1.0 | 1.0 |
| `ups_min` | optimised | 0.5 | 1.0 | 1.0 |
| `ups_075` | optimised | 0.75 | 1.0 | 1.0 |
| `ups_125` | optimised | 1.25 | 1.0 | 1.0 |
| `ups_max` | optimised | 1.5 | 1.0 | 1.0 |
| `tes_min` | optimised | 1.0 | 0.5 | 1.0 |
| `tes_075` | optimised | 1.0 | 0.75 | 1.0 |
| `tes_125` | optimised | 1.0 | 1.25 | 1.0 |
| `tes_max` | optimised | 1.0 | 1.5 | 1.0 |
| `flex_min` | optimised | 1.0 | 1.0 | 0.5 |
| `flex_075` | optimised | 1.0 | 1.0 | 0.75 |
| `flex_125` | optimised | 1.0 | 1.0 | 1.25 |
| `flex_max` | optimised | 1.0 | 1.0 | 1.5 |

UPS and TES multipliers change energy capacity only. Charge and discharge power
limits remain fixed. Initial stored energy is scaled to preserve the central
initial state-of-charge fraction.

The workload multiplier transfers CPU demand between the flexible and
inflexible pools in every interval while preserving total CPU demand. Flexible
workload is capped at total workload, so the realised increase can be less than
1.5 times in saturated intervals.

## Commands

The commands below are run from the repository root.

Select and inspect the weeks without solving:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --select-only
```

Optionally inspect a case without solving:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case central --dry-run
```

Run the sampled baseline:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case baseline
```

Run the central optimised case:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case central
```

Run the UPS endpoints independently:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case ups_min
python run_monthly_week_sensitivity.py --year 2025 --case ups_max
```

Run the TES endpoints independently:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case tes_min
python run_monthly_week_sensitivity.py --year 2025 --case tes_max
```

Run the flexible-workload endpoints independently:

```powershell
python run_monthly_week_sensitivity.py --year 2025 --case flex_min
python run_monthly_week_sensitivity.py --year 2025 --case flex_max
```

Run the central, UPS and TES cases sequentially in one resumable batch:

```powershell
python run_remaining_monthly_week_cases.py --year 2025
```

The batch wrapper skips completed cases, resumes partial cases through their
daily checkpoints, stops on the first failure, and runs the complete comparison
after all five cases finish.

Run all six intermediate cases in a resumable batch:

```powershell
python run_intermediate_monthly_week_cases.py --year 2025
```

Completed block checkpoints are resumable. Re-running the same command resumes
completed weeks rather than solving them again.

Compare every case completed so far:

```powershell
python summarise_monthly_week_sensitivity.py --year 2025
```

Require all seven optimised cases to be present:

```powershell
python summarise_monthly_week_sensitivity.py --year 2025 --require-all
```

## Annualisation

The selected week's cost and energy are multiplied by:

\[
w_m =
\frac{\text{number of quarter-hour intervals in month }m}
     {\text{number of quarter-hour intervals in its selected week}}.
\]

Using interval counts handles the March and October daylight-saving changes.
The comparison script calculates annual saving from the weighted totals:

\[
100
\frac{\hat C_{\mathrm{baseline}}-\hat C_{\mathrm{optimised}}}
     {\hat C_{\mathrm{baseline}}}.
\]

It does not average weekly saving percentages.

The sampled maximum grid import is reported as a sample diagnostic and is not
annualised. It must not be described as the annual peak unless the selected
weeks happen to contain the true annual maximum.

## Outputs

Outputs are written under:

`static/data/monthly_week_sensitivity/2025/`

Each case contains:

- `blocks/`: the twelve independently initialised weekly rolling runs;
- `monthly_results.csv`: raw and annualised monthly contributions;
- `case_summary.json`: annual estimates, solver quality and runtime.

After comparison, `sensitivity_comparison.csv` contains the estimated annual
cost, saving, energy and solver-quality results for every completed optimised
case. When the completed reformulated full-year outputs are present,
`sampling_validation.csv` also compares the sampled baseline and central
results with their full-year values.

## Required validation

Before using the endpoint results in the paper, compare the sampled central
baseline and optimised estimates with the completed full-year results. The
sampling approach should be reconsidered if it:

- changes the direction or ranking of important effects;
- misses negative-price exposure materially;
- differs from annual cost or energy by more than approximately 1%; or
- differs from the known 5.049% central annual saving by more than about
  0.2-0.3 percentage points.

The absence of warm-up days is an explicit design assumption. Because every
selected week starts from the default physical and workload state, the paper
must state this limitation when presenting the sampled sensitivity results.
