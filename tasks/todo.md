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
