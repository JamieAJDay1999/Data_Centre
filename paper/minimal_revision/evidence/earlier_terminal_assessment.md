# Year-end terminal-treatment assessment

The final 2025 horizon starts from the exact closing physical state and carried
workload recorded on 30 December. Its 12 look-ahead intervals use settlement
periods 1--3 on 1 January 2026.

## Workload accounting

- Workload outstanding at midnight: 0.642875 CPU-h.
- Workload completed during the three-hour look-ahead: 0.642875 CPU-h.
- Core workload unserved after the look-ahead: 0 CPU-h.
- The full 1 January look-ahead uses 2313.104 kWh and costs GBP 89.542; this is planning evidence and is not included in the committed 2025 accounting.

The annual convention is therefore to report costs for settlement intervals
inside calendar year 2025 and disclose the midnight backlog separately. The
look-ahead proves service completion but is not added to 2025 cost because it
also contains ordinary 1 January 2026 facility demand.

## Storage terminal sensitivity

The zero-terminal-value rerun reproduces the stored 31 December committed cost
to GBP 2.27e-13. Price-derived continuation values
were then calculated from the median and maximum 1 January 2026 IMRP, adjusted
for UPS discharge efficiency and TES/chiller conversion.

Across those continuation-value cases, the largest change in committed annual
cost is 0.000381% of the annual total. This is below 0.01%, so the
reported 5.049% annual saving is not materially driven by final-horizon storage
depletion. Zero central terminal value is retained as an explicit modelling
assumption; continuation values remain a terminal sensitivity.

## Frozen convention

1. Commit and account for calendar-year intervals only.
2. Use actual 1 January 2026 periods 1--3 for the final look-ahead.
3. Require all workload arriving in 2025 to be serviceable within its deadline;
   report midnight backlog and verify it is zero after the look-ahead.
4. Retain zero storage terminal value centrally because the price-derived
   sensitivity is immaterial, and disclose the assumption.
