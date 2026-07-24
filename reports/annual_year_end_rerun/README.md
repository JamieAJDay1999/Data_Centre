# 2025 annual year-end rerun

All ten completed 365-day scenarios were checked from the closing physical
state and carried workload recorded on 30 December 2025.

The final-day horizon uses 96 committed quarter-hours for 31 December followed
by 12 look-ahead quarter-hours sourced from periods 1, 2, and 3 on 1 January
2026. The three hourly IMRP values are GBP 46.49, 40.19, and 28.99/MWh.

Eight reformulated/current scenarios reproduce their stored 31 December
results within a numerical tolerance of 1e-8. The two pre-reformulation
scenarios differ if they are solved with the current formulation, as shown in
`comparison.csv`. Repeating those two solves with their recorded historical
model revision reproduces both stored results within the same tolerance; the
evidence is in `legacy_revision_verification.json`.

The source annual output chains were not overwritten because every stored
final-day result was shown to have already used the correct 1 January 2026
look-ahead.
