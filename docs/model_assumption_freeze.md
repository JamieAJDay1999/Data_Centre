# Frozen annual-model assumptions

Date frozen: 24 July 2026

This record fixes the interpretation of the completed 2025 rolling-horizon
results. Rejected alternatives would require a new central annual chain and new
full-year sensitivity endpoints.

## Central comparison

The annual baseline is a workload-at-arrival, storage-disabled benchmark with
cost-optimised cooling. It is not the fixed-22.5 C conventional baseline
described in the earlier single-day manuscript. The revised paper must use the
implemented definition and must not compare the annual saving with the earlier
fixed-setpoint result as though they were identical.

Both baseline and optimised cases use a cold-aisle range of 18--23 C. This
common bound avoids giving Scenario 2 extra thermal headroom and remains below
the 27 C ASHRAE recommended maximum cited in the paper.

## Thermal and cooling formulation

The annual model uses the implemented implicit-Euler boundary-state thermal
recursions. The constraint requiring cooling provision to be at least
contemporaneous IT heat is retained as a conservative operating rule. It
prevents the model from using unvalidated heat accumulation as a source of
flexibility; consequently, the reported thermal flexibility is conservative
under this assumption. The paper must state the constraint explicitly.

Outdoor temperature (22 C), chiller COP (5), airflow, heat capacities and heat
transfer coefficients remain constant through the year. The 24-hour workload
shape repeats by local clock time, with the UTC timeline handling both daylight
saving transitions. These are stylised annual assumptions, not measurements of
a specific facility.

## Storage

The first annual interval begins with the UPS at 600 kWh (100% of its usable
nameplate energy) and TES at 500 kWh (50%). Thereafter, both physical states are
passed exactly between daily horizons. No daily cyclic restoration is imposed.

The central case includes no storage throughput/degradation cost, standing
loss, reserve above the 50% UPS lower bound, or terminal-energy credit. These
omissions are disclosed as idealised operating assumptions. The year-end
continuation-value sensitivity in
`reports/terminal_treatment/terminal_treatment_report.md` changes annual cost
by at most 0.000381%, so zero central terminal value is retained.

UPS and TES energy-capacity sensitivities vary energy capacity only. Power
ratings remain fixed, and the starting stored-energy fraction is preserved.

## Workload and horizon

Flexible workload is represented as timestamped cohorts with absolute
deadlines of 0.5, 1, 2 and 3 hours. Physical state and unfinished cohorts pass
between local-day cores. Every solve contains the complete facility load over
the committed core and a 12-interval (three-hour) look-ahead.

At the 2025 year boundary, 0.642875 CPU-h remains outstanding at midnight and
is fully served in the 1 January 2026 look-ahead. Annual cost includes only
settlement intervals in calendar year 2025. The look-ahead proves deadline
feasibility but is excluded from 2025 cost because it also contains ordinary
2026 facility demand.

The final look-ahead uses actual IMRP periods 1--3 from 1 January 2026:
GBP 46.49, 40.19 and 28.99/MWh.

## Prices, objective and numerics

The central price input is the signed 2025 Intermittent Market Reference Price
(IMRP) actual series in GBP/MWh, expanded from hourly to quarter-hour
resolution. Realised settlement cost is always reconciled against the signed
source price. Zero-capped and shifted prices are sensitivities only.

IT power is represented by the four-segment non-uniform DLOG approximation. Its
maximum approximation error is 6.241 kW (0.624% of maximum IT power). The
annual models are solved through Pyomo with HiGHS, using a requested relative
MIP gap of 0.1%, a 300-second horizon limit and a normal maximum accepted gap
of 1%.

The low-flexibility 25 May horizon is retained with an exception note. A
900-second targeted rerun left a 5.652% relative gap because the objective is
close to zero and negative, but the absolute primal-bound width is GBP 0.102.
The committed-day cost changed by GBP 0.002, the downstream state was
unchanged, and every feasibility, continuity and reconciliation audit passed.

## Sensitivity interpretation

Full-year 0.5x and 1.5x endpoints establish the annual effect magnitudes and
ranking. Twelve representative weeks provide the intermediate 0.75x and 1.25x
points and the response shape. Because the sampled central saving is 6.713%
versus 5.049% annually, sampled percentages are not presented as annual
estimates; curves are centred on their respective central case and interpreted
as comparative response shapes.

No additional full-year intermediate cases or factorial interactions are
required for the present paper. The frozen annual ranking is:

1. flexible-workload share;
2. UPS energy capacity;
3. TES energy capacity.
