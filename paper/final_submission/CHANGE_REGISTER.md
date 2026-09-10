# Corrected-results change register - 10 September 2026

The active manuscript is `paper/final_submission/main.tex`, with `supplement.tex`. Earlier drafts and the pre-edit submission are preserved. No annual or event optimisation was rerun during this paper update.

## Principal numerical changes

| Quantity | Previous result | Corrected result |
|---|---:|---:|
| Reference cost (GBP) | 550,503.59 | 550,503.59 |
| Central cost (GBP) | 522,702.95 | 520,876.68 |
| Annual saving (GBP) | 27,800.64 | 29,626.91 |
| Annual saving (%) | 5.050039 | 5.381784 |
| Central grid electricity (GWh) | 6.793996 | 6.788545 |
| Central peak import (kW) | 1,681.94 | 1,682.51 |

The exact increase in annual saving is GBP 1,826.27, or 0.331745 percentage points. This corrects the completion note's comparison against a rounded GBP 27,800 baseline saving. The representative day remains 22 September 2025; its selection score changes from 0.13487744 to 0.23481769.

## All annual scenarios

| Case | Previous saving (%) | Corrected saving (%) | Corrected annual cost (GBP) |
|---|---:|---:|---:|
| Baseline | 0.000 | 0.000 | 550,503.59 |
| Flexible workload 0.5x | 3.522 | 3.737 | 529,933.98 |
| Flexible workload 0.75x | 4.323 | 4.604 | 525,156.86 |
| Flexible workload 1.0x | 5.050 | 5.382 | 520,876.68 |
| Flexible workload 1.25x | 5.694 | 6.087 | 516,996.94 |
| Flexible workload 1.5x | 6.257 | 6.633 | 513,988.68 |
| UPS capacity 0.5x | 4.565 | 4.897 | 523,543.71 |
| UPS capacity 0.75x | 4.814 | 5.144 | 522,187.20 |
| UPS capacity 1.25x | 5.285 | 5.613 | 519,601.93 |
| UPS capacity 1.5x | 5.501 | 5.829 | 518,416.79 |
| TES capacity 0.5x | 4.800 | 5.129 | 522,267.82 |
| TES capacity 0.75x | 4.937 | 5.269 | 521,496.14 |
| TES capacity 1.25x | 5.120 | 5.450 | 520,501.98 |
| TES capacity 1.5x | 5.167 | 5.499 | 520,228.83 |

## Inputs and table placement

- Restored all 24 intended hourly tranche distributions, with each repeated across four consecutive quarter-hours. Confirmed exact agreement with the first-edit source table.
- Added supplementary Table S3 for the hourly distributions and regenerated Table S4 with all 96 combined rows. The earlier Table 3/Table 4/S1.1 discussion now maps to Section S2.1.
- Kept the quarter-hour utilisation inputs used in the completed runs. Explained explicitly that 0.40 at midnight is an hourly starting value while the first-hour mean is 0.37; neither replaces the four input values.
- Rebuilt annual cost components, endpoints, solver-quality and both UPS financial-screen tables from unrounded corrected results.

## Changes to physical interpretation

| Example | Previous manuscript | Corrected manuscript |
|---|---|---|
| Noon reduced import | 100 and 150 kW for six hours | 100 kW for six hours; 150 kW example is five hours from 13:00 |
| 06:00 reductions (100/200 kW) | 1.25 h, entirely UPS | 1 h, predominantly UPS with small IT/cooling changes |
| 15:00 reductions | 200 kW entirely IT/cooling; one UPS interval for 100 kW | Both last 3 h; IT/cooling dominate, with mean UPS contributions about 6/5 kW |
| 06:00 increased import | 25 kW at least 11.25 h; 75 kW 7.75 h | 25 kW at least 10.25 h; 75 kW 7.25 h |
| 25 kW twelve-hour starts | 03:00, 10:00, 11:00, 12:00 | Also includes 04:00 and 08:00 |
| Unresolved adjacent duration trials | 13 | 8 |

The abstract, highlights and conclusion now quote the supported 100 kW six-hour result. Duration results remain certified feasible responses, not a claim of global maximum duration in every cell. Eight lower-bound markers are retained.

## Figures and captions

- Refreshed Figures 4a/4b/5/6/7/8/9 from stored corrected solutions and copied them into the submission. In the compiled paper these are Figures 3(a/b), S1, 4, 5, 6 and 7 respectively.
- Removed inherited vertical-axis limits that clipped the new midday peaks in import, utilisation and the component stack.
- Added certified-duration labels and lower-bound markers to event panels; dashed targets now stop at event end. Captions distinguish unplotted recovery from grey alignment space.
- Adjusted component-plot margins to keep labels inside the image. Preserved the established colours, typography and figure roles.
- Figure saving now renders a temporary asset and replaces the final file atomically, avoiding intermittent Windows preview-file write failures.

## Economic interpretation

Annual electricity increases by 1.052179% while peak import increases; lower cost is not an energy-saving or network-benefit claim. The component-cost changes are IT -GBP 16,722.17, direct CRAC -GBP 14,425.46, TES charging +GBP 6,897.29, UPS net -GBP 5,376.57, with auxiliary cost unchanged. These are accounting contributions, not standalone asset profits.

UPS enlargement from 600 to 750/900 kWh adds GBP 1,274.75/2,459.89 per year. At ten years and 8%, supported incremental investments are GBP 8,554/16,506, or GBP 57.02/55.02 per added nameplate kWh. Updated all rate/life combinations. Financial assumptions and their limitations are unchanged.

## Solver and terminal qualifications

- Updated central non-optimal horizons to 25 May, 6 August and 4 October, with gaps 0.868%, 0.228% and 0.612%.
- Disclosed the three corrected 25 May annual sensitivity exceptions: 0.5 workload/UPS/TES have gaps 1.301%/1.111%/1.009%, with daily bound widths GBP 0.103/0.329/0.359. These daily bounds are not annual error guarantees.
- Removed superseded longer-limit solver checks from the current-result narrative and moved their CSV under historical evidence.
- Updated the central year-end residual to 0.787115 CPU-hours, all served in the planned continuation. That continuation contains 2,531.359 kWh and GBP 98.275 including January demand; its whole cost is excluded from annual settlement.
- Earlier terminal sensitivity evidence is explicitly historical and does not certify the corrected central or capacity cases.

## Verification and deliverables

- Verified all 14 annual settlements directly from 35,040 committed rows each, continuous quarter-hour timestamps, 365-day coverage, 5,110 checkpoint fingerprints/hash links and daily CSV checksums.
- Confirmed recorded physical handoff, workload, storage-mode and year-end service checks.
- Reconciled component sums and target tracking for all 148 positive event dispatches; confirmed the 288-cell grid and eight unresolved boundaries.
- Verified the package asset hashes, numerical claims, sensitivity monotonicity and TeX file dependencies. Both documents also compiled successfully from a separately extracted copy of the source ZIP, without repository result dependencies.
- Compiled the 12-page manuscript and 14-page supplement; checked rendered pages and all regenerated figures. No undefined references or overfull boxes remain. Minor underfull-box warnings reflect normal line/page spacing.
- Fixed the bibliography packager to add commas before appended DOI/URL fields, preserving the existing reviewed bibliographic corrections.
- Preserved the pre-edit submission in `Archive/paper_before_corrected_results_20260910/`.
- PDFs are delivered under `output/pdf/`; the standalone source ZIP is under `output/overleaf/`. The package includes selected plotted event data and current audit evidence.
