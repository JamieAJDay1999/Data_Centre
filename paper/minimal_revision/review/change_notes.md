# Reasons for the minimal revision

The controlling source is `paper/first_edit.tex`. The September supervisor notes supply the contribution hierarchy; the retained annual and event outputs supply the methods and numbers. The earlier rewritten paper was used as a source of already-checked numerical details, not as the prose template.

| Location | Necessary change | Reason |
|---|---|---|
| Abstract and highlights | Replace daily savings, old event durations and price-sensitivity findings; qualify duration certification | Those results belong to the earlier experiment. Background wording and abstract structure remain. |
| Introduction | Reorder the contribution list; add a short explanation of annual valuation; qualify maximum-duration wording | Put physical characterisation first without shortening the introduction. |
| Literature review | Resolve four inherited missing citation keys; change the closing maximum-duration claim | Preserve the review while ensuring citations resolve and the claim matches the search. |
| Nomenclature | Correct time sets, active power/temperature bounds, state indexing and event sampling | Existing rows remain, updated to the retained model. |
| Workload inputs | Retain historical range columns; update proposed hourly means and tranche means; distinguish them from actual quarter-hour arrays | The old hourly summaries did not specify the inputs used by the annual model. Figure 2 is regenerated directly from stored arrivals. |
| IT formulation | Replace extension-only incremental power with full-site power; include carried and look-ahead arrivals; retain workload/power/capacity equations with original labels | The rolling model must not subtract baseline power in look-ahead or count advisory processing as completed work. |
| UPS and TES | Replace daily cyclic equalities with state handoff; correct active minima, reserve and initial energy | The retained annual chain does not reset storage daily. |
| Cooling | Update state equations to backward Euler, the effective-cooling factor in the aggregate balance, common bounds and the gross-cooling lower limit | These are mathematical consistency corrections to the model used for the reported results, not a new model run. |
| Scenario 1 | Describe cost-optimised cooling with work at arrival and storage disabled | The retained benchmark is not a fixed-temperature, wholly unoptimised facility. |
| Scenario 2 | Replace synthetic daily prices and SCIP/SOS2 descriptions; add linked annual operation and committed-only settlement | Matches actual IMRP inputs, DLOG linearisation and HiGHS run evidence. |
| Scenario 3 | Keep exact arrival/tranche histories; update recovery, selected day, hourly requests, caps and search status | Retranching work or treating unsuccessful trials as proof of infeasibility would misdescribe the retained assessment. |
| Results | Replace outdated quantities and mechanisms; keep operating profiles first, physical responses next, annual cost evidence afterwards | Retains scenario identities while applying the agreed physical-then-economic hierarchy. |
| Sensitivities | Use completed annual workload/UPS/TES curves; remove old daily price-volatility findings | Those are different experiments and must not be combined. |
| UPS investment | Add a compact incremental break-even calculation | Uses existing annual savings and answers the supervisor's sizing question without inventing installed battery prices. |
| Integration discussion and conclusion | Correct mechanisms, obsolete numbers, reversed attribution and unsupported export/market/scaling claims | Necessary factual alignment, with the surrounding discussion retained. |

## What remains in the main paper

All original component topics and their equation labels remain. The existing nomenclature, literature-source summary, workload-range table, tranche table, architecture diagram, computational-demand explanation and component-dispatch figure remain. Updated evidence replaces obsolete results in place. The only substantial movement is of the annual cost discussion/table within Results. No existing main-paper component material was transferred into the supplement.

The supplement contains additional quarter-hour records, rolling/calendar details, representative-day and event procedures, complete annual sensitivity values, solver/terminal diagnostics and investment assumptions. Its six sections match the main-text pointers.

## Scientific points worth reviewing

1. **Workload provenance.** The earlier minimum/maximum/average columns are retained as requested, but the repository does not establish a reproducible source-by-source derivation of them. The manuscript now distinguishes these contextual summaries from adopted inputs. They should not be described as a validated statistical sample without further provenance from the authors.
2. **Deferral convention.** The implemented final execution interval starts at arrival plus the stated delay and may finish 15 minutes later. This is disclosed rather than silently changing the model or claiming a stricter completion deadline.
3. **Duration certification.** Positive reported durations have feasible audited dispatches. Twelve-hour/local-day caps and thirteen unresolved next steps prevent interpreting every cell as an intrinsic maximum. A moving recovery reference also means an adjacent infeasible duration is not a global maximality proof.
4. **UPS economics.** The 750/900 kWh cases add approximately GBP 1,292/2,481 per year relative to the 600 kWh coordinated case. The investment thresholds assume repeated benefits; reserve scaling and unequal initial/terminal energy prevent interpreting the single-year differences as independently demonstrated lifetime returns. No new solver run is required for the conditional interpretation used here.
5. **Literature wording.** Existing literature summaries are intentionally retained, rather than subjected to another broad rewrite. Resolving citation keys and carrying forward previously checked bibliography corrections is not a new full audit of every historical literature claim.

## Verification and provenance

Annual values come from `reports/final_annual_results/annual_endpoints.csv` and `annual_cost_components.csv`, checked against all fourteen raw annual summaries. Event values come from `reports/representative_day_flexibility/flexibility_results.csv`; retained component figures are `paper/images/Figure_4a.png` through `Figure_9.png`. Input CSVs originate in `static/data/inputs/`. The complete first-edit source, a source diff, and replacement reasons are included for review. `verification.json` records numerical checks and PDF compilation status.

No simulation or optimisation was executed. Changes to diagram captions, input summaries and the input plot use existing data. Formatting has been kept to a short compilation/readability check; the long first-edit main-text material is deliberately retained for the authors' later editing decisions.
