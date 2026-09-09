**Final submission implementation plan — data-centre flexibility paper**

Prepared by Jarvis, 8 September 2026. This is an editorial implementation plan, not a revised manuscript. It follows the September supervision notes and uses the completed numerical evidence. No optimisation or simulation was run to prepare it.

**1. Recommended approach and scope**

Construct a new final manuscript using the original six-section structure, selected reviewer-responsive prose from `first_edit.tex`, and the current annual model and results loaded by `main_new.tex`. Do not concatenate entire versions or use a single cut-off line between old and new text: useful earlier amendments extend into the methodology, while some early contribution statements also need rewriting.

The governing story should be:

> An integrated model characterises the magnitude, duration and internal sources of data-centre flexibility. A year of economic operation provides a credible operating reference and quantifies electricity-cost savings. Existing UPS-capacity sensitivities then support a limited assessment of the additional investment that those savings could support.

The paper's intellectual priority is flexibility characterisation; annual rolling optimisation is the means of establishing operation and economic value. Retain the newer physical and temporal formulation even where its explanation moves to supplementary material.

There is sufficient existing evidence to proceed without commissioning new optimisation results. The remaining work is source reconciliation, restructuring, concise mathematical exposition, reuse or reformatting of existing figures, a small optional financial calculation, and publication checks. This does not mean that every claim in the current drafts is ready to publish unchanged.

**2. Source hierarchy and exact manuscript dependencies**

| Source | Role in the final construction | What not to inherit automatically |
|---|---|---|
| [September revision notes](/C:/GitHub/Data_Centre/docs/data_centre_flexibility_paper_revision_notes.md) | Governing editorial direction: integrated flexibility first, annual valuation second, selective supplement, UPS investment screen | Suggestions to run capacity cases are already satisfied by the completed results |
| [Original submission](/C:/GitHub/Data_Centre/paper/original_submission.tex) | Six-section architecture, motivation, asset-based explanation and emphasis on flexibility magnitude/duration | Original numerical results, single-day equations, baseline definition, old component interpretations |
| [First edit](/C:/GitHub/Data_Centre/paper/first_edit.tex) | Preferred donor for useful introductory, literature, modelling-rationale and reviewer-clarification prose | Daily sensitivity study, price-volatility conclusions, 6.25%/6.3% savings, old event results, all blue text as an undifferentiated set |
| [Current entry point](/C:/GitHub/Data_Centre/paper/main_new.tex) | Current references and front matter; identifies the active body | Annual-optimisation-led abstract, highlights and contributions |
| [Current body](/C:/GitHub/Data_Centre/paper/balanced/annual_revision_body.tex) | Current case design, event method, results, limitations, implementation appendix and back matter | Repetition, audit-led narrative and claims stronger than the evidence |
| [Expanded method](/C:/GitHub/Data_Centre/paper/balanced/methodology_expanded.tex) | Current mathematical formulation and retained workload explanations | Its current length, historical revision commentary and redundant equation/prose pairs |
| [Reviewer responses](/C:/GitHub/Data_Centre/paper/revisions/reviewers_comments_and_colleague_response.md) and [22 pinned colleague comments](/C:/GitHub/Data_Centre/paper/revisions/paper_with_colleagues_comments.md) | Coverage checklist for substantive and presentational concerns | Colour flags as final decisions; speculative suggestions as completed work |
| [July methodology plan](/C:/GitHub/Data_Centre/docs/paper_methodology_amendment_plan.md) | Technical checklist explaining why the newer formulation supersedes the original | Annual-method novelty claims, Scenario 3 deferral, sampled-week sensitivity as the current design |
| [July revision plan](/C:/GitHub/Data_Centre/docs/paper_revision_plan.md) | Historical rationale for annual coverage and continuity | Old action dates, proposed experiments, renewed solver work |
| [Frozen assumptions](/C:/GitHub/Data_Centre/docs/model_assumption_freeze.md) | Baseline, thermal and storage assumptions, price accounting and terminal policy | Its outdated claim that intermediate sensitivity points remain sampled weeks |

The live include chain is `main_new.tex` → `balanced/annual_revision_body.tex` → `balanced/methodology_expanded.tex`; the body also includes two generated tables. `balanced/main_balanced.tex` is a wrapper that sets author/header macros and inputs `main_new.tex`. Therefore the apparent size of `main_new.tex` understates the current manuscript's content.

Existing Overleaf exports and PDFs are presentation snapshots, not independent scientific authorities. The old and new TeX files also reference identically named `images/Figure_*.png` assets. The current images cannot be assumed to be the images originally associated with the old prose. In particular, the current Figure 7 has four panels, whereas the original discusses six. Preserve the present sources and use a separate final directory so future figure changes do not silently alter archival drafts.

**3. Numerical evidence to freeze before writing**

Use [annual_endpoints.csv](/C:/GitHub/Data_Centre/reports/final_annual_results/annual_endpoints.csv), [annual_cost_components.csv](/C:/GitHub/Data_Centre/reports/final_annual_results/annual_cost_components.csv), and the corresponding scenario summaries under `static/data/rolling_year_outputs/` as the annual numerical sources. The generator [generate_annual_results.py](/C:/GitHub/Data_Centre/paper/generate_annual_results.py) identifies the comparator as `2025_baseline_reformulated` and central operation as `2025_optimised_cohort_trace`. It reads completed results; there is no need to run it merely to recover numbers already tabulated.

| Quantity | Current value to carry forward | Interpretation |
|---|---:|---|
| Annual benchmark cost | £550,503.59 | Workload at arrival, storage dispatch disabled, cooling cost optimised |
| Annual coordinated cost | £522,702.95 | Central IT/UPS/cooling/TES operation |
| Annual saving | £27,800.64; 5.050% | Signed reference-price electricity cost; not total site operating cost |
| Annual grid electricity | 6.718 → 6.794 GWh | Energy increases by approximately 1.13% |
| Peak import | 974.9 → 1,681.9 kW | Price optimisation does not imply peak reduction |
| Workload sensitivity, 0.5–1.5× | 3.522–6.257% savings | Fixed total arriving workload; flexible share changes |
| UPS energy sensitivity, 0.5–1.5× | 4.565–5.501% savings | Energy capacity changes with fixed power ratings |
| TES energy sensitivity, 0.5–1.5× | 4.800–5.167% savings | Thermal energy capacity changes with fixed power ratings |
| Representative event day | 22 September 2025 | Selected from annual operation |
| Reduced-import example | 100 and 150 kW for 6 h, starting at noon | Feasible under the stated event and recovery rules |
| Increased-import example | 25 kW reaches the 12 h assessment cap | A tested cap, not proof of an intrinsic maximum |
| Event grid | 288 cells; 275 reported adjacent boundaries; 13 lower bounds | Preserve unresolved-status distinctions |

There are 14 annual result rows: one benchmark and 13 coordinated settings, with the central setting shared by the three five-point sweeps. Sampled-week intermediate values are no longer needed in the main paper. All 14 underlying scenario summaries were checked: each contains 365 local dates and 35,040 committed intervals, and every settlement cost agrees with the current annual table to within £0.000001.

Use [event summary](/C:/GitHub/Data_Centre/reports/representative_day_flexibility/summary.json) and [event results](/C:/GitHub/Data_Centre/reports/representative_day_flexibility/flexibility_results.csv) for flexibility claims. Retain [intermediate quality reruns](/C:/GitHub/Data_Centre/reports/final_annual_results/intermediate_quality_reruns.csv) and [terminal assessment](/C:/GitHub/Data_Centre/reports/terminal_treatment/terminal_treatment_report.md) as supporting evidence, with explicit source-version attribution. The terminal report still quotes the earlier 5.049% headline; do not copy that into the final results.

Create a small provenance register during implementation: claim, value/unit, source file, scenario, denominator/reference, proposed location and qualification. Freeze unrounded CSV values for calculations; round only for presentation.

**4. Two analytical questions, with the computational dependency preserved**

The September notes ask for flexibility characterisation followed by economic valuation. That is a narrative hierarchy, not a reason to reverse the actual computation. The event results depend on a previously computed annual operating trajectory.

Use three clearly named cases, retaining the original scenario mapping only where useful:

| Case | Legacy label | Definition and purpose |
|---|---|---|
| Reference operation | Scenario 1 | Annual workload-at-arrival, UPS/TES-disabled comparison with cost-optimised cooling |
| Coordinated operation | Scenario 2 | Same exogenous inputs with workload shifting and storage enabled; provides annual savings and the event reference trajectory |
| Flexibility event assessment | Scenario 3 | Additional signed import deviations relative to coordinated operation, with inherited state and fixed recovery allowance |

Do not use the same symbol or unqualified word “baseline” for both the annual comparator and the event reference. Prefer `ref` for the annual comparator and `op` for the coordinated trajectory.

The workflow should show inputs → integrated model → reference/coordinated annual operation, followed by two outputs: annual cost comparison and representative-state event assessment → duration envelope/component responses. Explain once that computation establishes the operating reference before testing events, even though results foreground the flexibility question.

Annual cost savings do not monetise the event heatmap. No service dispatch/payment simulation links the event durations to revenues. These are complementary evaluations of the same model, not two quantities that can be added to form a financial total.

**5. Section-by-section final manuscript blueprint**

Keep the original top-level sequence: Introduction; Literature Review; Methodology; Case Study; Results and Discussion; Conclusion. The subsection order and space allocation should change to serve the September direction. Suggested lengths below are editorial working budgets, not journal requirements: approximately 6,300–7,650 words including abstract, excluding references and supplementary material, with an initial target near 7,000.

| Element | Working budget | Construction instruction |
|---|---:|---|
| Abstract | 200–250 words | Rewrite after results; physical contribution and findings first, annual value second |
| 1. Introduction, architecture and workload background | 900–1,100 | Select from first edit; remove repeated contribution lists and soften universal workload claims |
| 2. Literature Review | 650–850 | Preserve asset-to-integration progression; make gap specific and evidenced |
| 3. Methodology | 1,600–1,900 | Explain model, couplings, event/recovery definition and concise annual procedure |
| 4. Case Study | 650–850 | Inputs, three cases, representative day and annual sensitivity design |
| 5. Results and Discussion | 2,000–2,350 | Flexibility envelope and mechanisms first; annual value, sensitivities and limitations next |
| 6. Conclusion | 250–350 | Re-state integrated flexibility findings; annual savings and bounded design implication |

**Front matter.** Retain the existing title initially: “Characterisation and Quantification of Data Centre Flexibility for Power System Support.” It already expresses the intended subject. Rewrite the abstract in this order: problem/gap; integrated modelling approach; magnitude–duration and component findings; annual valuation; qualified implication. Mention rolling optimisation in one supporting clause or sentence, not as the opening contribution. Remove software audits, 288-cell bookkeeping, solver gaps and cohort-ledger implementation from the abstract.

Candidate contribution wording:

1. An integrated representation of IT scheduling, UPS operation and cooling/TES that preserves the couplings governing facility-level flexibility.
2. A state-dependent magnitude–duration assessment with post-event recovery and component decomposition, showing how the internal assets deliver the requested response.
3. A full-year assessment of electricity-cost savings and resource-sizing effects, including a limited screen of the additional investment supported by larger UPS capacity.

The third contribution is an application/evidence contribution. Do not claim a new rolling-horizon algorithm or a new battery model. Remove the investment clause if that screen is ultimately omitted.

Candidate highlights should cover integration, different asset contributions, a quantitative reduced-import duration, the capped increased-import example, and annual savings. Avoid a standalone rolling-horizon highlight. Use 5.05% in prose if preferred; retain 5.050% in a comparison table where three decimals support the sensitivity results.

**1. Introduction.** Use the first edit's more direct data-centre motivation, flexibility definitions, architecture discussion and explanation that an IT change alters both electrical demand and heat. Retain the architecture and workload-background subsections in shortened form to remain close to the original structure. Put one contribution list after the gap; delete the current duplication between introduction paragraphs and the end of the literature review.

Keep the distinction between reduced import/upward flexibility and increased import/downward flexibility explicit. Describe flexible workloads as jobs eligible for rescheduling under assumed deadlines and SLAs, not a fixed universal category. “No service-quality sacrifice” should become “computational work and the modelled deadline constraints are preserved”; the aggregate model does not validate every application-level SLA.

**2. Literature Review.** Retain the improved first-edit opening and asset-by-asset review followed by integrated work. Tighten descriptions of individual studies to the point relevant to the gap. The key comparison is whether studies jointly represent workload timing, electrical storage, cooling/TES, event duration and recovery. A compact literature comparison table is optional if it replaces substantial prose; do not add it on top of the full review.

Qualify “largely unexplored,” “most studies” and “largely absent” against the cited integrated studies. Do not replace an uncertain novelty claim with a different unsupported novelty claim. Verify material source claims and citation keys during drafting, particularly the unrestricted battery-lifetime/no-additional-cost statement. Preserve citations that explain aggregate modelling, while distinguishing published validation of a source model from validation of this modified facility model.

**3. Methodology.** Suggested subsections:

1. Model purpose, system boundary and common inputs/controls/states.
2. IT workload, conservation, deadlines and utilisation-to-power relation.
3. UPS and cooling/TES flexibility, with the key coupling constraints.
4. Definition of the grid-deviation event, duration, recovery and component response.
5. Economic operation and annual evaluation, with a short implementation paragraph.

The first-edit modelling rationale and CPU-hours explanation are worth retaining even though they lie inside methodology. Rewrite their surrounding equations using the current cohort and boundary-state formulation. Remove historical phrases such as “unlike the original formulation,” “the smaller revision” and “the revised implementation”; the final paper must stand alone.

Limit the rolling subsection to roughly 250–350 words: 15-minute resolution; local-day core plus 3 h look-ahead; carry UPS/TES energy, four thermal states and unfinished cohorts; commit each interval once; signed 2025 price cost; actual 2026 look-ahead; reference to full algorithm and solver policies in the supplement. Mention retrospective known-input operation, without suggesting forecast accuracy or a globally solved 365-day perfect-foresight optimum.

**4. Case Study.** Replace the original synthetic-price table and all isolated-day setup. Put the stylised 1 MW IT-rated facility, repeated workload profile, fixed 22°C ambient temperature, COP 5, 600 kWh UPS, 1,000 kWh-th TES and 53.095 kW auxiliary load in a concise central-parameter table. Clarify that 1 MW is rated IT power, not a 1 MW site connection limit: the model's structural import ceiling is 1,723.095 kW.

Explain the 2025 IMRP input, hourly-to-quarter-hour replication, retained negative prices and wholesale-reference scope. Keep the 0.5/1/2/3 h workload delay classes and common 18–23°C cold-aisle range visible. Place full workload arrays, source derivation and detailed thermal parameters in the supplement.

Use the three-case table from Section 4 above to replace lengthy repeated scenario descriptions. Define the representative-day selection briefly and provide the full feature list, eligible-day rule and score in the supplement. Define annual one-at-a-time sensitivities at 0.5, 0.75, 1, 1.25 and 1.5×, with fixed storage power ratings and workload conservation. TES capacity is the completed cooling-side sensitivity; do not describe it as a chiller-sizing experiment.

**5. Results and Discussion.** Recommended order:

1. **Operating reference for the flexibility assessment:** a short paragraph identifying the selected day and inherited operating state, with a compact day-profile figure if needed. This establishes the reference without opening with a long annual audit.
2. **Magnitude, duration and direction:** lead with the current heatmap; discuss time dependence, asymmetric tested request ranges, 6 h reduced-import examples and the 12 h increased-import assessment cap.
3. **How components deliver the response:** use the current component plots to explain mechanisms, opposing actions and changes with start time. Include the 06:00 UPS-only versus 15:00 IT/cooling comparison from the current results. It directly demonstrates why component participation depends on operating state.
4. **Annual electricity-cost value:** report £27,800.64/5.050%, annual energy and peak import, then the compact cost-composition table. Explain coordinated accounting without claiming standalone asset profits.
5. **What changes with resource assumptions:** give the UPS capacity question the main discussion; provide a concise workload/TES comparison and all annual points in the supplement. Close with the battery investment screen if included.
6. **Practical interpretation and limitations:** consolidate the current repeated limitations into a few focused paragraphs.

Do not retain old prose saying that downward flexibility relies mainly on cooling/UPS because IT work cannot be advanced. The current cohort-based state may contain eligible unfinished work, which can be executed earlier than in the coordinated reference, although never before arrival. Also do not reuse the old IT-first/UPS-later sequencing explanation as a universal description: the new selected reduced-import cases show different patterns.

Keep UPS prominent because its size is a design decision, while stating honestly that workload share has the largest economic response within the tested ranges. “Most interesting design sensitivity” and “largest numerical sensitivity” are different claims. Equal percentage multipliers are not equal capital investments, so the ranking is not an investment-efficiency ranking.

**6. Conclusion.** Lead with state-dependent flexibility magnitude/duration and the roles of coupled resources. Follow with the annual savings result and its energy-price scope. End with the UPS threshold implication if supported. Retain a short future-work sentence on wider operating conditions, degradation and market participation; do not turn a full factorial study into an essential next step for this paper. Remove “prosumer” where it implies export, and avoid market-readiness, frequency-response qualification or network-stability claims unsupported by a 15-minute model.

**6. Equation allocation: main paper versus supplement**

The current expanded method has roughly 40 labelled mathematical statements before the event equations. Keeping all of them as large equation floats would obscure the contribution. Recommend approximately 10–14 compact main-text equation groups; this is a layout target, not a quota. The supplement should contain the complete implemented formulation in one coherent notation, including any strategically duplicated defining equations.

| Mathematical material | Main paper | Supplement / action |
|---|---|---|
| Facility grid balance, IT–heat coupling and shared chiller capacity | Retain: these explain integration and prevent double counting | Repeat definitions where needed for a standalone full model |
| CPU-hours conservation, release/deadline window and total utilisation limit | Retain one compact group | Full horizon-specific cohort completion, residual-work and arrival bookkeeping |
| IT utilisation-to-power law | Retain with an explicit piecewise approximation statement | DLOG formulation, breakpoints and 6.241 kW maximum error evidence; do not call the nonlinear equation itself the MILP equality |
| UPS and TES energy recursions | Retain a compact pair if space permits; explain reserve/power constraints alongside | Complete bounds, one-mode binaries, efficiencies, IT-only discharge restriction and initialisation |
| Cooling conversion and restrictive heat-removal rule | Retain the substantive relation and disclose the operating rule | Full gross/effective cooling definitions and all temperature constraints |
| Four thermal-node recursions and inlet relation | Explain states, coupling and implicit discretisation; no need for all five equations | Retain all implemented backward-Euler equations and parameters |
| Grid target and duration definition | Retain; this is central to what flexibility means | Full time-index sets, cap, search and solver classification |
| Recovery of UPS, TES, temperatures and cohort work | Retain grouped conditions or compact vector form with explicit inequalities | Exact tolerances and implementation indexing |
| Component deviation identity | Retain | Additional dispatch interpretation and complete event panel set |
| Signed-price objective and annual saving definition | Retain, including kW-to-MW conversion where prices are GBP/MWh | Core-only settlement sum, overlapping horizons and terminal details |
| Rolling state and cohort handoff | Explain in main prose; one symbolic relation only if it helps | Complete algorithm, UTC/local-day handling and year-end policy |
| Elementary energy = power × time and duplicated baseline/optimised formulas | Usually remove from main displayed equations | Define once where needed; avoid mechanically preserving redundant displays |

Do not relocate outdated equations into the supplement. Specifically discard the old extension-only incremental IT-power expression, daily cyclic storage equalities, explicit-Euler thermal updates, minimum storage powers and two-binary formulation where they differ from the active model.

The main text must still disclose assumptions that materially determine the answer: 50% UPS reserve, fixed storage power limits, IT-only UPS discharge/no export, common temperature bounds, conservative cooling lower bound, workload deadlines and post-event recovery. These should not become discoverable only in an appendix.

Recovery is not exact equality of every state: current rules require UPS/TES energy no lower than the reference, temperatures within 0.05°C and cohort work no greater than the reference plus tolerance. Say “recovered relative to the reference under the stated criteria,” not “all states return exactly.”

**7. Supplement structure**

Prepare a separately compiled supplementary document, subject to the eventual journal's permitted format. If it must be an appendix instead, preserve the same content division. Use S-numbered equations, figures and tables and explicit main-text cross-references.

| Part | Contents | Required reproducibility outcome |
|---|---|---|
| S1. Notation, model scope and parameters | Complete symbol table, units, central settings, equipment bounds and initial conditions | No reliance on the disabled/obsolete nomenclature in `main_new.tex` |
| S2. Workload and price inputs | Literature-source construction, actual hourly workload values, tranche shares, clipping rule, input files, IMRP provenance and time conversion | Reader can reconstruct the inputs actually used |
| S3. Complete integrated formulation | Cohorts, IT approximation, UPS, chiller/TES, all thermal recursions, grid balance and domains | Every material active constraint is documented |
| S4. Annual procedure | Pseudocode, 92/96/100-interval days, 12-interval look-ahead, commitment, state/cohort transfer, year-end accounting | Reader can reproduce the evaluation timeline |
| S5. Flexibility experiment | Day-selection details, exact opening state/cohort logic, request grid, recovery, search, timeout and cap classifications | Reader knows what each heatmap cell establishes |
| S6. Extended results | All 14 annual rows, full sensitivity curves, extended operation and component panels, numerical heatmap | Main figures can be traced to retained outputs |
| S7. Numerical verification | Software/solver record, approximation error, continuity/conservation checks, exceptions and existing terminal/rerun evidence | Feasibility, approximation and optimisation status are distinguished |
| S8. UPS investment screen, if used | Incremental savings arithmetic, CRF inputs, break-even thresholds and any sourced cost assumptions | Investment inference is transparent and separate from dispatch modelling |

Keep a brief verification statement in the main text. Move repeated solver troubleshooting, dates of individual timeouts, checkpoint hashes and tiny closure residuals to S7. Supporting numerical checks are not experimental validation of a real data centre.

**8. Figure and table implementation plan**

Use the current numerical figures, not archived daily results. Prefer six to seven main figures and three to four tables; cut duplication before reducing readability.

| Existing item | Recommended destination / treatment |
|---|---|
| Figure 1, DC layout | Main paper; retain or simplify labels consistently |
| Annual workflow TikZ figure | Replace in main with whole-study workflow; retain detailed daily loop in S4 |
| Figure 2, proposed workload profile | Main case-study inset or S2, depending on space; preserve clear workload provenance in main prose |
| Figure 3, constant CPU-hours illustration | S2 or omit if the short explanation is sufficient |
| Figure 4a/b, representative operation | One compact main figure establishing the event reference; shorten interval-by-interval narration |
| Figure 5, full component dispatch | Supplement unless needed to support a specific main claim not visible in event plots |
| Figure 6, current duration heatmap | Main paper, high priority and generous width |
| Figures 7 and 8, current event decomposition | Keep both if readable; otherwise select contrasting panels into one main figure and place complete sets in S6 |
| Sensitivity curves | One compact main figure with UPS emphasis and comparable workload/TES context; build only from existing CSV values if a new layout is needed |
| Full nomenclature, workload-rate and tranche tables | S1/S2; retain only central parameters and delay-class explanation in main |
| Central-parameter table | Main, reduced to flexibility-relevant settings; thermal constants move to S1 |
| Three-case control table | Main; replaces scenario prose and distinguishes the two reference meanings |
| Annual component-cost table | Main; addresses the explicit reviewer request for cost composition |
| Complete annual endpoint table | S6; retain central totals and UPS increments in main text or a small investment table |

The inspected current heatmap already separates signs, uses hourly start labels, darkens with increasing duration and marks lower bounds. Preserve those improvements. Its blank cells must be defined as “no positive duration certified” unless per-cell solver evidence warrants the stronger claim of infeasibility. A 12 h cap-reaching cell also needs to be distinguished from a proven physical maximum. Explain that late-day declines can reflect the end-of-day assessment limit as well as physical state.

The inspected current Figure 7 has informative panel labels and a shared component legend. The grey region identifies time after the event, but does not itself display a full recovery trace. Do not claim that recovery is visibly demonstrated there; the terminal checks provide that evidence. Shorten or qualify dashed requested-target lines extending through post-event space so they do not suggest the grid target remains imposed during recovery.

Keep common component colours, legible fonts at final print size, clear units, panel letters and explicit sign conventions. Any replotting should be presentation-only from stored outputs. Check scripts before execution because shared-image destinations can overwrite assets used by archival drafts.

**9. UPS investment calculation: include a bounded break-even screen**

Recommendation: include this as a short engineering interpretation of the existing sensitivity. Start with the maximum additional investment supported by the additional annual saving. This is useful even without a defensible vendor cost, and avoids inserting an arbitrary battery price to manufacture a positive business case.

The completed results give:

| UPS nameplate/model energy capacity | Added capacity versus 600 kWh | Annual coordinated cost | Additional saving versus central | Added operational energy window with 50% reserve |
|---|---:|---:|---:|---:|
| 600 kWh | — | £522,702.95 | — | — |
| 750 kWh | 150 kWh | £521,410.67 | £1,292.28/year | 75 kWh |
| 900 kWh | 300 kWh | £520,221.54 | £2,481.41/year | 150 kWh |

The corresponding additional annual savings are £8.62 and £8.27 per added nameplate kWh. The second 150 kWh increment, from 750 to 900 kWh, adds approximately £1,189.13/year. This is a modest taper in marginal value, not evidence of a dramatic sizing optimum.

Define additional capital investment `I_add`, real discount rate `r`, assessment life `n`, and additional annual non-electricity costs `O_add`. For constant real annual benefits:

\[
\Delta S(E)=C_{\mathrm{op}}(600)-C_{\mathrm{op}}(E),\qquad
\mathrm{CRF}(r,n)=\frac{r(1+r)^n}{(1+r)^n-1},
\]

\[
\Delta S(E)-O_{\mathrm{add}}-I_{\mathrm{add}}\,\mathrm{CRF}\geq0,
\qquad
I_{\mathrm{break-even}}=\frac{\Delta S(E)-O_{\mathrm{add}}}{\mathrm{CRF}}.
\]

For zero discount rate, CRF is `1/n`. The use of a capital recovery factor to annualise investment is standard; the [System Advisor Model financial-method documentation](https://samrepo.nlr.gov/help/fin_lcoefcr.html) describes this factor and the preliminary nature of simplified financial screening. This is an annualised-cost screen, not an LCOE calculation.

As an explicitly illustrative calculation only, at 8% real discount rate and 10 years, CRF ≈ 0.14903. With constant annual savings and no additional O&M, degradation, replacements or residual value, the supported additional investment is about £8,671 for the 750 kWh case and £16,650 for the 900 kWh case: approximately £57.81 and £55.50 per added nameplate kWh. These are break-even thresholds, not estimates of battery purchase costs. The 8% and 10-year assumptions are examples for assessing the proposal, not selected or literature-validated final inputs.

The operational model reserves half of nameplate capacity, so do not divide nameplate installation costs by the smaller usable increment without changing the unit label. The wording “600 kWh usable capacity” in the expanded method needs correction: the permitted 300–600 kWh range gives a 300 kWh operational swing.

For the final paper, either report a small threshold range over explicitly stated life/discount assumptions, or compare the threshold with a verifiable installed incremental UPS battery cost. Any cost source must identify chemistry, price year, currency, nameplate versus usable kWh, installation and integration scope, and whether existing power electronics can support expansion. A generic utility battery-pack price is not automatically an installed UPS extension cost.

The default final-paper footprint should be one short methods paragraph, one result paragraph and, if useful, a two-row threshold table. Put alternative rates/lives and cost-source detail in S8. A separate NPV analysis is unnecessary unless savings, replacement or residual values vary over time; under constant benefits it merely restates the same comparison.

Qualifications that must accompany the inference:

- Compare larger UPS cases with the 600 kWh coordinated case, not with the storage-disabled benchmark. The entire £30,282 saving in the 900 kWh scenario is not earned by the extra battery.
- The existing sensitivity scales both initial stored energy and the minimum reserve with capacity. It does not represent an expansion with a fixed absolute backup reserve. Keep this policy explicit.
- Larger initial/final stocks and zero terminal credit mean a single-year difference is not automatically a repeatable lifetime annuity. Use existing terminal/checkpoint evidence to discuss this; the central terminal sensitivity alone does not prove that every incremental-capacity effect is unchanged. Label the lifetime calculation a conditional screen.
- Battery efficiency is already included in dispatch costs; do not charge those losses again. Degradation and additional O&M are omitted and reduce supported investment if introduced.
- Repeating the 2025 saving over a project life is a scenario assumption, not a multi-year price forecast. Wholesale-reference savings are not retail-bill savings, and increased peak import remains relevant.
- Do not add hypothetical service revenues from the heatmap or value improved backup resilience without a separate basis.

If costs exceed the threshold, the useful conclusion is that energy arbitrage under these assumptions does not independently justify the enlargement. If credible costs are below it, say that enlargement may be supported under the stated assumptions, not that an investable UPS design has been established. If no suitable cost evidence is available, retain the threshold result and leave the actual investment decision open.

**10. Scientific and editorial corrections to make during the merge**

These are targeted writing/reconciliation tasks, not a request to reopen model development.

| Issue found | Required correction |
|---|---|
| Current annual results attribute the lower saving than the original day to annual price coverage | Remove the causal comparison: the original reported 10.02%, first edit 6.25%, and annual 5.050%, but baseline controls and model assumptions also changed. Full-year coverage is better temporal evidence, not a controlled explanation of that difference |
| First edit actually includes a mean-preserving price-volatility sensitivity | Do not import it. Use the completed annual workload/UPS/TES design. Annual price coverage addresses day selection but does not demonstrate robustness across other years or all tariff/volatility regimes |
| Frozen-assumption sensitivity paragraph conflicts with current results | Mark that paragraph superseded in the new provenance register; use all completed annual points |
| Workload prose says proposed values lie within observed ranges | The displayed table contains counterexamples: at 02–03 proposed inflexible utilisation is 17% versus reported minimum 25%; at 03–04 proposed total is 39% versus maximum 37%; at 17–18 proposed flexible is 40% versus maximum 39%. Explain these as adopted stylised assumptions, or resolve table provenance. Do not change the numerical inputs to make the prose true |
| Workload source table supplies references but not a fully reproducible derivation | Retain the sources and supply actual adopted arrays. Verify whether a source-to-hour derivation record exists; where absent, describe judgement-based profile construction rather than claiming a reproducible averaging algorithm or measured validation |
| CPU-hours equated to constant energy / guaranteed service quality | Preserve computational work; electrical energy can change under nonlinear utilisation-to-power and storage losses. Qualify application-level performance claims |
| Old integrated-value paragraph says ignoring cooling reduction overestimates grid reduction | Correct the sign logic. Omitting a real cooling reduction would omit an additional reduction, although independent asset assessments may also double-count constraints or headroom. Do not claim a quantified benefit over separate optimisations without those comparison runs |
| Current method writes the analytic IT power curve as if it were the implemented equality | Clearly distinguish analytic relation from the piecewise-linear representation used in both cases; put exact breakpoint construction in S3 |
| Cohort completion/handoff notation is informal about new arrivals and visible deadlines | Define horizon-restricted execution sums, remaining rather than original work, initialisation of new cohorts and interval/deadline convention consistently in S3. Cross-check relevant implementation by reading it, without running it |
| Cooling is described as universally physical or fully validated | Disclose the conservative gross-cooling lower bound and calibrated effective-airflow factor. Separate the adopted modelling rule from a universal thermodynamic law |
| “All solves optimal” / “all audits pass” conflates feasibility and objective bounds | Report accepted feasible operation and actual solver qualifications. Put existing exceptional gaps and targeted reruns in S7; do not silently replace full-chain outputs with diagnostic reruns |
| “Maximum/exact duration” can exceed what adjacent testing establishes | Preserve feasible durations, caps and unresolved lower bounds. An infeasible adjacent duration alone does not prove that all longer durations are infeasible unless the search's monotonicity premise is justified; recovery reference time moves with duration. Check existing search evidence and use “certified feasible duration” or “adjacent tested boundary” where a global maximum is not established |
| “Representative day” interpreted as annual flexibility availability | State that it is one objectively selected illustration conditional on the annual trajectory. No annual service-availability distribution has been computed |
| “Annual optimization” interpreted as annual global optimum or live forecasted control | Describe sequential local-day optimisation using the historical input series and accepted MILP solutions. Distinguish numerical optimality gaps from finite-look-ahead limitations |
| Monetary component decomposition interpreted as asset value | Label as signed settlement accounting within joint operation; TES benefit partly appears through CRAC changes. It is not an ablation study or causal allocation |
| Current raw nomenclature contains old values/definitions but is disabled by `\\iffalse` | Rebuild from active model definitions. Do not re-enable it wholesale |
| Author order and running author differ across versions | Preserve the latest intended author order provisionally and reconcile the final wrapper/header and CRediT statement with the authors; do not infer authorship from the original draft |

Also remove claims of effortless proportional scaling from 1 MW to 100 MW unless the precise proportional assumptions are stated. Per-MW presentation is useful; general linear scalability across facility designs is not established by the present study.

**11. Reviewer and colleague coverage**

Prepare an internal response matrix even if the next submission is to a different journal. A formal point-by-point response is only part of the submission package if the submission route calls for it.

| Concern | Final-paper response | Acceptance evidence |
|---|---|---|
| R2.1 and R5.1: novelty and aggregate-model rationale | Integration and duration/recovery contribution; conventional battery model; retained state variables and scope | Revised abstract/contribution list and method-opening paragraph |
| R2.2, R5.8; C018–C021: figures, signs and readability | Explicit import terminology, common styles, labelled panels, legible heatmap, hourly mesh and status markers | Visual check at publication size |
| R2.3: verbose scenarios and sensitivity | Compact case-control table; full-year price coverage; annual workload/UPS/TES results | Scenario definitions plus cited annual tables; no claim of unperformed price perturbations |
| R2.4: cost composition | Annual signed component and total cost comparison | Main table linked to unrounded CSV |
| R2.5 and R5.5: reproducibility and workflow | MILP, key equations, full-study workflow, complete supplementary formulation and solver record | Every plotted quantity and scenario reproducible from documented definitions |
| R2.6: grid implications | Report peak-import increase and modelled connection boundary; explain absence of network model | Main limitations and no unsupported stability/hosting claims |
| R2.7 and R5.3; C007–C010: workload construction | Source summary, adopted arrays, judgement assumptions, tranche definitions and aggregate CPU-hours rationale | Main case-study summary plus S2 data |
| R5.6: dispatch selection/non-uniqueness | Cost objective subject to event/recovery; accepted feasible incumbent can be non-unique | Event-method explanation and captions |
| R5.7; C004: value of integration | Specific current IT/cooling and UPS examples; compensating component actions | Main component discussion tied to selected panels |
| C001–C002 and C022: concrete contribution/results | Flexibility-led highlights, abstract and conclusion with current duration examples | No old 6.8 h/9.25 h claims |
| C003, C014–C015: auxiliary load | Constant 53.095 kW, historical calibration, identical treatment | Main parameter entry and S1 derivation; no time-varying decision variable |
| C011–C012 and C016: scale and temperature assumptions | 1 MW IT rating and common 18–23°C cold aisle; no unsolicited change to results | Consistent text/equations/parameters |
| C017: event versus recovery constraints | Grid target applies only during event; recovery conditions at event end plus 3 h | Correct time sets in main and S5 |

No UKPN dataset integration, new chiller-capacity sweep, forecast experiment, network simulation or factorial optimisation is necessary for this editorial revision. Mention such extensions only if they sharpen a relevant limitation.

**12. Implementation sequence and concrete deliverables**

Create a separate source package under `paper/final_submission/` during the writing phase. Suggested contents: `main.tex`, `sections/`, `supplement.tex`, `supplement/`, `figures/`, `tables/`, `references.bib` and `README.md`. Keep provenance and the reviewer matrix alongside the editorial notes. These are proposed outputs; this plan does not create or replace manuscript files.

| Phase | Work | Deliverable / exit condition |
|---|---|---|
| 1. Lock sources | Snapshot active include chain and selected result/figure sources; record scenario IDs and superseded documents | Claim/provenance register, six-section outline and asset manifest |
| 2. Build the supplement | Consolidate current equations, inputs and implementation; resolve notation and disclose material assumptions | Complete reproducibility document independent of obsolete equations |
| 3. Assemble results | Move current event findings ahead of annual economic discussion; trim repetitive descriptions; select figures | Every reported result traceable to the frozen outputs; no old results |
| 4. Write main method and case study | Extract core equations from S3; add whole-study workflow and compact case/parameter tables | Reader can understand physical meaning and experimental design without reading implementation detail |
| 5. Resolve UPS screen | Use existing incremental savings; choose threshold-only or sourced-cost comparison; state financial assumptions | Short, conditional engineering implication without new optimisation |
| 6. Merge earlier prose | Retain appropriate first-edit introduction/literature/rationale; rewrite gap, contributions and conclusion | Consistent flexibility-led argument throughout |
| 7. Write front matter | Finalise abstract, highlights, keywords and title fit from completed narrative | Quantitative statements agree with results and qualifications |
| 8. Verify scientific consistency | Check signs, units, references, accounting, case names, caps/lower bounds and source-version exceptions | Reviewer matrix and claim register completed; unresolved points explicitly limited |
| 9. Produce submission package | Compile main and supplement, inspect all pages, verify references/assets and journal-required declarations | Clean PDF/source package and optional marked comparison ready for author review |

Budget by editorial passes rather than solver runtime: this is likely several focused writing/review sessions, with the supplement/notation reconciliation and final visual layout being the largest uncertainties. The first checkpoint should be the frozen source register, outline and figure allocation, before drafting long replacement prose. The next checkpoint should be the complete results and supplement, so coauthors can review the scientific content before front-matter polish.

Read source and output files where needed. No solver commands belong in this sequence. Document compilation and, if necessary, presentation-only plotting are distinct from rerunning the scientific model. Inspect any existing build script before using it, and ensure it does not trigger result generation or overwrite archival assets unexpectedly.

The destination journal/submission route is not specified. Do not assume the rejection permits a formal revision to Applied Energy, and do not switch templates or invent journal limits now. At packaging, verify the selected journal's current requirements, including supplementary format, highlights, declarations, data/code availability and source files. The present IEEEtran class is not evidence of the next journal's required format.

**13. Definition of a submission-ready outcome**

- The title, abstract, highlights, contribution list, result order and conclusion all lead with integrated flexibility characterisation.
- Annual rolling optimisation is explained accurately and concisely, without a novelty claim or an unsupported global-optimality claim.
- Reference operation, coordinated operation and event reference are unambiguous.
- Every numerical headline comes from the selected current result set; old 10.02%, 6.25%, 6.8 h and daily sensitivity conclusions have been removed.
- The full-year scope is not overstated: annual price coverage, repeated workload, constant ambient temperature/COP and one event day are explicit.
- Main-text equations explain the contribution; the supplement contains the full implemented formulation and actual inputs.
- Workload conservation, electrical energy, reserve capacity, recovery and solver certification are described precisely.
- Figures agree with their captions and accompanying interpretations, remain legible at final size, and distinguish caps/unresolved results.
- The UPS screen uses incremental benefit, correct capacity units and explicit assumptions; it reaches whichever conclusion the evidence supports.
- The cost-composition table and workload-provenance material remain available to address the reviewers rather than disappearing during condensation.
- Main and supplementary PDFs compile with no unresolved citations/references, and all pages are visually checked for equation overflow, float order, table size and symbol consistency.
- A clean source package, final PDFs, evidence register and internal reviewer-coverage matrix are ready for coauthor review. No new optimisation results are required to complete this package under the recommended scope.
