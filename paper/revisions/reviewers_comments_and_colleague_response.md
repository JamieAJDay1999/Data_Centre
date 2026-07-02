# Response to Reviewer Comments

> **Flag legend** — colour flags carried over from the original Word document, indicating whether my colleague thinks the reviewer comment warrants a change to the paper:
> - 🟢 **GREEN** = warrants a change (accept / act on it)
> - 🔴 **RED** = does not warrant a change (rebuttal / disagree)
> - ⚪ (no flag) = neutral author note (uncoloured in original)

---

## Reviewer #2


> 1. The UPS model established in this paper lacks significant differentiated innovation compared with traditional "data centre + energy storage" models. Existing studies have extensively covered topics related to the coordinated optimization of data centres and energy storage, and this model fails to reflect breakthrough technical contributions, making it difficult to highlight the innovative value of the research.

🔴 We appreciate the reviewer's comment, but we believe this criticism may misunderstand the paper's primary contribution. The UPS model intentionally utilizes a standard battery ESS formulation as the paper's novelty does not lie in UPS modelling. Instead, our contribution clearly resides in the flexibility characterization framework rather than re-deriving battery physics. Therefore, we believe no further modifications are required on this point.


> 2. There are obvious confusions in the paper's language expression and figure presentation: first, the logic of describing figures is unclear. For example, it is mentioned that "Figure 7 shows six stacked bar charts arranged into a 3-by-2 grid...

🟢 While the subfigures can be understood with a bit of time and effort, it would be much clearer for the reader if they were labelled as 'a', 'b', 'c', etc. Additionally, including a brief explanation of what each label represents in the heading would greatly improve readability.


> Figure 7 shows negative ∆P (upward flexibility) and Figure 8 shows positive ∆P (downward flexibility)", which is likely to cause confusion for readers. Readers do not know whether this sentence refers to Figure 7 or Figure 8; second, the font size in the figures is too small, inconsistent with the font size in the main text, affecting readability; third, the use of abbreviations is not unified, with "DC" used in some places and "data centre" in others; fourth, the two subfigures in Fig.4 have significant style differences and are not suitable for presentation as subfigures of the same figure.

🟢 Regarding the confusion over upward and downward flexibility, I am not sure if we need a separate subtitle to explain the results. However, we should at least split the paragraphs to wrap up the discussion for Figure 7, and then start a new paragraph with: 'Figure 8 presents the downward flexibility (positive ∆P ) results.' This way, everything related to Figure 7 is explained in one paragraph, followed by a separate paragraph for Figure 8.

🟢 Additionally, I wasn't aware that figures with different styles couldn't be used as subfigures. The reason we combined them is to show the relationship between power prices and IT workload distribution, so I think we should keep both graphs together unless there is a better visualization option. It’s worth to check this with Meysam. That being said, we could definitely improve the colour choices for this figures and others as well.


> 3. The case study section is verbose, with an excessive proportion of scenario descriptions, while only the operational results of three scenarios are actually compared, making it difficult to fully reflect the contributions of this research. It is recommended to supplement multi-dimensional case analyses, such as the model response comparison under different electricity price fluctuation ranges, IT load flexibility ratios, and equipment parameters to enhance the universality and persuasiveness of the research.

🟢 We can adopt the reviewer's first suggestion to improve the readability and clarity of that section.

🔴 However, the second suggestion is unnecessary. The content is already dense with a lot of information to digest, so adding more analysis would only increase the complexity. Furthermore, introducing such multi-dimensional case studies would expand the scope substantially without strengthening the core methodological contribution.


> 4. The paper only mentions a "10% cost reduction" in the conclusion without providing specific cost composition and comparative data to support this claim. It is suggested to add a cost comparison table in the case study section, clearly listing the sub-item costs (e.g., IT load, cooling system, energy storage operation) and total costs of each scenario, so as to improve the credibility of the conclusion.

🟢 This could be done, but I am not sure if a highly detailed cost breakdown table is necessary. Instead, we could include a sentence that mentions the individual contributions of IT costs, cooling costs, and other factors to the total cost. This would still allow readers to easily assess which areas offer the most room for cost reduction. Like x amount/ratio of cost reduction coming from IT/Cooling etc.


> 5. The core information of the optimization problem is not clearly elaborated. For example, the problem type, the adopted solution algorithm are not specified, resulting in insufficient reproducibility of the research method. It is recommended to supplement relevant technical details.

🟢 The paper already states that the model is implemented in Python with Pyomo and solved using SCIP, and it mentions the SOS2 linearization, so this is partially addressed. However, the reviewers are correct that the problem class is never explicitly stated. We should add one sentence at the start of Section IV-B explicitly stating: 'The resulting optimization problem is a Mixed-Integer Linear Program (MILP), as it includes binary variables for UPS charging/discharging status and TES operation, alongside continuous decision variables.'

🟢 Additionally, since we previously discussed adding a flowchart to illustrate the optimization approach and problem statement, I think it would be beneficial to include it now. I think the reviewer 5  have the similar comment on this in comment 5.


> 6. According to the optimization strategy proposed in this paper, the data centre load will show a significant peak-valley difference (as shown in Fig.4). However, the paper fails to fully demonstrate the feasibility and adaptability of the data centre model connecting to the power system under this load characteristic, and lacks analysis on the impact on power grid stability and corresponding solutions.

🔴 This comment requests a grid stability analysis that is entirely outside the scope of this paper. Our focus is strictly on characterizing and quantifying flexibility from the data center's perspective, while treating the grid simply as a power source.


> 7. Fig.2 presents the IT load ratio data used in this study, but does not specify the data source in detail. The reliability and representativeness of the data are not supported. It is recommended to supplement a clear source description and data verification basis.

🟢 Actually the methodology and the resource was stated in the paper but both reviewers mentioned the same so I will take a look and make it clear how we derived the workload profile.

🟢 Also, when we initially wrote this paper, there was very little data available. However, we now have access to the UKPN dataset, which provides valuable insights into power consumption ratios, even though it does not explicitly show actual workload utilization. We could potentially integrate this dataset with our existing workload characteristics. This is totally depending on if Jamie have time to do this extra work? And also if it is worth it?


## Reviewer #5


> While co-scheduling IT workloads, UPS batteries, and cooling infrastructure provides an insightful integrated approach, the current presentation fails to clearly distinguish the proposed contribution from existing studies on the coupled operation of data center subsystems. The framework appears more like a combination of three simplified subsystem models, while the underlying rationale behind these simplifications is not sufficiently explained. The authors should clarify how the proposed modeling approach balances computational efficiency with the necessity of preserving the key operational and physical variables that determine data center flexibility provision.

🟢 Maybe we should answer these questions directly: why we use aggregate modeling , and why these simplifications are acceptable.

🟢 Regarding whether flexibility disrupts normal operations or efficiency, we should clarify that the required computational power remains the same, meaning performance is not sacrificed. We also need to define our baseline better, showing that standard operations continue uninterrupted.

🟢 We can explain why integrating the three subsystems provides better insights than three separate optimizations. We can clearly stated that

🟢 (a) What this model can do that others cannot—specifically, the duration-aware flexibility assessment (Scenario 3) over a continuous range of (t₀, ΔP) pairs ? Or more?

🟢 (b) The modeling rationale: since cooling load is thermally coupled to the IT load, optimizing them separately mischaracterizes upward flexibility. Deferring IT workload simultaneously reduces cooling needs, an interaction only captured in an integrated model.

🟢 We can extend this.

🟢 Regarding the justification of simplification we can write an answer to reviewer like this:

🟢 The objective of this work is not to reproduce the detailed operational behaviour of individual servers or cooling devices, or creating a digital twin of a data centre but rather to quantify the flexibility potential available to the power system from a data centre. Therefore, the modelling approach prioritises the representation of flexibility-relevant state variables while maintaining computational tractability.

🟢 The purpose of the cooling system model is to capture the energy-flexibility interaction between thermal storage, cooling demand, and electricity consumption rather than detailed airflow or thermal dynamics within server racks.


> 2. The introduction could benefit from a more structured and direct focus on data centers. The current narrative transitions from general energy demand to data center consumption, and then to general flexibility needs and data center flexibility resources and definition. This structure makes the research motivation less direct.

🔴 I think the current structure is: global digitalization → grid challenges → flexibility need → DC as a solution. This is a well-justified narrative funnel that properly contextualizes the research problem. However, we could add one or two sentences at the very beginning to highlight the paper's specific focus, ensuring that anyone reading the first paragraph immediately understands the core contribution.

🟢 I think the gap and contribution is stated at the end of introduction but maybe we can make it more clearer somehow like adding subsection of contributions? Or the sentence starting with Gap etc .


> 3. The workload utilization profiles and flexible/inflexible workload ratios are central to the results, but their data sources and construction process require greater transparency. The authors should provide detailed underlying data sources and the precise derivation steps used to establish these baseline workload characteristics.

🟢 This is the same as Reviewer 2’s 7th comment. While I think the source was already clear, I propose making this change by adding it. We might also consider merging this with the UKPN dataset, since it shows a flatter load utilization. However, this depends on how easy it is to implement and if Jamie would like to change it 😊


> 4. The simplification of computational demand as utilization multiplied by duration is acceptable for an aggregate macro-level model, but the authors should provide relevant literature references or brief discussion to justify the physical and operational validity of this simplification.

⚪ We can a paragraph like :

🟢 In this study, computational demand is represented using the product of cpu utilisation and execution duration. This approximation is commonly used in aggregate workload modelling because the total amount of computational work can be interpreted as the accumulation of processor occupancy over time. For example, a workload operating at 50% utilisation for two hours produces an equivalent computational demand to a workload operating at 100% utilisation for one hour.

⚪ I will take care of this by finding references and justifying this.


> 5. The equations are very detailed, but the overall optimization workflow is not sufficiently intuitive. The authors should provide a more logically organized explanation, or preferably a workflow diagram, to clarify the inputs, outputs, objective functions, decision criteria, and information flow of each scenario.

🟢 We should add a workflow diagram/flowchart as a new figure. The diagram might show the three-scenario flow:

🟢 Inputs: price profile π(t), workload profiles, DC parameters etc and their relations how they used in the model.

🟢 Scenario 1: compute base cost (no optimisation) shows the steps

🟢 Scenario 2: run MILP → output optimised schedule and cost saving

🟢 Scenario 3: take Scenario 2 output as fixed baseline; for each (t₀, ΔP) pair, binary-search over τ via repeated MILP solves → output heatmap and component decomposition

🟢 This will allow readers to easily understand the sequential steps and their relationships while we generate the results


> 6. Although the manuscript states that the Scenario 2 optimization is rerun for each duration in Scenario 3, the selection and interpretation of the reported component-level decomposition remain unclear. The authors should clarify whether Figs. 7 and 8 show just simply one feasible solution returned by the solver? If multiple feasible dispatch solutions exist for the same start time, flexibility magnitude, and duration, the authors should clarify how the component contributions shown in the figures are determined.

🟢 This is fair comment; we should explain this a little bit more.


> 7. The current results mainly describe the contribution of each component. It would be helpful if the authors could further discuss what additional insights can be obtained from modeling these subsystems together rather than separately.

🟢 We can highlight the value of the interaction between workload shifting, UPS batteries, cooling, and TES. We might add a subsection titled 'Value of Integrated Co-optimization' to address the following question: what does modelling IT, UPS, and cooling together reveal that separate models would miss? Although we have already stated this elsewhere, in this section we can specifically highlight:

🟢 The IT–cooling coupling: upward flexibility from IT deferral produces a secondary reduction in cooling power because less heat is generated. A separate IT flexibility model would over-estimate the grid power reduction (it would not credit the cooling reduction); a separate cooling model would under-estimate flexibility.

🟢 The UPS–IT sequencing: IT and UPS providing flexibility at different time slots (IT defers load, then UPS supplies deferred execution power). This temporal coordination is only visible in the integrated model.

🟢 The TES buffering role: downward flexibility relies on coordinated CRAC + TES charging; a cooling-only model would not account for the UPS's simultaneous contribution.

🟢 We can extend this.


> 8. In Fig. 5, same colors are used to denote different positive and negative components. Different hatch patterns or separate legends would improve readability.

🟢 This is also useful comment.  We can change it like that.

