// IT Workload Formulation Visualiser - Application Logic
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFlowArchitecture();
  initEquationComparator();
  initSimCarryover();
  initSimPower();
  initSimCurve();
  renderAllMath();
});

// Helper to safely render KaTeX math
function renderAllMath() {
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false
    });
  }
}

window.addEventListener('load', () => {
  setTimeout(renderAllMath, 100);
  setTimeout(renderAllMath, 600);
});

// ----------------------------------------------------
// TAB NAVIGATION
// ----------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = `tab-${tab.dataset.tab}`;
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
        // Trigger resize on canvases if switching to a sim tab
        if (targetId === 'tab-sim-carryover') drawCarryover();
        if (targetId === 'tab-sim-power') drawPower();
        if (targetId === 'tab-sim-curve') drawCurve();
      }
      setTimeout(renderAllMath, 50);
    });
  });
}

// ----------------------------------------------------
// INTERACTIVE ARCHITECTURE FLOW
// ----------------------------------------------------
const flowData = {
  nodeWorkload: {
    title: "IT Workload Arrivals & Cohort Management (\\(\\mathcal{A}_d\\))",
    content: "In the revised formulation, incoming computational demand is tracked in discrete cohorts \\((t,k) \\in \\mathcal{A}_d\\). Each cohort encapsulates its original local arrival time \\(t\\), its assigned flexibility tranche \\(k\\) (with allowable delay \\(D_k \\in \\{0.5, 1, 2, 3\\}\\) hours), its permitted execution interval window \\(W_{t,k}\\), and remaining work \\(A_{t,k,d}\\). Unfinished jobs carried from yesterday naturally exist in \\(\\mathcal{A}_d\\) alongside today's fresh arrivals! Related Equations: Eq. (4), (8), (9), (10)."
  },
  nodeCPU: {
    title: "CPU Allocation & Bounds (\\(u(t,k,s)\\))",
    content: "Decision variables \\(u(t,k,s)\\) allocate CPU resources to tranche \\(k\\) arriving at \\(t\\) and executing at slot \\(s\\). Bounded by \\(0 \\le u \\le u_{\\max}\\) (Eq. 4). Total CPU utilisation at slot \\(s\\) is the sum of inflexible load plus all eligible flexible cohorts executing at \\(s\\), constrained to server capacity \\(u_{\\max} = 1.0\\) (Eq. 11). Work execution accumulates to fulfill constant computational demand \\(R\\) (Eq. 10)."
  },
  nodePowerModel: {
    title: "IT Power Model \\(\\widehat{f}(u)\\) & Linearisation",
    content: "Calculates IT server power from total CPU utilisation. In Paper 1, the non-linear formula \\(P = P_{\\text{idle}} + (P_{\\text{max}} - P_{\\text{idle}})u^{1.32}\\) was written directly. Paper 2 explicitly writes \\(\\widehat{f}\\) (Eq. 3 & Eq. 5), denoting the 4-segment piecewise linear interpolation solved via SOS2/DLOG in the MILP solver. This guarantees mathematical transparency and ensures baseline and optimised cases share the exact same power function."
  },
  nodePowerBalance: {
    title: "Electrical Power Balance (Eq. 43)",
    content: "Couples IT power to electrical sources: \\(P^{\\mathrm{IT}}_{\\mathrm{opt}}(t) = P^{\\mathrm{Grid}}_{\\mathrm{IT}}(t) + P^{\\mathrm{UPS}}_{\\mathrm{disch}}(t)\\). Crucially, because Equation 3 in Paper 2 models full physical power (rather than subtracting baseline in the extension), this converter balance correctly models the real electrical power drawn through the UPS and grid connection in every time slot."
  },
  nodeGridCost: {
    title: "Grid Optimization Objective & Annual Settlement",
    content: "Minimises total electricity cost over the horizon: \\(\\min \\sum (P^{\\mathrm{Grid}}_{\\mathrm{IT}} + P^{\\mathrm{Grid}}_{\\mathrm{OD}} + P^{\\mathrm{UPS}}_{\\mathrm{ch}} + P^{\\mathrm{Chil}}) \\pi \\Delta t\\). In annual operation, only the 24-hour core day \\(T_d\\) decisions are committed into the 365-day annual ledger; advisory look-ahead intervals are re-optimised the next morning, avoiding double-counting (Eq. 12 in annual_operation.tex)."
  },
  nodeCooling: {
    title: "Cooling Infrastructure (Chiller, CRAC, TES)",
    content: "Thermal cooling power \\(Q_{\\mathrm{cool}}(t) = Q^{\\mathrm{Chil}\\text{-}\\mathrm{CRAC}}(t) + Q^{\\mathrm{TES}\\text{-}\\mathrm{CRAC}}(t)\\). Relates electrical chiller power to thermal cooling via \\(\\mathrm{COP}\\). Because IT power generates physical heat inside server racks, the cooling system must modulate to reject heat while keeping temperatures within ASHRAE limits (18°C–23°C)."
  },
  nodeThermal: {
    title: "5-Node Discrete Thermal Dynamics",
    content: "Models heat exchange across Supply Air (\\(T_{Ain}\\)), Server Silicon (\\(T_{IT}\\)), Server Racks (\\(T_{R}\\)), Cold Aisle (\\(T_{CA}\\)), and Hot Aisle (\\(T_{HA}\\)) using implicit backward Euler equations. Hot aisle containment eliminates external convective loss. Coupled directly to IT heat generation: \\(Q^{\\mathrm{IT}}(t) = P^{\\mathrm{IT}}(t)\\)."
  },
  nodeHeat: {
    title: "ITE Electrical Heat Conversion (\\(Q^{\\mathrm{IT}} = P^{\\mathrm{IT}}\\))",
    content: "The first law of thermodynamics dictates that 100% of the electrical energy consumed by processors is converted into heat dissipation within the server chassis. Subtracting baseline power in the extension (as Paper 1 did in Eq. 3) created an artificial cooling deficit. Paper 2 feeds true physical kW into the thermal mass equations."
  },
  nodeUPS: {
    title: "UPS Battery Storage System (ESS)",
    content: "Governed by dynamic energy balance \\(E^{\\mathrm{UPS}}(t+1) = E^{\\mathrm{UPS}}(t) + \\eta_{\\mathrm{ch}} P_{\\mathrm{ch}} \\Delta t - (P_{\\mathrm{dis}}/\\eta_{\\mathrm{dis}})\\Delta t\\). Paper 1 forced \\(E_{\\text{start}} = E_{\\text{end}} = 50\\%\\) each day. Paper 2 chains battery state continuously across days: \\(E^{\\mathrm{UPS}}_{d+1,\\mathrm{start}} = E^{\\mathrm{UPS}}_{d,\\mathrm{core\\ end}}\\), removing artificial end-of-day recharge cycles."
  }
};

function initFlowArchitecture() {
  const nodes = document.querySelectorAll('.flow-node');
  const titleElem = document.getElementById('flowDetailTitle');
  const contentElem = document.getElementById('flowDetailContent');

  nodes.forEach(node => {
    node.addEventListener('click', () => {
      nodes.forEach(n => {
        n.classList.remove('active');
        const rect = n.querySelector('rect');
        if (rect && !n.id.includes('Workload') && !n.id.includes('GridCost') && !n.id.includes('Thermal') && !n.id.includes('UPS')) {
          rect.setAttribute('stroke', '#475569');
          rect.setAttribute('stroke-width', '1.5');
        }
      });

      node.classList.add('active');
      const activeRect = node.querySelector('rect');
      if (activeRect) {
        activeRect.setAttribute('stroke', '#38bdf8');
        activeRect.setAttribute('stroke-width', '2.5');
      }

      const info = flowData[node.id];
      if (info) {
        titleElem.textContent = info.title;
        contentElem.innerHTML = info.content;
        renderAllMath();
      }
    });
  });
}

// ----------------------------------------------------
// EQUATIONS DATA & COMPARATOR
// ----------------------------------------------------
const equationsData = [
  {
    num: "Eq. (3)",
    title: "Optimised IT Power Consumption",
    category: "correction",
    badge: "Substantive Correction + Notation",
    paper1_latex: "P^{\\mathrm{IT}}_{\\mathrm{opt}}(t) = \\begin{cases} P^{\\mathrm{IT}}_{\\mathrm{idle}} + (P_{\\mathrm{max}} - P_{\\mathrm{idle}}) \\left( u^{\\text{inflex}}(t) + \\sum_{j=0}^{D_{\\max}} \\sum_{k=1}^K u(t-j,k,s)|_{s=t} \\right)^{1.32}, & t \\in T \\\\[6pt] P^{\\mathrm{IT}}_{\\mathrm{idle}} + (P_{\\mathrm{max}} - P_{\\mathrm{idle}}) \\left( u^{\\text{inflex}}(t) + \\sum_{j=0}^{D_{\\max}} \\sum_{k=1}^K u(t-j,k,s)|_{s=t} \\right)^{1.32} - P^{\\mathrm{IT}}_{\\mathrm{base}}(t), & t \\in T^{\\text{ext}} \\setminus T \\end{cases}",
    paper1_loc: "original_submission.tex : line 435",
    paper2_latex: "P^{\\mathrm{IT}}_{\\mathrm{opt}}(s) = \\widehat f\\!\\left(u^{\\mathrm{inflex}}(s) + \\sum_{(t,k) \\in \\mathcal A_d :\\, s \\in W_{t,k}} u(t,k,s)\\right), \\quad s \\in T^{\\mathrm{ext}}",
    paper2_loc: "minimal_revision/main_clean.tex : line 469",
    what_changed: "1. The double sum \\(\\sum_{j=0}^{D_{\\max}} \\sum_k u(t-j,k,s)|_{s=t}\\) is replaced by the cohort sum \\(\\sum_{(t,k) \\in \\mathcal{A}_d: s \\in W_{t,k}} u(t,k,s)\\).<br>2. The piecewise definition with extension-period baseline subtraction (\\(- P^{\\mathrm{IT}}_{\\mathrm{base}}\\)) is completely removed.<br>3. The power curve is expressed as \\(\\widehat{f}(\\cdot)\\), explicitly showing the piecewise linear solver approximation.",
    why_necessary: "Subtracting baseline power in the 3h extension was originally conceived to measure incremental cost. However, \\(P^{\\mathrm{IT}}_{\\mathrm{opt}}\\) is physical electrical power that directly drives converter balance (\\(P = P_{\\mathrm{grid}} + P_{\\mathrm{ups}}\\)) and thermal heat generation (\\(Q = P\\)). Subtracting baseline demand breaks physical conservation of energy. Full physical power must be preserved throughout all slots.",
    was_original_wrong: "<strong>Yes, in one specific physical sense:</strong> The baseline subtraction made extension-period power an incremental metric (\\(\\Delta P\\)) rather than physical facility power. Using it as physical demand in energy/heat balances caused an artificial cooling and electrical deficit in the extension. In contrast, the change in summation notation is not a correction of error, but a cleaner mathematical representation."
  },
  {
    num: "Eq. (4)",
    title: "CPU Allocation Bounds",
    category: "extension",
    badge: "Multi-Day Extension",
    paper1_latex: "0 \\le u(t,k,s) \\le u_{\\text{max}}, \\quad \\forall t \\in T, \\forall k \\in K, \\forall s \\in W_{t,k}",
    paper1_loc: "original_submission.tex : line 444",
    paper2_latex: "0 \\le u(t,k,s) \\le u_{\\max}, \\quad (t,k) \\in \\mathcal A_d, \\ s \\in W_{t,k}",
    paper2_loc: "minimal_revision/main_clean.tex : line 478",
    what_changed: "Replaced the single-day arrival index \\(t \\in T, k \\in K\\) with the active cohort set \\((t,k) \\in \\mathcal{A}_d\\).",
    why_necessary: "In continuous annual operation, the horizon contains not only today's arrivals, but also unfinished work carried from yesterday. Indexing over \\(\\mathcal{A}_d\\) bounds utilisation for both new and carried cohorts.",
    was_original_wrong: "<strong>No.</strong> The utilization bounds (\\([0, u_{\\max}]\\)) are identical. The change simply widens the indexing domain to accommodate carried-over workloads across days."
  },
  {
    num: "Eq. (5)",
    title: "Baseline IT Power Consumption",
    category: "notation",
    badge: "Linearisation Consistency",
    paper1_latex: "P^{\\mathrm{IT}}_{\\mathrm{base}}(t) = P^{\\mathrm{IT}}_{\\mathrm{idle}} + (P^{\\mathrm{IT}}_{\\mathrm{max}} - P^{\\mathrm{IT}}_{\\mathrm{idle}}) \\times (u^{\\text{inflex}}(t) + u^{\\text{flex,base}}(t))^{1.32}, \\quad \\forall t \\in T^{\\text{ext}}",
    paper1_loc: "original_submission.tex : line 445",
    paper2_latex: "P^{\\mathrm{IT}}_{\\mathrm{base}}(s) = \\widehat f\\bigl(u^{\\mathrm{inflex}}(s) + u^{\\mathrm{flex,base}}(s)\\bigr)",
    paper2_loc: "minimal_revision/main_clean.tex : line 479",
    what_changed: "Replaced the raw non-linear formula with \\(\\widehat{f}(\\cdot)\\), the exact same 4-segment piecewise linear approximation used in Eq. (3).",
    why_necessary: "Ensures mathematical consistency. If the baseline used the continuous curve while the optimized case used piecewise line segments, cost savings could be biased by piecewise interpolation error. Using \\(\\widehat{f}\\) for both ensures fair benchmark comparison.",
    was_original_wrong: "<strong>No.</strong> The underlying power curve is identical. It makes the solver's internal interpolation explicit in the baseline."
  },
  {
    num: "Eqs. (6–7)",
    title: "IT Energy Relationship",
    category: "notation",
    badge: "Consistent Physical Energy",
    paper1_latex: "E^{\\mathrm{IT}}_{\\mathrm{base}}(t) = \\Delta t \\cdot P^{\\mathrm{IT}}_{\\mathrm{base}}(t), \\quad E^{\\mathrm{IT}}_{\\mathrm{opt}}(t) = \\Delta t \\times P^{\\mathrm{IT}}_{\\mathrm{opt}}(t), \\quad \\forall t \\in T^{\\text{ext}}",
    paper1_loc: "original_submission.tex : lines 447–448",
    paper2_latex: "E^{\\mathrm{IT}}_{\\mathrm{base}}(t) = \\Delta t \\cdot P^{\\mathrm{IT}}_{\\mathrm{base}}(t), \\quad E^{\\mathrm{IT}}_{\\mathrm{opt}}(t) = \\Delta t \\times P^{\\mathrm{IT}}_{\\mathrm{opt}}(t), \\quad \\forall t \\in T^{\\text{ext}}",
    paper2_loc: "minimal_revision/main_clean.tex : lines 480–481",
    what_changed: "The equations themselves are identical (\\(E = P \\Delta t\\)), but the meaning of \\(P^{\\mathrm{IT}}_{\\mathrm{opt}}\\) in the extension period changed due to Equation 3.",
    why_necessary: "Because \\(P^{\\mathrm{IT}}_{\\mathrm{opt}}\\) now represents full physical power throughout \\(T^{\\text{ext}}\\), \\(E^{\\mathrm{IT}}_{\\mathrm{opt}}\\) consistently represents real physical energy rather than incremental energy.",
    was_original_wrong: "<strong>No.</strong> The energy-power integral is always \\(E = P \\Delta t\\). It inherits the physical correction from Equation 3."
  },
  {
    num: "Eq. (8)",
    title: "Permitted Execution Window (Feasibility)",
    category: "notation",
    badge: "Cohort Indexing",
    paper1_latex: "u(t,k,s) = 0 \\quad \\text{if } s < t \\text{ or } s > t + D_k, \\quad \\forall t \\in T, \\forall k \\in K, \\forall s \\in T^{\\text{ext}}",
    paper1_loc: "original_submission.tex : line 449",
    paper2_latex: "u(t,k,s) = 0 \\quad \\text{if } s < t \\text{ or } s > t + D_k, \\quad (t,k) \\in \\mathcal A_d",
    paper2_loc: "minimal_revision/main_clean.tex : line 482",
    what_changed: "Applied to \\((t,k) \\in \\mathcal{A}_d\\) rather than \\(t \\in T, k \\in K\\).",
    why_necessary: "Ensures that jobs carried over from yesterday (which arrived before \\(t=0\\)) still respect their original arrival time \\(t\\) and maximum delay deadline \\(t + D_k\\).",
    was_original_wrong: "<strong>No.</strong> The scheduling rule (cannot run before arrival or after deadline) is identical."
  },
  {
    num: "Eq. (9)",
    title: "Tranche Share Distribution",
    category: "notation",
    badge: "Arrival Clarification",
    paper1_latex: "\\sum_{k=1}^K \\alpha_{t,k} = 1, \\quad \\forall t \\in T",
    paper1_loc: "original_submission.tex : line 451",
    paper2_latex: "\\sum_{k \\in K} \\alpha_{t,k} = 1 \\quad \\text{for each new arrival } t",
    paper2_loc: "minimal_revision/main_clean.tex : line 483",
    what_changed: "Clarified that shares sum to 1 specifically for each <em>new arrival</em> \\(t\\).",
    why_necessary: "Carried cohorts from earlier days already have their tranche assignment fixed when they first arrived. This avoids re-partitioning carried work.",
    was_original_wrong: "<strong>No.</strong> Shares always sum to 100% across the tranches."
  },
  {
    num: "Eq. (10)",
    title: "Job Work Completion Constraint",
    category: "extension",
    badge: "Multi-Day Extension (Crucial)",
    paper1_latex: "\\sum_{s \\in W_{t,k}} u(t,k,s) \\cdot \\Delta t = R_t \\cdot \\alpha_{t,k}, \\quad \\forall t \\in T, \\forall k \\in K",
    paper1_loc: "original_submission.tex : line 452",
    paper2_latex: "\\sum_{s \\in W_{t,k}} u(t,k,s) \\Delta t \\begin{cases} = A_{t,k,d}, & t + D_k \\le L_d, \\\\[4pt] \\le A_{t,k,d}, & t + D_k > L_d \\end{cases}",
    paper2_loc: "minimal_revision/main_clean.tex : lines 484–485",
    what_changed: "1. Strict equality is split into equality (\\(=\\)) if the job's deadline falls within horizon \\(L_d\\), and upper bound (\\(\\le\\)) if the deadline extends beyond \\(L_d\\).<br>2. Work demand is written as \\(A_{t,k,d}\\) (remaining work), which equals \\(R_t \\alpha_{t,k}\\) for new jobs and residual work for carried jobs.",
    why_necessary: "In rolling annual operation, jobs arriving in the late evening or during the look-ahead have legal deadlines that fall into tomorrow's core day. Demanding strict equality within today's solve would artificially force late arrivals to finish before midnight or render the MILP infeasible! The \\(\\le\\) bound allows partial processing, with residual work passed to Day d+1.",
    was_original_wrong: "<strong>For an isolated single day: No.</strong> If all jobs are required to complete within that single day, equality was fine. <strong>For continuous annual operation: Yes, strict equality would be wrong</strong> because it would prematurely clamp deadlines for jobs arriving near the boundary."
  },
  {
    num: "Eq. (11)",
    title: "Server CPU Capacity Constraint",
    category: "notation",
    badge: "Notation & Carried Work",
    paper1_latex: "u^{\\text{inflex}}(t) + \\sum_{k=1}^K \\sum_{j=0}^{D_k} u(t-j, k, s)|_{s=t} \\le u_{\\text{max}}, \\quad \\forall t \\in T^{\\text{ext}}",
    paper1_loc: "original_submission.tex : line 453",
    paper2_latex: "u^{\\mathrm{inflex}}(s) + \\sum_{(t,k) \\in \\mathcal A_d :\\, s \\in W_{t,k}} u(t,k,s) \\le u_{\\max}, \\quad s \\in T^{\\mathrm{ext}}",
    paper2_loc: "minimal_revision/main_clean.tex : line 486",
    what_changed: "Replaced the backwards double summation \\(\\sum_k \\sum_{j=0}^{D_k} u(t-j,k,s)|_{s=t}\\) with the cohort set sum \\(\\sum_{(t,k) \\in \\mathcal{A}_d: s \\in W_{t,k}} u(t,k,s)\\).",
    why_necessary: "Exactly matches the formulation in Equation 3. Avoids awkward lookback indices and naturally includes unfinished carried cohorts from yesterday.",
    was_original_wrong: "<strong>No.</strong> The capacity principle (total CPU usage cannot exceed \\(u_{\\max} = 1.0\\)) is unchanged. The new notation is cleaner and includes carried work."
  },
  {
    num: "Eq. (18)",
    title: "UPS ESS Multi-Day Chaining",
    category: "extension",
    badge: "Multi-Day Extension",
    paper1_latex: "E^{\\mathrm{UPS}}_{\\mathrm{start}} = E^{\\mathrm{UPS}}_{\\mathrm{end}} = (50\\%) \\cdot E^{\\mathrm{UPS}}_{\\mathrm{base}}",
    paper1_loc: "original_submission.tex : line 477",
    paper2_latex: "E^{\\mathrm{UPS}}_{d+1,\\mathrm{start}} = E^{\\mathrm{UPS}}_{d,\\mathrm{core\\ end}}, \\quad E^{\\mathrm{UPS}}_{1,\\mathrm{start}} = E^{\\mathrm{UPS}}_{\\mathrm{base}}",
    paper2_loc: "minimal_revision/main_clean.tex : line 510",
    what_changed: "Replaced artificial daily cyclic reset with continuous inter-day energy continuity.",
    why_necessary: "A real data centre does not force its battery back to exactly 50% at 23:59 every single night. Chaining the committed closing energy to the next day's opening state enables true multi-day energy arbitrage.",
    was_original_wrong: "<strong>No for single-day; Yes for rolling annual operation.</strong> The daily cyclic constraint was an artificial modeling assumption suitable only for isolated single days."
  },
  {
    num: "Handoff",
    title: "Cohort Workload Residual Handoff",
    category: "extension",
    badge: "Rolling Horizon Engine",
    paper1_latex: "\\text{[Not present in original submission: isolated day model]}",
    paper1_loc: "N/A",
    paper2_latex: "A_{t,k,d+1} = A_{t,k,d} - \\sum_{s \\in T_d \\cap W_{t,k}} u(t,k,s)\\Delta t",
    paper2_loc: "annual_operation.tex : line 6 (Eq. 6)",
    what_changed: "Introduced the exact mathematical transfer rule for workload cohorts crossing daily boundaries.",
    why_necessary: "Only work scheduled and committed in the 24-hour core \\(T_d\\) reduces the carried workload. Advisory processing planned during the look-ahead is re-optimized the next day and never counted as finished work, preventing double counting.",
    was_original_wrong: "<strong>N/A.</strong> This was a brand new equation required to enable continuous 365-day rolling horizon execution."
  },
  {
    num: "Annual Cost",
    title: "Annual Settlement Accounting",
    category: "extension",
    badge: "Multi-Day Settlement",
    paper1_latex: "C = \\sum_{t \\in T} (P^{\\mathrm{Grid}}_{\\mathrm{IT}} + P^{\\mathrm{Grid}}_{\\mathrm{OD}} + P^{\\mathrm{Chil}}) \\Delta t \\cdot \\pi(t)",
    paper1_loc: "original_submission.tex : line 561",
    paper2_latex: "C^{(i)}_{\\mathrm{year}} = \\sum_{d=1}^{365} \\sum_{t \\in T_d} P^{\\mathrm{Grid},(i)}(t) \\pi(t) \\Delta t / 1000",
    paper2_loc: "annual_operation.tex : line 12 (Eq. 12)",
    what_changed: "Formulated annual cost as the sum of committed daily core intervals \\(T_d\\) across all 365 days of 2025.",
    why_necessary: "Resolves the economic accounting problem cleanly: each quarter-hour is settled exactly once when it is in the committed core day, ignoring trial look-ahead costs.",
    was_original_wrong: "<strong>No.</strong> Generalizes single-day cost calculation to the full 365-day continuous operational year."
  },
  {
    num: "Flexibility",
    title: "Scenario 3 Cohort Tracking",
    category: "notation",
    badge: "Flexibility Accuracy",
    paper1_latex: "k \\in K = \\{1, 2, \\dots, 12\\}, \\quad D_k = \\{1, 2, \\dots, 12\\} \\text{ [slots]}",
    paper1_loc: "original_submission.tex : line 615",
    paper2_latex: "W^{\\mathrm{event}}_{t,k} = \\{s : \\max(t, t_0) \\le s \\le t + D_k\\} \\cap T^{\\mathrm{event}}, \\quad (t,k) \\in \\mathcal A_{t_0}",
    paper2_loc: "minimal_revision/main_clean.tex : lines 675–676",
    what_changed: "Instead of discarding tranches and creating 12 synthetic remaining-delay classes, Paper 2 preserves the original cohort identity and truncates the remaining allowed execution window.",
    why_necessary: "Tracking individual cohorts prevents an already-deferred job from receiving an artificial fresh delay allowance when an emergency balancing event begins.",
    was_original_wrong: "<strong>No.</strong> Both represent remaining flexibility, but the revised method is mathematically tighter and easier to audit."
  }
];

function initEquationComparator() {
  const container = document.getElementById('equationsGrid');
  const countSpan = document.getElementById('eqCount');
  const filterPills = document.querySelectorAll('#eqFilterPills .filter-pill');

  function render(filter = 'all') {
    container.innerHTML = '';
    const filtered = filter === 'all' 
      ? equationsData 
      : equationsData.filter(d => d.category.includes(filter) || (filter === 'correction' && d.category === 'correction'));

    countSpan.textContent = filtered.length;

    filtered.forEach(eq => {
      const card = document.createElement('div');
      card.className = 'eq-card';

      const badgeClass = eq.category === 'correction' ? 'correction' : (eq.category === 'extension' ? 'extension' : 'notation');

      card.innerHTML = `
        <div class="eq-header">
          <div class="eq-title-group">
            <span class="eq-num-badge">${eq.num}</span>
            <span class="eq-title">${eq.title}</span>
          </div>
          <span class="cat-badge ${badgeClass}">${eq.badge}</span>
        </div>
        <div class="eq-body">
          <div class="eq-comparison-split">
            <div class="eq-column">
              <div class="eq-col-header">
                <span class="eq-col-label paper1">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                  Paper 1: Original Submission
                </span>
                <span class="eq-location">${eq.paper1_loc}</span>
              </div>
              <div class="math-display">$$${eq.paper1_latex}$$</div>
            </div>

            <div class="eq-column">
              <div class="eq-col-header">
                <span class="eq-col-label paper2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                  Paper 2: Minimal Revision
                </span>
                <span class="eq-location">${eq.paper2_loc}</span>
              </div>
              <div class="math-display">$$${eq.paper2_latex}$$</div>
            </div>
          </div>

          <div class="eq-analysis-box">
            <div class="analysis-row">
              <div class="analysis-label">What Changed:</div>
              <div class="analysis-val">${eq.what_changed}</div>
            </div>
            <div class="analysis-row">
              <div class="analysis-label">Why Necessary:</div>
              <div class="analysis-val">${eq.why_necessary}</div>
            </div>
            <div class="analysis-row">
              <div class="analysis-label">Was Original Wrong?</div>
              <div class="analysis-val">${eq.was_original_wrong}</div>
            </div>
          </div>
        </div>
      `;
      container.appendChild(card);
    });

    renderAllMath();
  }

  filterPills.forEach(pill => {
    pill.addEventListener('click', () => {
      filterPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      render(pill.dataset.filter);
    });
  });

  render();
}

// ----------------------------------------------------
// SIMULATOR 1: HORIZON & WORKLOAD CARRYOVER
// ----------------------------------------------------
let sim1State = {
  arrivalHour: 22,
  tranche: 4,
  trancheDelayHours: 3.0,
  mode: 'new' // 'new' or 'old'
};

function initSimCarryover() {
  const arrivalSlider = document.getElementById('jobArrivalSlider');
  const arrivalVal = document.getElementById('jobArrivalVal');
  const trancheSlider = document.getElementById('jobTrancheSlider');
  const trancheVal = document.getElementById('jobTrancheVal');
  const btnNew = document.getElementById('btnModeNew');
  const btnOld = document.getElementById('btnModeOld');

  const delays = [0.5, 1.0, 2.0, 3.0];
  const trancheNames = ["Tranche 1 (0.5h)", "Tranche 2 (1.0h)", "Tranche 3 (2.0h)", "Tranche 4 (3.0h)"];

  arrivalSlider.addEventListener('input', (e) => {
    sim1State.arrivalHour = parseFloat(e.target.value);
    const h = Math.floor(sim1State.arrivalHour);
    const m = (sim1State.arrivalHour % 1) * 60;
    arrivalVal.textContent = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
    updateCarryover();
  });

  trancheSlider.addEventListener('input', (e) => {
    sim1State.tranche = parseInt(e.target.value);
    sim1State.trancheDelayHours = delays[sim1State.tranche - 1];
    trancheVal.textContent = trancheNames[sim1State.tranche - 1];
    updateCarryover();
  });

  btnNew.addEventListener('click', () => {
    sim1State.mode = 'new';
    btnNew.classList.add('btn-primary');
    btnOld.classList.remove('btn-primary');
    updateCarryover();
  });

  btnOld.addEventListener('click', () => {
    sim1State.mode = 'old';
    btnOld.classList.add('btn-primary');
    btnNew.classList.remove('btn-primary');
    updateCarryover();
  });

  window.addEventListener('resize', drawCarryover);
  updateCarryover();
}

function updateCarryover() {
  const deadlineHour = sim1State.arrivalHour + sim1State.trancheDelayHours;
  const horizonEnd = 27.0; // 24h + 3h
  const statusElem = document.getElementById('carryoverStatus');
  const explElem = document.getElementById('carryoverExplanation');

  const crossesHorizon = deadlineHour > horizonEnd;
  const crossesMidnight = deadlineHour > 24.0;

  if (sim1State.mode === 'new') {
    statusElem.style.color = "var(--accent-emerald)";
    statusElem.textContent = "FEASIBLE & CONTINUOUS";
    if (crossesHorizon) {
      explElem.innerHTML = `Deadline is at <strong>${formatTime(deadlineHour)}</strong> (beyond horizon 27:00). Revised Eq. (10) activates the <code>&le;</code> branch: the solver processes up to what fits today, and residual work transfers cleanly to Day d+1 without shortening the job's legal delay window.`;
    } else if (crossesMidnight) {
      explElem.innerHTML = `Deadline is at <strong>${formatTime(deadlineHour)}</strong> (inside look-ahead). Eq. (10) uses <code>=</code> because deadline &le; 27:00. Unfinished work at 24:00 transfers to Day d+1 via Eq. (7) committed residual handoff.`;
    } else {
      explElem.innerHTML = `Deadline is at <strong>${formatTime(deadlineHour)}</strong> (inside Day d Core). Job completes normally today under full equality.`;
    }
  } else {
    // Old mode (strict equality)
    if (crossesHorizon) {
      statusElem.style.color = "var(--accent-rose)";
      statusElem.textContent = "DEADLINE TRUNCATION OR INFEASIBLE!";
      explElem.innerHTML = `Deadline is at <strong>${formatTime(deadlineHour)}</strong>, but the solve horizon ends at 27:00! Original Eq. (10) demanded <code>&sum; u &Delta; t = R</code> within the solve, artificially forcing the job to complete prematurely before 27:00 (violating user SLA) or causing <strong>MILP Infeasibility</strong> if CPU capacity is saturated!`;
    } else {
      statusElem.style.color = "#93c5fd";
      statusElem.textContent = "SOLVABLE (SINGLE DAY)";
      explElem.innerHTML = `Deadline is at <strong>${formatTime(deadlineHour)}</strong>. Completed within the single-day window, but lacks multi-day cohort chaining mechanisms.`;
    }
  }

  drawCarryover();
}

function formatTime(decimalHours) {
  const d = Math.floor(decimalHours / 24);
  const rem = decimalHours % 24;
  const h = Math.floor(rem);
  const m = Math.round((rem % 1) * 60);
  const dayStr = d > 0 ? ` (+${d}d)` : '';
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}${dayStr}`;
}

function drawCarryover() {
  const canvas = document.getElementById('carryoverCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;

  ctx.clearRect(0, 0, W, H);

  // Time range: 0 to 32 hours (Core: 0-24, Look-ahead: 24-27, Next day core: 27-32)
  const totalHours = 32;
  const padLeft = 50;
  const padRight = 30;
  const plotW = W - padLeft - padRight;
  const timeToX = (h) => padLeft + (h / totalHours) * plotW;

  // Background zones
  // Day d Core (0 - 24)
  ctx.fillStyle = 'rgba(30, 58, 138, 0.25)';
  ctx.fillRect(timeToX(0), 40, timeToX(24) - timeToX(0), H - 90);

  // Day d Look-ahead (24 - 27)
  ctx.fillStyle = 'rgba(6, 95, 70, 0.25)';
  ctx.fillRect(timeToX(24), 40, timeToX(27) - timeToX(24), H - 90);

  // Day d+1 Core (27 - 32)
  ctx.fillStyle = 'rgba(55, 65, 81, 0.25)';
  ctx.fillRect(timeToX(27), 40, timeToX(32) - timeToX(27), H - 90);

  // Zone dividing vertical lines
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);

  // Midnight line (24:00)
  ctx.beginPath();
  ctx.moveTo(timeToX(24), 30);
  ctx.lineTo(timeToX(24), H - 40);
  ctx.stroke();

  // Horizon End line (27:00)
  ctx.strokeStyle = '#f43f5e';
  ctx.beginPath();
  ctx.moveTo(timeToX(27), 30);
  ctx.lineTo(timeToX(27), H - 40);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';

  for (let h = 0; h <= totalHours; h += 4) {
    const x = timeToX(h);
    ctx.fillText(`${h % 24}:00`, x, H - 20);
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, H - 40);
    ctx.lineTo(x, H - 35);
    ctx.stroke();
  }

  // Zone Title Headers
  ctx.font = 'bold 12px Inter, sans-serif';
  ctx.fillStyle = '#93c5fd';
  ctx.fillText("Day d Core (Committed T_d)", (timeToX(0) + timeToX(24)) / 2, 25);

  ctx.fillStyle = '#6ee7b7';
  ctx.fillText("3h Look-ahead", (timeToX(24) + timeToX(27)) / 2, 25);

  ctx.fillStyle = '#cbd5e1';
  ctx.fillText("Day d+1 Core", (timeToX(27) + timeToX(32)) / 2, 25);

  // Draw Job Arrival and Permitted Execution Window
  const arrX = timeToX(sim1State.arrivalHour);
  const deadlineH = sim1State.arrivalHour + sim1State.trancheDelayHours;
  const deadX = timeToX(deadlineH);
  const jobY = 120;
  const jobHeight = 40;

  // Window bracket
  ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(arrX, jobY, deadX - arrX, jobHeight, 6);
  ctx.fill();
  ctx.stroke();

  // Arrival marker
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(arrX, jobY + jobHeight / 2, 6, 0, Math.PI * 2);
  ctx.fill();

  // Deadline marker
  const deadlineColor = (deadlineH > 27.0 && sim1State.mode === 'old') ? '#f43f5e' : '#10b981';
  ctx.fillStyle = deadlineColor;
  ctx.beginPath();
  ctx.arc(deadX, jobY + jobHeight / 2, 6, 0, Math.PI * 2);
  ctx.fill();

  // Text inside job window
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 12px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Arrival: ${formatTime(sim1State.arrivalHour)}`, arrX + 12, jobY + 24);

  // Truncation indicator if in old mode and past horizon
  if (deadlineH > 27.0) {
    const horizX = timeToX(27.0);
    if (sim1State.mode === 'old') {
      // Draw red slashed forbidden zone
      ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.fillRect(horizX, jobY, deadX - horizX, jobHeight);
      ctx.fillStyle = '#fda4af';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText("FORCED TRUNCATION", (horizX + deadX) / 2, jobY + jobHeight + 20);
    } else {
      // Draw graceful carryover bracket
      ctx.fillStyle = 'rgba(16, 185, 129, 0.3)';
      ctx.fillRect(horizX, jobY, deadX - horizX, jobHeight);
      ctx.fillStyle = '#6ee7b7';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText("Carried Work A_(t,k,d+1)", (horizX + deadX) / 2, jobY + jobHeight + 20);
    }
  }

  // Draw Handoff boundary indicator
  const midnightX = timeToX(24.0);
  ctx.fillStyle = '#3b82f6';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText("Committed Boundary (Midnight)", midnightX, H - 50);

  const horizonX = timeToX(27.0);
  ctx.fillStyle = '#f43f5e';
  ctx.fillText("Horizon Boundary L_d (03:00)", horizonX, H - 50);
}

// ----------------------------------------------------
// SIMULATOR 2: PHYSICAL POWER CORRECTION (EQ 3)
// ----------------------------------------------------
let sim2State = {
  extUtil: 0.65,
  extBase: 0.50,
  mode: 'revised' // 'revised' or 'original'
};

function initSimPower() {
  const extUtilSlider = document.getElementById('extUtilSlider');
  const extUtilVal = document.getElementById('extUtilVal');
  const extBaseSlider = document.getElementById('extBaseSlider');
  const extBaseVal = document.getElementById('extBaseVal');
  const btnRev = document.getElementById('btnPowerRevised');
  const btnOrig = document.getElementById('btnPowerOriginal');

  extUtilSlider.addEventListener('input', (e) => {
    sim2State.extUtil = parseInt(e.target.value) / 100;
    const kw = calcPowerKw(sim2State.extUtil);
    extUtilVal.textContent = `${e.target.value}% (${kw.toFixed(1)} kW)`;
    updatePower();
  });

  extBaseSlider.addEventListener('input', (e) => {
    sim2State.extBase = parseInt(e.target.value) / 100;
    const kw = calcPowerKw(sim2State.extBase);
    extBaseVal.textContent = `${e.target.value}% (${kw.toFixed(1)} kW)`;
    updatePower();
  });

  btnRev.addEventListener('click', () => {
    sim2State.mode = 'revised';
    btnRev.classList.add('btn-primary');
    btnOrig.classList.remove('btn-primary');
    updatePower();
  });

  btnOrig.addEventListener('click', () => {
    sim2State.mode = 'original';
    btnOrig.classList.add('btn-primary');
    btnRev.classList.remove('btn-primary');
    updatePower();
  });

  window.addEventListener('resize', drawPower);
  updatePower();
}

function calcPowerKw(util) {
  // P_idle = 200 kW, P_max = 1000 kW, exponent = 1.32
  return 200 + 800 * Math.pow(util, 1.32);
}

function updatePower() {
  const actualKw = calcPowerKw(sim2State.extUtil);
  const baseKw = calcPowerKw(sim2State.extBase);

  const reportedKw = sim2State.mode === 'revised' ? actualKw : (actualKw - baseKw);

  const reportedElem = document.getElementById('metricReportedPower');
  const subReportedElem = document.getElementById('subReportedPower');
  const coolingElem = document.getElementById('metricCoolingHeat');
  const subCoolingElem = document.getElementById('subCoolingHeat');
  const verdictElem = document.getElementById('metricPhysicalVerdict');
  const subVerdictElem = document.getElementById('subPhysicalVerdict');

  reportedElem.textContent = `${reportedKw.toFixed(1)} kW`;
  coolingElem.textContent = `${actualKw.toFixed(1)} kW`;

  if (sim2State.mode === 'revised') {
    reportedElem.style.color = "var(--accent-blue-light)";
    subReportedElem.textContent = "True physical electrical load drawn by servers";
    verdictElem.style.color = "var(--accent-emerald)";
    verdictElem.textContent = "100% PHYSICALLY CONSERVED";
    subVerdictElem.textContent = "Grid + UPS power balance and server heat dissipation equal actual consumption.";
  } else {
    reportedElem.style.color = "var(--accent-rose)";
    subReportedElem.textContent = `Incremental Power Delta above baseline (lost ${baseKw.toFixed(1)} kW!)`;
    verdictElem.style.color = "var(--accent-rose)";
    verdictElem.textContent = "PHYSICAL CONSERVATION BROKEN!";
    subVerdictElem.textContent = `Cooling system is fed ${reportedKw.toFixed(1)} kW instead of ${actualKw.toFixed(1)} kW, causing massive thermodynamic under-cooling error!`;
  }

  drawPower();
}

function drawPower() {
  const canvas = document.getElementById('powerCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const padLeft = 60;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotW = W - padLeft - padRight;
  const plotH = H - padTop - padBottom;

  const hours = 27;
  const timeToX = (h) => padLeft + (h / hours) * plotW;
  const kwToY = (kw) => padTop + plotH - (kw / 1100) * plotH;

  // Background zones
  ctx.fillStyle = 'rgba(30, 58, 138, 0.12)';
  ctx.fillRect(timeToX(0), padTop, timeToX(24) - timeToX(0), plotH);
  ctx.fillStyle = 'rgba(244, 63, 94, 0.1)';
  ctx.fillRect(timeToX(24), padTop, timeToX(27) - timeToX(24), plotH);

  // Vertical boundary line at 24h
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(timeToX(24), padTop);
  ctx.lineTo(timeToX(24), padTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Grid lines
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let kw = 0; kw <= 1000; kw += 200) {
    const y = kwToY(kw);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${kw} kW`, padLeft - 8, y + 4);
  }

  // Baseline power profile curve
  ctx.strokeStyle = '#9ca3af';
  ctx.lineWidth = 2;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  for (let h = 0; h <= 27; h += 0.5) {
    const baseU = h < 24 ? (0.45 + 0.25 * Math.sin((h - 8) * Math.PI / 12)) : sim2State.extBase;
    const kw = calcPowerKw(Math.max(0.2, Math.min(0.9, baseU)));
    const x = timeToX(h);
    const y = kwToY(kw);
    if (h === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Optimised / Reported power profile curve
  const actualExtKw = calcPowerKw(sim2State.extUtil);
  const baseExtKw = calcPowerKw(sim2State.extBase);
  const reportedExtKw = sim2State.mode === 'revised' ? actualExtKw : (actualExtKw - baseExtKw);

  ctx.strokeStyle = sim2State.mode === 'revised' ? '#3b82f6' : '#f43f5e';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let h = 0; h <= 24; h += 0.5) {
    const optU = 0.40 + 0.30 * Math.cos((h - 14) * Math.PI / 10);
    const kw = calcPowerKw(Math.max(0.15, Math.min(0.95, optU)));
    const x = timeToX(h);
    const y = kwToY(kw);
    if (h === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  // Extension segment
  ctx.lineTo(timeToX(24.01), kwToY(reportedExtKw));
  ctx.lineTo(timeToX(27), kwToY(reportedExtKw));
  ctx.stroke();

  // If in original mode, show the gap to actual physical heat
  if (sim2State.mode === 'original') {
    // Draw true heat line
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(timeToX(24), kwToY(actualExtKw));
    ctx.lineTo(timeToX(27), kwToY(actualExtKw));
    ctx.stroke();
    ctx.setLineDash([]);

    // Shaded deficit zone
    ctx.fillStyle = 'rgba(244, 63, 94, 0.25)';
    ctx.fillRect(timeToX(24), kwToY(actualExtKw), timeToX(27) - timeToX(24), kwToY(reportedExtKw) - kwToY(actualExtKw));

    ctx.fillStyle = '#fca5a5';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`MISSING POWER DEFICIT: -${baseExtKw.toFixed(1)} kW`, (timeToX(24) + timeToX(27)) / 2, (kwToY(actualExtKw) + kwToY(reportedExtKw)) / 2 + 4);
  }

  // Time axis labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  for (let h = 0; h <= 27; h += 3) {
    ctx.fillText(`${h}:00`, timeToX(h), padTop + plotH + 18);
  }

  ctx.font = 'bold 12px Inter, sans-serif';
  ctx.fillStyle = '#93c5fd';
  ctx.fillText("Main 24h Core Period (T)", (timeToX(0) + timeToX(24)) / 2, padTop - 10);
  ctx.fillStyle = '#fca5a5';
  ctx.fillText("3h Extension Window", (timeToX(24) + timeToX(27)) / 2, padTop - 10);
}

// ----------------------------------------------------
// SIMULATOR 3: PIECEWISE LINEAR POWER CURVE (EQ 3 & 5)
// ----------------------------------------------------
const breakpoints = [0.0, Math.pow(0.25, 1.5), Math.pow(0.5, 1.5), Math.pow(0.75, 1.5), 1.0];
// [0, 0.125, 0.35355, 0.64952, 1.0]
const pwPowerValues = breakpoints.map(b => 200 + 800 * Math.pow(b, 1.32));

let sim3Util = 0.65;

function initSimCurve() {
  const slider = document.getElementById('curveUtilSlider');
  const utilVal = document.getElementById('curveUtilVal');

  slider.addEventListener('input', (e) => {
    sim3Util = parseFloat(e.target.value);
    utilVal.textContent = sim3Util.toFixed(3);
    updateCurve();
  });

  window.addEventListener('resize', drawCurve);
  updateCurve();
}

function calcLinearApprox(u) {
  for (let i = 0; i < breakpoints.length - 1; i++) {
    if (u >= breakpoints[i] && u <= breakpoints[i + 1]) {
      const frac = (u - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i]);
      return pwPowerValues[i] + frac * (pwPowerValues[i + 1] - pwPowerValues[i]);
    }
  }
  return pwPowerValues[pwPowerValues.length - 1];
}

function updateCurve() {
  const exact = 200 + 800 * Math.pow(sim3Util, 1.32);
  const approx = calcLinearApprox(sim3Util);
  const err = approx - exact;

  document.getElementById('metricNonlinearPower').textContent = `${exact.toFixed(2)} kW`;
  document.getElementById('metricLinearPower').textContent = `${approx.toFixed(2)} kW`;

  const errElem = document.getElementById('metricApproxError');
  errElem.textContent = `${err >= 0 ? '+' : ''}${err.toFixed(2)} kW (${((err / 1000) * 100).toFixed(3)}%)`;
  errElem.style.color = Math.abs(err) < 2 ? "var(--accent-emerald)" : "var(--accent-amber)";

  drawCurve();
}

function drawCurve() {
  const canvas = document.getElementById('curveCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const padLeft = 60;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotW = W - padLeft - padRight;
  const plotH = H - padTop - padBottom;

  const uToX = (u) => padLeft + u * plotW;
  const kwToY = (kw) => padTop + plotH - ((kw - 200) / 850) * plotH;

  // Grid
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let kw = 200; kw <= 1000; kw += 100) {
    const y = kwToY(kw);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${kw} kW`, padLeft - 8, y + 4);
  }

  for (let u = 0; u <= 1.0; u += 0.2) {
    const x = uToX(u);
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + plotH);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(u.toFixed(1), x, padTop + plotH + 18);
  }

  // Draw Exact Continuous Curve
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let u = 0; u <= 1.0; u += 0.01) {
    const kw = 200 + 800 * Math.pow(u, 1.32);
    const x = uToX(u);
    const y = kwToY(kw);
    if (u === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Piecewise Linear Approximation Segments
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i < breakpoints.length; i++) {
    const x = uToX(breakpoints[i]);
    const y = kwToY(pwPowerValues[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Breakpoint Nodes
  for (let i = 0; i < breakpoints.length; i++) {
    const x = uToX(breakpoints[i]);
    const y = kwToY(pwPowerValues[i]);
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0b0f19';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label breakpoint
    ctx.fillStyle = '#93c5fd';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`b${i}=${breakpoints[i].toFixed(2)}`, x, y - 10);
  }

  // Draw Current Operating Point
  const curX = uToX(sim3Util);
  const curExactKw = 200 + 800 * Math.pow(sim3Util, 1.32);
  const curApproxKw = calcLinearApprox(sim3Util);
  const curY = kwToY(curApproxKw);

  // Crosshair
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(curX, padTop);
  ctx.lineTo(curX, padTop + plotH);
  ctx.moveTo(padLeft, curY);
  ctx.lineTo(padLeft + plotW, curY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Active dot
  ctx.fillStyle = '#f59e0b';
  ctx.beginPath();
  ctx.arc(curX, curY, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
}
