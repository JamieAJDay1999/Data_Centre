# IT Workload Mathematical Formulation Visualiser

An interactive web application providing a deep visual exploration and comparative breakdown of the mathematical equations governing data centre IT workload shifting, comparing:
1. **Original Submission** (historical source in Git commit `f1607db` at `paper/original_submission.tex`)
2. **Revised manuscript** (now `paper/main.tex` and `paper/pdf/main.pdf`)

The visualiser is a static historical comparison; it does not load the current
manuscript dynamically.

---

## Quick Start: How to Open

Double-click `index.html` in this folder, or open it in any modern browser (Chrome, Edge, Firefox, Safari):

```powershell
Start-Process "C:\GitHub\Data_Centre\it_equations_visualiser\index.html"
```

No build tools, node modules, web servers, or installations are required.

---

## Key Features & Visual Modules

### 1. Interactive Architecture Flow
- Animated coupled system diagram showing how computational workload demand ($R, \alpha$), CPU utilization allocations ($u(t,k,s)$), piecewise power modeling ($\widehat{f}$), power balance ($P_{\text{grid}} + P_{\text{ups}}$), heat conversion ($Q^{\text{IT}} = P^{\text{IT}}$), and thermal cooling dynamics are linked.
- Click on any block to inspect the governing equations, variables, and physics.

### 2. Side-by-Side Equation Comparator
- Equations 3–11 and downstream equations (UPS continuity, thermal dynamics, annual cost settlement, and flexibility tracking).
- Side-by-side rendered LaTeX math.
- Categorised by:
  - **Substantive Corrections** (Red)
  - **Multi-Day Chaining Extensions** (Green)
  - **Notation & Clarity Streamlining** (Blue)
- Detailed "What Changed", "Why Necessary", and "Was the Original Wrong?" verdicts for every formula.

### 3. Simulator: Horizon & Workload Carryover (Eq. 10)
- Interactive time slider (00:00 to 27:00).
- Visualizes job arrivals across the 24h core and 3h look-ahead.
- Toggle between **Revised Eq. 10 (Split $=$ / $\le$)** and **Original Eq. 10 (Strict $=$)**.
- Explains visually why late-arriving jobs require the $\le$ bound and the cohort residual handoff ($A_{t,k,d+1}$) to avoid truncating legal SLAs or causing MILP infeasibility.

### 4. Simulator: Power Correction (Eq. 3)
- Visualizes the 24h core + 3h extension power profile.
- Toggle between **Paper 1 (Subtracted Baseline)** and **Paper 2 (Full Physical Power)**.
- Shows why subtracting baseline in the extension window broke energy conservation for electrical converter balances and thermodynamic cooling heat rejection.

### 5. Simulator: Piecewise Linear Approximation ($\widehat{f}$)
- Interactive CPU utilization slider ($0 \le u \le 1$).
- Cross-plots the exact non-linear equation $P = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}})u^{1.32}$ against the 4-segment linear approximation $\widehat{f}(u)$ with breakpoints $b_i = (i/4)^{1.5}$.
- Live calculation of approximation error $\Delta P$ (max error: 6.241 kW / 0.624%).

---

## File Structure

```
it_equations_visualiser/
├── index.html                 # Main interactive application with visual simulators
├── equations_deep_dive.html   # New in-depth page: term-by-term reference of all equations
├── styles.css                 # Dark-theme engineering UI and layout
├── app.js                     # Interactive physics engines, canvas plots & math logic
└── README.md                  # Documentation and quick-start guide
```
