# Revenue-Stacking Analysis Code (Paper 2)

Implements the methodology of `rev_stack/paper/main.tex` (Sections 4.1–4.9) on
top of the Paper 1 facility model, which is imported unchanged from the repo
root (`constraints.py`, `inputs/parameters_optimisation.py`), scaled ×10 to a
10 MW facility by replication.

All commands are run **from the repo root with the venv active**:

```powershell
.\venv\Scripts\Activate.ps1     # or use .\venv\Scripts\python.exe directly
```

## Run order

```powershell
# 0. Generate the synthetic market-data pack (one year, seeded)
python -m rev_stack.model.generate_data

# 1. Single-day co-optimisation + duration certification (start here)
python -m rev_stack.model.run_day --date 2025-01-15
#    faster, without the certification loop:
python -m rev_stack.model.run_day --date 2025-01-15 --no-certify

# 2. Benchmark ladder B0–B5 (value of co-optimisation, certification &
#    market-sequential operation, RQ3)
python -m rev_stack.model.benchmarks --date 2025-01-15

# 3. Per-asset attribution: leave-one-out + standalone + synergy (RQ2)
python -m rev_stack.model.attribution --date 2025-01-15

# 4. Annual simulation (RQ1). Start with a 30-day sample; --all for 365 days
python -m rev_stack.model.run_year --days 30 --no-certify --jobs 4
python -m rev_stack.model.run_year --days 30 --jobs 4              # certified
#    Capacity Market outer sweep (RQ5):
python -m rev_stack.model.run_year --days 30 --no-certify --cm-sweep 0 2000 5000

# 5. Out-of-sample risk / foresight premium (RQ4, Stage-3 light version)
python -m rev_stack.model.run_scenarios --date 2025-01-15 --n 12

# 6. Figures from whatever results exist
python -m rev_stack.model.plots
```

Useful flags on `run_day` / `run_year`:
`--solver scip|appsi_highs` (default: auto — SCIP if on PATH, else HiGHS),
`--cm-kw 2000` (standing CM commitment), `--soc-floor 0.3|0.5|0.7`
(UPS resilience-floor sensitivity), `--scale 100` (100 MW variant).

## Module map (↔ paper sections)

| File | Paper section | Contents |
|---|---|---|
| `config.py` | — | paths, scaling, defaults, slot/EFA/SP arithmetic |
| `market_parameters.py` | §3, Table III | product definitions: direction, τ_j, granularity, eligible assets |
| `generate_data.py` | §5 | synthetic data pack (schema = real-data schema) |
| `market_data.py` | §5 | `MarketDay` loader — the data/model interface |
| `facility.py` | §4.1 | ×10 parameter scaling + facility model builder (reuses `constraints.py`) |
| `stack_model.py` | §4.2–4.6 | commitments, allocations, headrooms, reservations, objective, solver glue |
| `certification.py` | §4.5 | worst-case staircase delivery tests + envelope-cut loop |
| `postprocess.py` | — | summaries, commitment/dispatch frames |
| `run_day.py` | §4 | single-day solve + certify |
| `run_year.py` | §4.9/Stage 2 | multi-day simulation, CM sweep, annual roll-up |
| `benchmarks.py` | §4.9 | B0 energy-only … B4 certified (phantom flexibility) … B5 market-sequential (sequencing gap) |
| `attribution.py` | §4.9/RQ2 | leave-one-out, standalone, synergy |
| `run_scenarios.py` | §4.8 | out-of-sample price/utilisation scenarios, CVaR, foresight premium |
| `plots.py` | §7 of doc 04 | revenue waterfall, stack heatmap, asset allocation, dispatch, ladder |

## Swapping in real data

Replace the CSVs in `rev_stack/data/` file-for-file (schemas documented in
`generate_data.py`'s docstring; sources in `../04_analysis_plan.md` §4).
Nothing downstream changes. Then re-calibrate in `config.py` /
`utilisation_factors.csv`:

- `BM_ACCEPTANCE` (κ) — from comparable units' dispatch rates; sanity-check
  the BM revenue line against the Modo GB benchmark (~£73k/MW/yr all-in for
  a 2 h BESS, of which BM is a minority share).
- `utilisation_factors.csv` (φ_j) — from 1 s frequency data (DC/DM/DR) and
  published activation statistics (BR/QR/SR).
- `cm_params.csv` — actual clearing price and DSR de-rating for the delivery year.

## Modelling notes (documented deviations from the full paper formulation)

- **BM settled on the spread**: accepted offers earn (offer − DA), accepted
  bids earn (DA − bid), scaled by κ — gross-price settlement would create
  free money in a deterministic model.
- **IT headroom** uses a single utilisation floor at the largest IT-eligible
  duration (conservative for shorter products); **IT turn-up headroom = 0**
  (Paper 1 asymmetry). Both are re-checked by certification.
- **SOS2 under HiGHS**: HiGHS lacks SOS support; the SOS2 sets are dropped
  because the u^1.32 curve is convex and only ever under downward pressure,
  making adjacency automatic. SCIP (if on PATH) keeps them.
- **Certification** samples worst-case simultaneous calls at each EFA block
  start (`CERT_START_OFFSETS` in config adds more sample times) and applies
  block-level scale cuts; it is the v1 of the paper's envelope loop.
- **Decision staging**: a single solve co-optimises the day-ahead layer
  (energy profile + availability commitments) and the within-day recourse
  layer (BM, DFS) under perfect foresight — the full-information upper bound
  B4. **B5 (market-sequential)** takes decisions in gate order: the day-ahead
  stage is solved with BM/DFS unavailable and certified; the schedule and
  commitments are then frozen and BM/DFS fill the residual aggregate headroom
  slot by slot (a per-slot LP with one shared pool per direction, solved in
  closed form by winner-take-pool, respecting DFS primacy and the connection
  cap). The B4−B5 gap is the value of within-day foresight.
- **Stochastic stage** is the light out-of-sample variant: fixed first-stage
  commitments **and** day-ahead purchase profile (the metered grid trajectory
  is pinned to its first-stage value, per the paper's balanced-facility
  assumption), re-dispatched per scenario with internal dispatch and BM/DFS
  as recourse; the full two-stage model is Stage-3 work.
- Days are independent (cyclic UPS/TES end-state constraints, as in Paper 1),
  so annual runs parallelise with `--jobs`.
