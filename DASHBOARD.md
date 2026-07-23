# FlexDC — Interactive Dashboard

A polished web front-end for the Data Centre demand-flexibility model. It turns the
CSV / image artefacts produced by the existing modelling scripts into an interactive,
presentation-quality dashboard, and can **re-solve the optimisation live** from the
browser.

![sections](static/images/optimisation_outputs/power_resource_bar_chart_reordered.png)

---

## Quick start

```bash
# 1. install dependencies (once)
python -m pip install -r requirements.txt

# 2. launch the dashboard  (opens http://127.0.0.1:5000 automatically)
python run_dashboard.py
```

Useful options:

```bash
python run_dashboard.py --port 8080      # choose a port
python run_dashboard.py --no-browser     # don't auto-open a browser tab
python run_dashboard.py --host 0.0.0.0   # expose on your network
```

Press **Ctrl + C** to stop the server.

> The dashboard reads results from `static/data/…`. If a section says results are
> missing, generate them first with the existing scripts
> (`python nominal_calculation.py`, `python optimisation.py`,
> `python flexibility_duration.py`). The **Scenario Lab** does not need any of these —
> it solves fresh each time.

---

## What's in it

| Section | What it shows |
|---|---|
| **Overview** | Headline KPIs (daily cost, saving, flexibility, PUE), an animated system schematic, and the energy mix. |
| **Baseline** | The unoptimised operating day: stacked grid power vs tariff, thermal response, cost profile, workload mix. |
| **Optimisation** | Cost-minimised schedule vs baseline: savings, load-shifting, cumulative cost, and storage dispatch. |
| **Flexibility** | An interactive envelope heatmap — how long a demand change of a given size can be sustained at each hour. Click any cell to drill into the source-by-source response. |
| **Model & Inputs** | The system schematic, key assumptions, the demand/tariff drivers and the workload-shiftability matrix. |
| **Scenario Lab** | Move sliders (tariff level & volatility, cooling COP, UPS / TES capacity, IT power) and **re-solve the MILP live** (~1–2 s) to see the effect on cost and demand. |

Both a refined **light** and a **dark** theme are included (toggle bottom-left); all
charts are interactive (zoom, hover, and one-click PNG export for papers/slides).

---

## Design principle: wrappers, not edits

The dashboard is a **presentation layer only**. None of the original modelling files
(`nominal_calculation.py`, `optimisation.py`, `flexibility_duration.py`,
`constraints.py`, `inputs/parameters_optimisation.py`) were modified.

```
run_dashboard.py          # launcher (chdir to repo root, start Flask, open browser)
webapp/
  app.py                  # Flask routes + JSON API
  data_access.py          # reads the existing CSVs into JSON (read-only)
  runners.py              # reuses build/solve/post-process from the model scripts
                          #   for live runs, applying whitelisted parameter overrides
  templates/index.html    # single-page shell
static/dashboard/
  css/style.css           # design system (light + dark)
  js/app.js               # app controller (nav, KPIs, scenario lab, theming)
  js/charts.js            # Plotly chart builders (theme-aware)
  js/schematic.js         # animated SVG system schematic
  vendor/plotly.min.js    # charting library, vendored for offline use
```

`webapp/runners.py` imports the model modules and calls their existing
`build_model` / `run_single_calculation` / `run_single_optimization` /
`post_process_results` functions unchanged — so live results are produced by exactly
the same code path as the command-line scripts.

---

## HTTP API (read-only unless noted)

| Endpoint | Purpose |
|---|---|
| `GET /api/overview` | headline KPIs across baseline / optimisation / flexibility |
| `GET /api/nominal` | baseline time-series |
| `GET /api/optimisation` | optimised time-series + savings |
| `GET /api/flexibility` | flexibility duration matrix (heatmap) |
| `GET /api/flexibility/detail?ts=&mag=` | detailed response for one event |
| `GET /api/inputs` | demand, tariff and shiftability inputs |
| `POST /api/run/nominal` | solve the baseline live (JSON body = overrides) |
| `POST /api/run/optimisation` | solve baseline + optimisation live and compare |

Scenario override keys accepted by the `POST` endpoints:
`tariff_scale`, `tariff_volatility`, `cop_hvac`, `ups_capacity_kwh`,
`tes_capacity_kwh`, `max_power_kw`, `idle_power_kw`, `overhead_kw`, `p_chiller_max_w`.
Anything else is ignored, so the endpoints are safe to call with partial input.
