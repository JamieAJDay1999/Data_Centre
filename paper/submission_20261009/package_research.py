"""Package saved study evidence and computational sources; never run a solver."""
from pathlib import Path
import csv, json, zipfile, hashlib, shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STAGE=ROOT/'.codex_tmp/submission_20261009/research_archive'
STAGE.mkdir(parents=True,exist_ok=True)
paths=set()
def include(pattern):
    paths.update(p for p in ROOT.glob(pattern) if p.is_file())
for pattern in [
    'rolling_optimisation/*.py','inputs/*.py','inputs/*.csv',
    'paper/audit_corrected_results.py','paper/generate_annual_results.py',
    'paper/generate_figure1.py','paper/generate_figure2.py','paper/generate_figure3.py','paper/figure_style.py',
    'paper/data/*','paper/evidence/*','paper/supplement/*.tex',
    'paper/tables/*.tex','paper/annual_operation.tex','paper/ups_investment.tex',
    'static/data/inputs/*.csv','static/data/imrp_actuals.csv',
    'static/data/representative_day_flexibility/*.csv',
    'reports/representative_day_flexibility/*.json',
    'reports/representative_day_flexibility/*.csv',
    'reports/final_annual_results/*.csv','reports/final_annual_results/*.json',
    'reports/final_annual_results/*.txt']:
    include(pattern)
with (ROOT/'reports/final_annual_results/annual_endpoints.csv').open() as f:
    scenarios=[r['scenario'] for r in csv.DictReader(f)]
for name in scenarios:
    base=f'static/data/rolling_year_outputs/{name}'
    for pattern in ['annual_committed.csv','annual_summary.json','daily_summary.csv',
                    'run_metadata.json','checkpoints/*.json','days/*.csv']:
        include(base+'/'+pattern)
include('static/data/rolling_year_outputs/2025_optimised_cohort_trace/annual_workload_trace.csv')
for p in sorted(paths):
    dest=STAGE/p.relative_to(ROOT)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)

(STAGE/'requirements.txt').write_text('''# Retained solver versions; Python 3.12 was used for the manuscript runs.
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
pyomo>=6.10.0
highspy>=1.14.0
reportlab>=4
tzdata
''',encoding='utf8')
(STAGE/'README.txt').write_text('''RESEARCH ARCHIVE — version 1.0, 13 October 2026
Characterisation and Quantification of Data Centre Flexibility for Power System Support
James Day, Mehmet Türker Takci and Meysam Qadrdan, Cardiff University
Corresponding author: DayJa1@cardiff.ac.uk
Related article: submitted to Applied Energy

PURPOSE AND SCOPE
This archive supplies the adopted inputs, computational model, retained annual
and event outputs and an independent accounting/checkpoint audit. It supports
the 5.38% annual cost-saving result and certified-duration event grid.
It does not contain manuscripts or the separate revenue-stacking study.

QUICK VERIFICATION (NO OPTIMISATION)
Extract the archive. In its root directory, with Python 3.12 and numpy/pandas:
    python paper/audit_corrected_results.py
This checks all 14 annual chains, 5,110 checkpoint/day pairs and 148 positive-
duration event decompositions. It normally completes in seconds and rewrites
reports/final_annual_results/corrected_evidence_audit.json and audit CSVs.
Expected: status passed; 288 event cells; 280 boundary/cap flags; 8 unresolved
next trials. Three sensitivity horizons exceed the normal 1% gap threshold;
they are explicitly listed in the audit and supplementary information.

Check archive integrity (standard Python only):
    python verify_checksums.py

CONTENTS
paper/data: compact machine-readable inputs, reported endpoints and event grid.
paper/evidence: retained numerical audit and supplementary evidence records.
rolling_optimisation: model, horizon/state handling, event search and plotting.
inputs: model parameters and supporting input helpers.
static/data/inputs: adopted workload arrays. The active day has 96 quarters.
static/data/imrp_actuals.csv: source price file retained with the computations.
static/data/rolling_year_outputs: 14 complete committed annual trajectories,
  daily CSVs, chained checkpoints, run metadata and summaries. The central
  annual cohort execution trace is also included. Non-central detailed cohort
  traces are omitted; the committed schedules and checkpoints are retained.
static/data/representative_day_flexibility: undisturbed planned trajectory and
  all 148 positive-duration event/recovery dispatch CSVs.
reports/representative_day_flexibility: 288 cell trial records and summary.
reports/final_annual_results: annual accounting, selected-day record and audits.

FIGURES FROM SAVED RESULTS
    python paper/generate_annual_results.py
    python -m rolling_optimisation.plot_flexibility_figures
These regenerate plotting outputs and tables from stored results without
launching the optimisation. The manuscript source archive contains the figures
and TeX needed to compile the paper independently of this larger archive.

OPTIONAL RECOMPUTATION (POTENTIALLY LONG)
Install requirements.txt in an isolated environment. Do not casually rerun
the full study: individual annual scenarios may take many minutes or longer.
The stored per-scenario run_metadata.json contains the exact configuration.
Start with input validation:
    python -m rolling_optimisation.run_rolling_year --validate-only
For a short fresh central benchmark with the retained solver and a new output:
    python -m rolling_optimisation.run_rolling_year --solver appsi_highs --start-date 2025-01-01 --end-date 2025-01-02 --scenario-id fresh_benchmark --output-root fresh_results
Inspect runtime and quality before attempting a complete year. Do not overwrite
the retained annual/evidence directories. Solver tie-breaking, versions and
time-limited incumbents may change exact schedules; saved outputs are the
record of the reported results. Historical monthly-week utilities are included
as source context and are not the basis of the current annual conclusions.

INTERPRETATION
Workload profiles are stylised assumptions, not measured site utilisation.
Hourly tranche rows repeat four times consecutively. UTC orders physical
intervals; local day/workload labels use Europe/London, including clock changes.
Costs retain signed prices. The 5.38% result is a reference-price expenditure
comparison, not a full retail bill or grid-service revenue. Electricity use and
peak import increase. Durations are certified feasible values, not all proven
global maxima; assessment caps and unresolved trials remain qualified.

PRICE SOURCE AND RIGHTS
Low Carbon Contracts Company, IMRP actuals:
https://dp.lowcarboncontracts.uk/dataset/imrp-actuals
The authors' bibliography records access on 7 August 2026. Consult the source
portal for the applicable third-party terms. No blanket open licence has been
invented for the third-party price data. The persistent identifier and licence
of this archive are given on its Zenodo record.
''',encoding='utf8')
(STAGE/'verify_checksums.py').write_text('''from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parent
items=json.loads((root/'SHA256SUMS.json').read_text())
for name, expected in items.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==expected, name
print(f'All {len(items)} file checksums verified.')
''',encoding='utf8')
members={p.relative_to(ROOT).as_posix() for p in paths}
members.update({'README.txt','requirements.txt','verify_checksums.py'})
manifest={name:hashlib.sha256((STAGE/name).read_bytes()).hexdigest()
          for name in sorted(members)}
(STAGE/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
archive=HERE/'research_data_and_code.zip'
temporary=HERE/'research_data_and_code.building.zip'
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in sorted(members | {'SHA256SUMS.json'}):
        z.write(STAGE/name,name)
temporary.replace(archive)
print(json.dumps(dict(files=len(manifest)+1,zip_mb=round(archive.stat().st_size/1e6,1),path=str(archive)),indent=2))
