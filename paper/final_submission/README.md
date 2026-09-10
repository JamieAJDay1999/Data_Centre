# Corrected results submission package - 10 September 2026

The manuscript and supplement use the completed annual and flexibility runs with the intended hourly tranche mapping: each of the 24 hourly distributions is held constant for four consecutive quarter-hours. The central annual saving is GBP 29,626.91 (5.38%).

## Documents

- `main.tex`: manuscript, with seven numbered figures and four main tables.
- `supplement.tex`: complete inputs, model, procedures, numerical qualifications and investment calculations.
- `highlights.txt`: updated submission highlights.
- `CHANGE_REGISTER.md`: old-versus-corrected values, changed interpretations, and verification record.

The source filenames retain their historical figure numbers. `Figure_4a` and `Figure_4b` form manuscript Figure 3; `Figure_6`, `Figure_7`, `Figure_8` and `Figure_9` become manuscript Figures 4, 5, 6 and 7. `Figure_5` is supplementary Figure S1. The architecture and workflow remain Figures 1 and 2.

The earlier draft's Table 3/Table 4 and S1.1 concerns are addressed in current Section S2.1: Table S3 provides the 24 hourly tranche distributions and Table S4 gives all 96 utilisation/tranche rows. Quarter-hour utilisation is retained as simulated; an hourly starting value is explicitly distinguished from an hourly mean. Earlier drafts are historical sources, not parallel versions of the corrected submission.

## Compile and verify

This package compiles without optimisation results outside the package. On Overleaf, use pdfLaTeX and select `main.tex` or `supplement.tex` as the main document. Locally, run pdfLaTeX, BibTeX, then pdfLaTeX twice for each document. The existing `build.ps1` automates this sequence in the repository and copies the final PDFs into `output/pdf/`.

Run `python verify_package.py` to check asset hashes, corrected input mapping, annual/event values and TeX dependencies using the Python standard library. The `figures/`, `tables/`, `data/` and `evidence/` folders are already prepared; do not run `prepare_assets.py` in a standalone Overleaf upload.

For regeneration within the full Data_Centre repository, run these commands sequentially from the repository root, stopping on any error:

```powershell
python paper/generate_annual_results.py
python -m rolling_optimisation.plot_flexibility_figures
python paper/final_submission/audit_corrected_results.py
python paper/final_submission/prepare_assets.py
python paper/final_submission/verify_package.py
& paper/final_submission/build.ps1
```

These commands read stored solutions and do not execute a solver. `audit_corrected_results.py` requires the full repository's annual checkpoints and event dispatches. The archive includes the representative-day baseline and eight plotted event dispatches; all 148 positive event dispatches remain in the repository. `source_snapshot/` records selected implementation and plotting sources for provenance and is not a standalone full-year solver installation.

## Numerical qualifications

All 14 annual chains cover 365 local dates and 35,040 intervals. Three sensitivity horizons on 25 May have gaps above 1% (maximum 1.301%); their stored feasible solutions remain in the annual totals. Daily objective bounds are not full-year error guarantees.

All 288 event cells passed their recorded validation checks. There are 148 positive certified durations, 280 boundary/cap flags and eight unresolved adjacent trials. Duration boundary/cap flags do not establish global maximality; the supplement explains the changing recovery reference.

`evidence/corrected_evidence_audit.json`, `data/corrected_solver_horizons.csv` and `evidence/corrected_year_end_checkpoint.json` refer to the corrected runs. `evidence/historical/intermediate_quality_reruns.csv` and `evidence/earlier_terminal_assessment.md` predate the correction and are retained only as historical context. They do not certify the corrected runs. The bibliography audit and reference corrections retain the previous editorial review; this update repairs their packaging syntax but does not claim a new external reference review.

The pre-edit submission is preserved in the full repository under `Archive/paper_before_corrected_results_20260910/`.
