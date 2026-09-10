# Minimal revision of first_edit.tex

**10 September 2026 corrected-results update:** The manuscript, affected figures/tables and supplement now use the corrected workload results (5.38% central annual saving). See `review/corrected_results_notes.md` and `review/corrected_results.patch` for this narrowly scoped update. Unrelated paper content and the blue comparison against the original submission are preserved.

**Blue comparison against the original submission:** `main.tex` now marks added or changed text blue relative to `paper/original_submission.tex`, with deleted text suppressed. `main_clean.tex` preserves the unmarked manuscript. The first-edit comparison below remains available separately. Newly cited or corrected bibliography entries are also blue; embedded raster figures retain their original colours. Compile `main.tex` for the marked version or `main_clean.tex` for the clean version.

Open `main.tex` for the manuscript and `supplement.tex` for the additional supporting material. This version starts directly from `paper/first_edit.tex`; it does not use the shortened introduction or reorganised component methodology from the earlier rewrite. The original files remain unchanged.

The existing architecture description, literature discussion, nomenclature, workload tables, IT power/work equations, UPS equations and cooling equations remain in the main paper. Equations and numerical inputs that conflict with the retained runs have been corrected in place. No original component section has been relocated to the supplement.

## Review the changes

- `review/comparison.html`: side-by-side comparison with the first edit, highlighting changed text. The two new main-text insertions are expanded so they are visible in the comparison.
- `review/first_edit_to_minimal.patch`: complete source diff, also with insertions expanded.
- `review/change_notes.md`: reasons for the substantive changes and points for author review.
- `review/first_edit.tex`: unchanged source snapshot for comparison.
- `review/preservation_check.json`: automated preservation check. Approximately 97.2% of the introduction's original whitespace-delimited text and 99.3% of the literature review's are retained in order (nomenclature excluded from the introduction metric). All original labels are retained.

Earlier `\rev{...}` amendments remain in the manuscript; the macro renders them in black. It does not identify this latest round of changes. Use the comparison for that purpose.

## Compile and edit

Use pdfLaTeX, BibTeX, then pdfLaTeX twice, selecting `main.tex` or `supplement.tex` as the root document. `build.ps1` compiles both locally. `annual_operation.tex` and `ups_investment.tex` are short additions included in the main text. Figures and tables are local to this package. No optimisation software is needed to compile it.

The PDFs are under the repository's `output/pdf/`; the portable source ZIP is under `output/overleaf/`. The ZIP includes the review material and data as well as the compilable source. Local reconstruction scripts are deliberately excluded from that ZIP because they read the historical repository files; ordinary manuscript edits should be made directly in the TeX sources. Rerunning `revise.py` locally reconstructs the draft and overwrites subsequent manual edits.

The first edit's layout has broadly been retained, including its long main-text nomenclature and detailed equation blocks. This is a colleague-review draft, with final journal formatting and any decisions to shorten or relocate existing material left to the authors. No optimisation was rerun.
