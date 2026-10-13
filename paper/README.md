# Current paper

This is the single active package for **Characterisation and Quantification of
Data Centre Flexibility for Power System Support**. It preserves the latest
corrected results (5.38% central annual saving), with submission formatting and
declarations updated on 13 October 2026. See `submission_20261009/README.md` for
the upload package and the remaining Zenodo DOI step.

## Read or edit

- `pdf/main.pdf`: current clean paper. Edit `main.tex`.
- `pdf/main_marked.pdf`: changes against the submitted manuscript
  APEN-D-25-21553 (Takci first; 10.02% daily saving). Additions/replacements are
  blue; deleted material is suppressed. Regenerate after changing the clean text.
- `pdf/supplement.pdf`: current supporting information. Edit `supplement.tex`
  and the files in `supplement/`. This is a clean publication copy.

`images/`, `tables/`, `references.bib`, `annual_operation.tex`, and
`ups_investment.tex` are compilation inputs. `data/` and `evidence/` preserve
the numerical inputs, results and audit records; `revisions/` contains the
reviewer and colleague feedback, not alternative manuscripts.

## Build

After manuscript edits, first run
`python paper/submission_20261009/build_marked.py` from the repository root.
The helper accepts `--perl` and `--latexdiff` paths; its Windows defaults use
Git Perl and MiKTeX's standalone `latexdiff-so`. It compares the expanded current
source against the submitted APEN-D-25-21553 source in
`submission_20261009/author_records/` and marks changed equations as whole
displays. Do not use the unsubmitted 6.3% draft as the baseline.

From the repository root, with MiKTeX or TeX Live on PATH:

```powershell
./paper/build.ps1
```

This runs pdfLaTeX, BibTeX and further pdfLaTeX passes until references settle
(at least two further passes, at most four).
It checks unresolved references and writes only the three PDFs above.
Temporary compilation files go into ignored `paper/build/` and are removed
after success; a failed build retains its logs. No solver runs, Python packages,
old drafts, `output/` directory or external result files are needed to compile.

For Overleaf, upload the TeX files, bibliography, images, tables and supplement
folder from this directory. Choose `main.tex` (clean), `main_marked.tex` (review)
or `supplement.tex` as the main document and use pdfLaTeX.

## Numerical provenance

`generate_annual_results.py`, `generate_figure2.py`, and
`rolling_optimisation/plot_flexibility_figures.py` regenerate figures from
stored results. These are separate from compilation and can overwrite the
checked-in figure assets. `audit_corrected_results.py` checks the full stored
annual/event results; it requires the repository's local `static/` and
`reports/` data. Neither operation is necessary just to edit and build the paper.

The superseded terminal-treatment note is kept only in
`submission_20261009/author_records/`; it is not cited or packaged.

## Previous versions

Superseded submission/revision packages, duplicate PDFs, old Overleaf ZIPs,
one-off editing scripts, temporary renders, and the old balanced-draft review
app were removed on 9 October 2026. The pre-cleanup Git commit is `f1607db`.
Use Git history to recover those files instead of keeping parallel drafts here.
The cleanup also moved the obsolete material to
`C:\GitHub\Data_Centre_cleanup_20261009\` after the environment blocked bulk
deletion. This recovery folder is outside the repository and is not a build input.
The separate `rev_stack/paper/` manuscript belongs to the revenue-stacking
project and is not another version of this paper.
