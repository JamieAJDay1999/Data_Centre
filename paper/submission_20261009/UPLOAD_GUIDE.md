# Upload guide — Applied Energy resubmission (13 October 2026)

Checked against the Applied Energy Guide for Authors saved on 13 October 2026
(`paper/Guide for authors - Applied Energy ...pdf`).

## Editorial Manager — `paper/UPLOAD_TO_APPLIED_ENERGY/`

The guide requires the manuscript as editable source ("A PDF is not an
acceptable source file"); double-column layout is allowed for LaTeX. The portal
builds the review PDF from the LaTeX files.

| Upload | Portal item type (pick the closest offered) |
|---|---|
| `01_manuscript_LaTeX_source/main.tex` | Manuscript (LaTeX source file) |
| `IEEEtran.cls` (same folder) | LaTeX macro and style files |
| `Figure_01.pdf` … `Figure_10.pdf` (same folder) | Figure |
| `02_manuscript_marked.pdf` | Marked manuscript / revised manuscript with changes marked |
| `03_cover_letter.pdf` | Cover letter |
| `04_response_to_reviewers.pdf` | Response to reviewers |
| `05_highlights.docx` | Highlights |
| `06_declaration_of_interests.docx` | Declaration of interest |
| `07_supplementary_information.pdf` | Supplementary material |

The bibliography is embedded in `main.tex` (no `.bib`, `.bst` or `.bbl` needed),
`main.tex` is the only `.tex` file and there are no subfolders. If the portal
accepts a ZIP instead, `01_manuscript_LaTeX_source.zip` holds the same 12 files.
Fig. 4 (the workflow) is drawn in `main.tex`, so it has no separate file.

`CHECK_ONLY_manuscript_clean.pdf` is **not** uploaded. Compare it with the PDF
the portal builds before you approve the submission.

Enter the authors in the same order as the title page (James Day, Mehmet Türker
Takci, Meysam Qadrdan), the data statement from `submission_metadata.docx`, and use
`submission_metadata.docx` for the remaining form text.

No data repository is used: all inputs are public or tabulated in the article and
supplement. After any edit, run `./paper/submission_20261009/finalise.ps1`; it
refuses to finish if an upload still mentions Zenodo, a DOI placeholder or CSV files.
