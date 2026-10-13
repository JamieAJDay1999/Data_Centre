# Applied Energy resubmission package — 13 October 2026

**Upload from `paper/UPLOAD_TO_APPLIED_ENERGY/`;
see [the upload guide](UPLOAD_GUIDE.md).** This folder holds the editable
working records and the scripts that produce those files.

**Previous manuscript:** APEN-D-25-21553 (rejected; resubmission permitted)
**Title:** Characterisation and Quantification of Data Centre Flexibility for Power System Support
**Authors:** James Day, Mehmet Türker Takci, Meysam Qadrdan (previous order: Takci, Day, Qadrdan)
**Corresponding author:** James Day, DayJa1@cardiff.ac.uk

## Baseline for the marked copy

The submitted manuscript is the Editorial Manager PDF
`author_records/submitted_manuscript_APEN-D-25-21553.pdf` (Takci first; 10.02%
single-day saving; SCIP). Its source is `author_records/submitted_manuscript_source.tex`
(git `a72f90c:paper/original_submission.tex`). `build_marked.py` compares the
current `main.tex` against it. The `unsubmitted_draft_6.3pct*` files are a later
local draft that was never submitted and must not be used as the baseline.

## Rebuilding

`finalise.ps1` runs the whole chain: marked copy,
LaTeX build, correspondence DOCX/PDF (via Word), source/artwork packaging and
the refresh of the upload folder. No data repository is used; `package_research.py`
and `research_data_and_code.zip` are retained as an internal record only.

## Checklist

- [x] Cover letter cites APEN-D-25-21553, explains the author-order change and
  states that 5.38% (annual) supersedes the 10.02% single-day result.
- [x] Marked copy rebuilt against the submitted Editorial Manager version.
- [x] All 15 reviewer comments answered; page references checked against the clean PDF.
- [x] "DC" abbreviation removed; Figures 1–3 redrawn at printed size.
- [x] Supplement free of internal revision wording; superseded terminal note removed.
- [x] Source ZIP contains only `main.tex` and its inputs and builds standalone.
- [x] Data statement: inputs public or tabulated; no repository deposit, no file promises.
- [ ] Confirm coauthor agreement to the new author order (stated in the cover letter).
- [ ] Enter ORCIDs/coauthor details and publishing route in the portal; review
  the portal-built PDF before final approval.

`submission_metadata.docx` is a copy-ready aid for the portal form, not an attachment.
