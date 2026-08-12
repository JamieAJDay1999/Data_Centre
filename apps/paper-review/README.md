# Data Centre paper review app

This is a self-contained browser review copy of `data_centre_balanced_revision.pdf`.
It preserves each page visually, adds a selectable text layer, and lets reviewers pin
comments to exact page locations.

## Open the app

Double-click `index.html`. No server or install step is required.

Comments are stored in that browser's local storage. Use **Export comments** to create
a JSON backup or share comments with another reviewer; use **Import** to merge an export.
When the packaged PDF is rebuilt, existing comments are matched to their nearby source
text. Comments are moved with text that still exists and removed when their source text
has disappeared from the new revision.

Keyboard shortcuts:

- `C` toggles comment pin mode.
- `Esc` cancels comment pin mode or a draft.
- `+` and `-` change the page zoom.

## Rebuild after the PDF changes

Run the asset builder from the repository root. It requires Python with `pdfplumber`
and a Poppler `pdftoppm` executable:

```powershell
python apps/paper-review/scripts/build_assets.py --source output/pdf/data_centre_balanced_revision.pdf
```

If `pdftoppm` is not on `PATH`, pass its full path with `--pdftoppm`.

Generated page images live in `assets/pages/`, and the extracted selectable text lives
in `data/paper-data.js`. The builder also writes `data/comment-migration.js` from the
previous text layer so browser-local comments can be checked on first load. Keeping
everything under `apps/paper-review/` makes the review tool easy to remove, move, or
archive without scattering files through the project.
