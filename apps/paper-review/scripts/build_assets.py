#!/usr/bin/env python3
"""Render a PDF and create the selectable text data used by the review app."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pdfplumber


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = APP_ROOT.parents[1] / "output" / "pdf" / "data_centre_balanced_revision.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="PDF to package")
    parser.add_argument("--pdftoppm", type=Path, help="Path to the pdftoppm executable")
    parser.add_argument("--dpi", type=int, default=144, help="Page render resolution")
    return parser.parse_args()


def find_pdftoppm(explicit: Path | None) -> str:
    if explicit:
        return str(explicit.resolve())
    discovered = shutil.which("pdftoppm")
    if discovered:
        return discovered
    raise SystemExit("pdftoppm was not found. Pass its path with --pdftoppm.")


def render_pages(source: Path, output_dir: Path, executable: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_page in output_dir.glob("page-*.png"):
        old_page.unlink()
    subprocess.run(
        [executable, "-r", str(dpi), "-png", str(source), str(output_dir / "page")],
        check=True,
    )
    pages = sorted(output_dir.glob("page-*.png"))
    for index, page in enumerate(pages, start=1):
        target = output_dir / f"page-{index:02d}.png"
        if page != target:
            page.replace(target)
    return sorted(output_dir.glob("page-*.png"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_existing_paper_data(path: Path) -> dict | None:
    if not path.is_file():
        return None
    prefix = "window.PAPER_DATA = "
    source = path.read_text(encoding="utf-8")
    if not source.startswith(prefix):
        return None
    payload = source[len(prefix):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("pages"), list) else None


def build_page_data(source: Path, rendered_pages: list[Path], fingerprint: str) -> dict:
    pages = []
    with pdfplumber.open(source) as document:
        if len(document.pages) != len(rendered_pages):
            raise RuntimeError("Rendered page count does not match the PDF page count")
        for page_number, page in enumerate(document.pages, start=1):
            words = []
            for word in page.extract_words(
                x_tolerance=1,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            ):
                text = word["text"].replace("\x00", "").strip()
                if not text:
                    continue
                words.append(
                    {
                        "text": text,
                        "x": round(word["x0"] / page.width * 100, 4),
                        "y": round(word["top"] / page.height * 100, 4),
                        "w": round((word["x1"] - word["x0"]) / page.width * 100, 4),
                        "h": round((word["bottom"] - word["top"]) / page.height * 100, 4),
                    }
                )
            pages.append(
                {
                    "number": page_number,
                    "width": float(page.width),
                    "height": float(page.height),
                    "image": f"assets/pages/page-{page_number:02d}.png",
                    "words": words,
                }
            )

    return {
        "title": "Characterisation and Quantification of Data Centre Flexibility for Power System Support",
        "shortTitle": "Data Centre Flexibility",
        "sourceFile": source.name,
        "fingerprint": fingerprint,
        "pageCount": len(pages),
        "pages": pages,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"PDF not found: {source}")

    assets_dir = APP_ROOT / "assets"
    pages_dir = assets_dir / "pages"
    data_dir = APP_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    packaged_pdf = assets_dir / "data_centre_balanced_revision.pdf"
    paper_data_path = data_dir / "paper-data.js"
    previous_data = load_existing_paper_data(paper_data_path)
    previous_fingerprint = sha256_file(packaged_pdf) if packaged_pdf.is_file() else None
    new_fingerprint = sha256_file(source)

    shutil.copy2(source, packaged_pdf)
    rendered_pages = render_pages(source, pages_dir, find_pdftoppm(args.pdftoppm), args.dpi)
    paper_data = build_page_data(source, rendered_pages, new_fingerprint)

    migration = {
        "version": 1,
        "fromFingerprint": previous_fingerprint,
        "toFingerprint": new_fingerprint,
        "pages": previous_data.get("pages", []) if previous_data else [],
    }
    migration_output = "window.PAPER_COMMENT_MIGRATION = " + json.dumps(
        migration, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"
    migration_path = data_dir / "comment-migration.js"
    if previous_fingerprint != new_fingerprint or not migration_path.is_file():
        migration_path.write_text(migration_output, encoding="utf-8")

    output = "window.PAPER_DATA = " + json.dumps(paper_data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    paper_data_path.write_text(output, encoding="utf-8")
    print(f"Packaged {source.name}: {len(rendered_pages)} pages, {sum(len(p['words']) for p in paper_data['pages'])} words")


if __name__ == "__main__":
    main()
