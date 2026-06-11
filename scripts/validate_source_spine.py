#!/usr/bin/env python3
"""Validate the PDF-derived source-spine contract against repo surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.source_spine import (  # noqa: E402
    SOURCE_PDF_PAGES,
    SOURCE_PDF_PATH,
    appendix_section_ids,
    chapter_numbers,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse source-spine validator arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pdf", action="store_true", help="Require the inspected local PDF to exist.")
    return parser.parse_args(argv)


def validate_source_spine(*, require_pdf: bool = False) -> list[str]:
    """Return source-spine validation errors."""
    errors: list[str] = []
    expected = list(chapter_numbers())
    chapters = sorted(
        int(path.name.removeprefix("chapter_"))
        for path in (REPO_ROOT / "chapters").glob("chapter_*")
        if path.is_dir()
    )
    docs = sorted(
        int(path.stem.removeprefix("chapter_"))
        for path in (REPO_ROOT / "docs" / "chapters").glob("chapter_*.md")
        if path.stem.removeprefix("chapter_").isdigit()
    )
    if expected != list(range(1, 15)):
        errors.append(f"source spine expected chapters 1-14, got {expected}")
    if 15 in expected or (REPO_ROOT / "chapters" / "chapter_15").exists():
        errors.append("PDF source spine has no Chapter 15; chapter_15 must not be present")
    if chapters != expected:
        errors.append(f"chapter directories mismatch: expected {expected}, found {chapters}")
    if docs != expected:
        errors.append(f"chapter docs mismatch: expected {expected}, found {docs}")
    matrix = REPO_ROOT / "docs" / "reference" / "book_topic_matrix.md"
    text = matrix.read_text(encoding="utf-8") if matrix.is_file() else ""
    for section in appendix_section_ids():
        if section not in text:
            errors.append(f"{matrix}: missing appendix section {section}")
    pdf = Path(SOURCE_PDF_PATH)
    if require_pdf and not pdf.is_file():
        errors.append(f"{pdf}: missing inspected source PDF")
    if SOURCE_PDF_PAGES != 1153:
        errors.append(f"expected source PDF page count 1153, got {SOURCE_PDF_PAGES}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the source-spine validator."""
    args = parse_args(argv)
    errors = validate_source_spine(require_pdf=args.require_pdf)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Validated PDF source spine: Chapters 1-14 and Appendices A-D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
