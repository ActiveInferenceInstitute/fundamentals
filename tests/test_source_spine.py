"""PDF source-spine contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from active_inference.source_spine import (
    SOURCE_PDF_PAGES,
    appendix_section_ids,
    chapter_numbers,
    has_chapter,
    has_section,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    """Import the standalone source-spine validator as a module."""
    path = REPO_ROOT / "scripts" / "validate_source_spine.py"
    spec = importlib.util.spec_from_file_location("validate_source_spine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdf_source_spine_has_chapters_1_to_14_only() -> None:
    """The inspected PDF has no Chapter 15."""
    assert chapter_numbers() == tuple(range(1, 15))
    assert SOURCE_PDF_PAGES == 1153
    assert has_chapter(14)
    assert not has_chapter(15)
    assert not (REPO_ROOT / "chapters" / "chapter_15").exists()


def test_appendix_sections_are_explicitly_known() -> None:
    """Appendices A-D are represented by section identifiers."""
    ids = set(appendix_section_ids())
    for section in ("A.1.7", "A.2.5", "B.12", "C.2.2.5", "C.11.4", "D.3.4", "D.4"):
        assert section in ids
        assert has_section(section)


def test_source_spine_validator_passes_current_repo() -> None:
    """The standalone validator agrees with the current chapter/docs surface."""
    validator = _load_validator()
    assert validator.validate_source_spine(require_pdf=False) == []
