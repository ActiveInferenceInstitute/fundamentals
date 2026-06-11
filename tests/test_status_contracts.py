"""Status and roadmap drift checks."""

from __future__ import annotations

import re
from pathlib import Path

from active_inference.source_spine import chapter_numbers


REPO_ROOT = Path(__file__).resolve().parents[1]


def _chapter_numbers() -> list[int]:
    return sorted(
        int(path.name.removeprefix("chapter_"))
        for path in (REPO_ROOT / "chapters").glob("chapter_*")
        if path.is_dir()
    )


def test_no_top_level_todo_file_replaces_isa() -> None:
    todo_files = sorted(path.name for path in REPO_ROOT.glob("TODO*") if path.is_file())
    assert not todo_files
    assert (REPO_ROOT / "ISA.md").exists()


def test_status_surfaces_agree_on_live_chapters() -> None:
    chapters = _chapter_numbers()
    assert chapters == list(chapter_numbers())
    assert 15 not in chapters

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    isa = (REPO_ROOT / "ISA.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "chapters" / "README.md").read_text(encoding="utf-8")
    runner = (REPO_ROOT / "scripts" / "run_all_figures.py").read_text(encoding="utf-8")

    for chapter in chapters:
        suffix = f"{chapter:02d}"
        assert f"chapter_{suffix}" in readme
        assert f"chapter_{suffix}.md" in docs_index

    assert "Chapters 1-10 are implemented" in isa
    assert "Chapters 11-14 are now first-class Part III chapter folders" in isa
    assert re.search(r"- \[~\] \*\*Part II, Ch\. 8\*\*", readme)
    assert re.search(r"- \[x\] \*\*Part II, Ch\. 9\*\*", readme)
    for chapter in range(11, 15):
        assert f"Part III, Ch. {chapter}" in readme

    assert "discover_chapters" in runner
    assert "range(1, 11)" not in runner
    assert "chapter_10" not in runner
