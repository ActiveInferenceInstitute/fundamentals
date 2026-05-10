"""Interactive text menu for running chapter orchestrator scripts.

The menu is consumed by the top-level ``run.sh`` and exposes a single entry
point :func:`main`. It can also be invoked directly via::

    python -m active_inference.menu
    uv run python -m active_inference.menu

Non-interactive flags allow the same module to be used in CI / batch
contexts (``--all``, ``--chapter N``, ``--list``, ``--script PATH``).

All discovery and execution logic lives in :mod:`active_inference.menu.runner`
and :mod:`active_inference.menu.tui`; this ``__init__`` re-exports the
small public surface so callers can build their own front ends.
"""

from .runner import (
    CHAPTER_DIRS,
    REPO_ROOT,
    ChapterEntry,
    ScriptEntry,
    discover_chapters,
    discover_scripts,
    run_all_chapters,
    run_chapter,
    run_script,
)
from .tui import main, prompt_menu, render_menu

__all__ = [
    "main",
    "CHAPTER_DIRS",
    "REPO_ROOT",
    "ChapterEntry",
    "ScriptEntry",
    "discover_chapters",
    "discover_scripts",
    "run_all_chapters",
    "run_chapter",
    "run_script",
    "prompt_menu",
    "render_menu",
]
