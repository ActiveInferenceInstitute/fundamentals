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
    REPO_ROOT,
    ChapterEntry,
    DiscoverySnapshot,
    ExtraTopicEntry,
    ScriptEntry,
    discover_chapters,
    discover_snapshot,
    discover_extra_scripts,
    discover_extras,
    discover_scripts,
    run_all_chapters,
    run_all_extras,
    run_chapter,
    run_extra_topic,
    run_script,
)
from .tui import main, prompt_menu, render_menu

__all__ = [
    "main",
    "REPO_ROOT",
    "ChapterEntry",
    "DiscoverySnapshot",
    "ExtraTopicEntry",
    "ScriptEntry",
    "discover_chapters",
    "discover_snapshot",
    "discover_extra_scripts",
    "discover_extras",
    "discover_scripts",
    "run_all_chapters",
    "run_all_extras",
    "run_chapter",
    "run_extra_topic",
    "run_script",
    "prompt_menu",
    "render_menu",
]
