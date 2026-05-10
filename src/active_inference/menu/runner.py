"""Chapter / script discovery and headless execution helpers.

This module factors the logic that used to live in
``scripts/run_all_figures.py`` into reusable functions so the text menu in
:mod:`active_inference.menu.tui` (and any future front end) can call them
without ``subprocess`` plumbing leaking into the UI layer.

Public surface:

* :class:`ChapterEntry` / :class:`ScriptEntry` — descriptors returned by the
  discovery functions.
* :func:`discover_chapters` — find every ``chapters/chapter_<N>/`` folder.
* :func:`discover_scripts` — list the runnable scripts in one chapter.
* :func:`run_script` / :func:`run_chapter` / :func:`run_all_chapters` —
  invoke chapter orchestrators with ``MPLBACKEND=Agg`` and ``--save``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def _resolve_repo_root() -> Path:
    """Locate the repository root by walking up from this file.

    Editable installs (``uv pip install -e .`` / ``pip install -e .``) leave
    the package inside ``src/active_inference``, so the repo root is three
    levels up. If the package is installed as a wheel into ``site-packages``
    the chapter folders won't exist; callers should detect that via
    :func:`discover_chapters` returning an empty list.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parents[3], here.parents[2], here.parents[1]):
        if (candidate / "chapters").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here.parents[3]


REPO_ROOT: Path = _resolve_repo_root()
CHAPTERS_ROOT: Path = REPO_ROOT / "chapters"
OUTPUT_DIR: Path = REPO_ROOT / "output" / "figures"

# Chapter index. Extend by dropping new ``chapter_<NN>/`` folders into
# ``chapters/`` — discovery picks them up automatically.
CHAPTER_DIRS: dict[int, Path] = {}


_CHAPTER_PATTERN = re.compile(r"^chapter_(\d+)$")


def _refresh_chapter_dirs() -> None:
    CHAPTER_DIRS.clear()
    if not CHAPTERS_ROOT.is_dir():
        return
    for entry in sorted(CHAPTERS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        match = _CHAPTER_PATTERN.match(entry.name)
        if not match:
            continue
        CHAPTER_DIRS[int(match.group(1))] = entry


_refresh_chapter_dirs()


@dataclass(frozen=True)
class ScriptEntry:
    """One runnable chapter orchestrator."""

    path: Path
    chapter: int
    kind: str  # "example", "animation", "visualize", "concept", "interactive", "other"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def relative(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass
class ChapterEntry:
    """A chapter folder and its scripts."""

    number: int
    path: Path
    scripts: list[ScriptEntry] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"Chapter {self.number:02d}"

    @property
    def relative(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


def _classify(script: Path) -> str:
    name = script.name
    if name.startswith("animation_"):
        return "animation"
    if name.startswith("visualize_"):
        return "visualize"
    if name.startswith("example_"):
        return "example"
    if "interactive" in name:
        return "interactive"
    if re.match(r"^\d{2}_", name):
        return "concept"
    return "other"


def _is_runnable(script: Path) -> bool:
    """Skip dunder files, READMEs, and GUI-only scripts."""
    if script.suffix != ".py":
        return False
    if script.name.startswith("_"):
        return False
    if "interactive" in script.name:
        return False
    return True


def discover_chapters() -> list[ChapterEntry]:
    """Return every chapter folder, populated with its scripts.

    The list is sorted by chapter number ascending. Chapter ``0X`` folders
    that don't yet contain any ``.py`` orchestrators are still returned
    (with an empty ``scripts`` list) so callers can communicate "in
    progress".
    """
    _refresh_chapter_dirs()
    return [
        ChapterEntry(
            number=number,
            path=path,
            scripts=discover_scripts(number),
        )
        for number, path in sorted(CHAPTER_DIRS.items())
    ]


def discover_scripts(
    chapter: int,
    *,
    include_animations: bool = True,
    include_visualizations: bool = True,
    include_interactive: bool = False,
) -> list[ScriptEntry]:
    """List the runnable scripts in one chapter.

    The default policy matches the headless smoke tests:

    * Include every ``example_*.py``, ``visualize_*.py``, ``animation_*.py``
      and ``NN_*.py`` script.
    * Skip files whose name contains ``interactive`` (slider windows that
      block until the user closes them).
    """
    _refresh_chapter_dirs()
    if chapter not in CHAPTER_DIRS:
        raise KeyError(f"Unknown chapter: {chapter!r}")
    base = CHAPTER_DIRS[chapter]
    out: list[ScriptEntry] = []
    for candidate in sorted(base.glob("*.py")):
        if not _is_runnable(candidate) and not include_interactive:
            continue
        kind = _classify(candidate)
        if kind == "animation" and not include_animations:
            continue
        if kind == "visualize" and not include_visualizations:
            continue
        if kind == "interactive" and not include_interactive:
            continue
        out.append(ScriptEntry(path=candidate, chapter=chapter, kind=kind))
    return out


def _build_env(extra_path: Iterable[Path] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    src_dir = REPO_ROOT / "src"
    parts = [str(src_dir)]
    if extra_path:
        parts.extend(str(p) for p in extra_path)
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_script(
    script: ScriptEntry | Path,
    *,
    save: bool = True,
    extra_args: Sequence[str] = (),
    capture_output: bool = False,
    timeout: int | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    """Run one chapter script with ``MPLBACKEND=Agg``.

    Returns the :class:`subprocess.CompletedProcess`. The caller is
    responsible for inspecting the return code.
    """
    path = script.path if isinstance(script, ScriptEntry) else Path(script)
    cmd = [sys.executable, str(path)]
    if save:
        cmd.append("--save")
    cmd.extend(extra_args)
    if not quiet:
        try:
            display = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display = str(path)
        print(f"  > {display}", flush=True)
    return subprocess.run(
        cmd,
        env=_build_env(),
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def run_chapter(
    chapter: int,
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
) -> list[tuple[ScriptEntry, int]]:
    """Run every script in one chapter, returning ``(script, returncode)`` pairs."""
    results: list[tuple[ScriptEntry, int]] = []
    scripts = discover_scripts(
        chapter,
        include_animations=include_animations,
        include_visualizations=include_visualizations,
    )
    if not quiet:
        print(f"\n=== Chapter {chapter:02d} ({len(scripts)} scripts) ===", flush=True)
    for script in scripts:
        completed = run_script(script, save=save, quiet=quiet)
        results.append((script, completed.returncode))
        if completed.returncode != 0 and not keep_going:
            if not quiet:
                print(
                    f"  X {script.name} failed (returncode={completed.returncode}). "
                    f"Aborting (use --keep-going to continue).",
                    flush=True,
                )
            break
    return results


def run_all_chapters(
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
) -> dict[int, list[tuple[ScriptEntry, int]]]:
    """Run every chapter, returning a per-chapter results dict."""
    out: dict[int, list[tuple[ScriptEntry, int]]] = {}
    for entry in discover_chapters():
        out[entry.number] = run_chapter(
            entry.number,
            save=save,
            keep_going=keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            quiet=quiet,
        )
        first_failure = next(
            (rc for _, rc in out[entry.number] if rc != 0), 0
        )
        if first_failure and not keep_going:
            break
    return out
