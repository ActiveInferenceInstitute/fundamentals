"""Chapter / script discovery and headless execution helpers.

This module factors the logic that used to live in
``scripts/run_all_figures.py`` into reusable functions so the text menu in
:mod:`active_inference.menu.tui` (and any future front end) can call them
without ``subprocess`` plumbing leaking into the UI layer.

Public surface:

* :class:`ChapterEntry` / :class:`ScriptEntry` — descriptors returned by the
  discovery functions.
* :func:`discover_chapters` — find every ``chapters/chapter_<N>/`` folder.
* :func:`discover_extras` — find every ``extras/<topic>/`` folder.
* :func:`discover_demos` — find every ``demo/<slug>/`` folder.
* :func:`discover_scripts` — list the runnable scripts in one chapter.
* :func:`run_script` / :func:`run_chapter` / :func:`run_all_chapters` —
  invoke chapter orchestrators with ``MPLBACKEND=Agg`` and ``--save``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from active_inference.demo_topics import demo_topic_slugs, demo_topic_spec
from active_inference.extra_topics import extra_topic_slugs, extra_topic_spec


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
EXTRAS_ROOT: Path = REPO_ROOT / "extras"
DEMO_ROOT: Path = REPO_ROOT / "demo"
OUTPUT_DIR: Path = REPO_ROOT / "output" / "figures"



_CHAPTER_PATTERN = re.compile(r"^chapter_(\d+)$")
_EXTRA_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _chapter_dirs() -> dict[int, Path]:
    """Capture chapter directories locally; never publish a rebuilding cache."""
    if not CHAPTERS_ROOT.is_dir():
        return {}
    return {
        int(match.group(1)): entry
        for entry in sorted(CHAPTERS_ROOT.iterdir())
        if entry.is_dir() and (match := _CHAPTER_PATTERN.fullmatch(entry.name))
    }


def _topic_dirs(root: Path, registered: Sequence[str]) -> dict[str, Path]:
    """Capture one topic root, ordered by registry then additional local names."""
    if not root.is_dir():
        return {}
    discovered = {
        entry.name: entry for entry in sorted(root.iterdir())
        if entry.is_dir() and _EXTRA_PATTERN.fullmatch(entry.name)
    }
    ordered = {slug: discovered.pop(slug) for slug in registered if slug in discovered}
    ordered.update(discovered)
    return ordered


@dataclass(frozen=True)
class ScriptEntry:
    """One runnable chapter or extras orchestrator."""

    path: Path
    chapter: int | None
    kind: str  # "example", "animation", "visualize", "simulate", "concept", "interactive", "other"
    topic: str | None = None
    demo: str | None = None

    @property
    def name(self) -> str:
        """Return display metadata derived from this repository path."""
        return self.path.name

    @property
    def relative(self) -> str:
        """Return display metadata derived from this repository path."""
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass(frozen=True)
class ChapterEntry:
    """A chapter folder and its scripts."""

    number: int
    path: Path
    scripts: tuple[ScriptEntry, ...] = ()

    @property
    def title(self) -> str:
        """Return display metadata derived from this repository path."""
        return f"Chapter {self.number:02d}"

    @property
    def relative(self) -> str:
        """Return display metadata derived from this repository path."""
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass(frozen=True)
class ExtraTopicEntry:
    """An extras topic folder and its runnable scripts."""

    slug: str
    path: Path
    scripts: tuple[ScriptEntry, ...] = ()

    @property
    def title(self) -> str:
        """Return display metadata derived from this extras topic slug."""
        try:
            return extra_topic_spec(self.slug).title
        except KeyError:
            pass
        return self.slug.replace("_", " ").title()

    @property
    def family(self) -> str:
        """Return the curriculum family for this extras topic when registered."""
        try:
            return extra_topic_spec(self.slug).family
        except KeyError:
            return "Extras"

    @property
    def summary(self) -> str:
        """Return the registered topic summary when available."""
        try:
            return extra_topic_spec(self.slug).summary
        except KeyError:
            return "Extras topic"

    @property
    def chapters(self) -> tuple[int, ...]:
        """Return book chapters associated with this extras topic."""
        try:
            return extra_topic_spec(self.slug).chapters
        except KeyError:
            return ()

    @property
    def sections(self) -> tuple[str, ...]:
        """Return book sections associated with this extras topic."""
        try:
            return extra_topic_spec(self.slug).sections
        except KeyError:
            return ()

    @property
    def relative(self) -> str:
        """Return display metadata derived from this repository path."""
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass(frozen=True)
class DemoTopicEntry:
    """A demo topic folder and its runnable scripts."""

    slug: str
    path: Path
    scripts: tuple[ScriptEntry, ...] = ()

    @property
    def title(self) -> str:
        """Return the registered demo title when available."""
        try:
            return demo_topic_spec(self.slug).title
        except KeyError:
            return self.slug.replace("_", " ").title()

    @property
    def summary(self) -> str:
        """Return the registered demo summary when available."""
        try:
            return demo_topic_spec(self.slug).summary
        except KeyError:
            return "Application demo"

    @property
    def chapters(self) -> tuple[int, ...]:
        """Return book chapters associated with this demo."""
        try:
            return demo_topic_spec(self.slug).chapters
        except KeyError:
            return ()

    @property
    def relative(self) -> str:
        """Return display metadata derived from this repository path."""
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass(frozen=True)
class DiscoverySnapshot:
    """Complete operation-local discovery; later operations capture fresh state."""

    chapters: Mapping[int, ChapterEntry]
    extras: Mapping[str, ExtraTopicEntry]
    demos: Mapping[str, DemoTopicEntry]


def discover_snapshot() -> DiscoverySnapshot:
    """Scan each root and each script directory once, without shared mutation."""
    return DiscoverySnapshot(
        MappingProxyType({entry.number: entry for entry in discover_chapters(include_interactive=True)}),
        MappingProxyType({entry.slug: entry for entry in discover_extras(include_interactive=True)}),
        MappingProxyType({entry.slug: entry for entry in discover_demos(include_interactive=True)}),
    )


def _classify(script: Path) -> str:
    """Classify a chapter script by filename convention."""
    name = script.name
    if name.startswith("animation_"):
        return "animation"
    if name.startswith("simulate_"):
        return "simulate"
    if name.startswith("visualize_"):
        return "visualize"
    if name.startswith("example_"):
        return "example"
    if "interactive" in name:
        return "interactive"
    if re.match(r"^\d{2}_", name):
        return "concept"
    return "other"


def _scan_scripts(
    base: Path, *, chapter: int | None = None,
    topic: str | None = None, demo: str | None = None,
) -> tuple[ScriptEntry, ...]:
    """Capture all launchable Python scripts, including GUI scripts, once."""
    return tuple(
        ScriptEntry(path, chapter, _classify(path), topic=topic, demo=demo)
        for path in sorted(base.glob("*.py"))
        if path.is_file() and not path.name.startswith("_")
    )


def _select_scripts(
    scripts: Iterable[ScriptEntry], *, include_animations: bool = True,
    include_visualizations: bool = True, include_interactive: bool = False,
) -> list[ScriptEntry]:
    """Filter an existing script inventory without another filesystem scan."""
    excluded = set()
    if not include_animations:
        excluded.add("animation")
    if not include_visualizations:
        excluded.add("visualize")
    if not include_interactive:
        excluded.add("interactive")
    return [script for script in scripts if script.kind not in excluded]


def discover_chapters(
    *, snapshot: DiscoverySnapshot | None = None, include_interactive: bool = False,
) -> list[ChapterEntry]:
    """Return numerically ordered chapters, retaining empty chapter folders."""
    entries = (
        snapshot.chapters.values() if snapshot is not None else
        (
            ChapterEntry(number, path, _scan_scripts(path, chapter=number))
            for number, path in sorted(_chapter_dirs().items())
        )
    )
    return [
        replace(entry, scripts=tuple(_select_scripts(
            entry.scripts, include_interactive=include_interactive,
        ))) for entry in entries
    ]


def discover_extras(
    *, snapshot: DiscoverySnapshot | None = None, include_interactive: bool = False,
) -> list[ExtraTopicEntry]:
    """Return ordered extras using one local directory and script inventory."""
    entries = (
        snapshot.extras.values() if snapshot is not None else
        (
            ExtraTopicEntry(slug, path, _scan_scripts(path, topic=slug))
            for slug, path in _topic_dirs(EXTRAS_ROOT, extra_topic_slugs()).items()
        )
    )
    return [
        replace(entry, scripts=tuple(_select_scripts(
            entry.scripts, include_interactive=include_interactive,
        ))) for entry in entries
    ]


def discover_demos(
    *, snapshot: DiscoverySnapshot | None = None, include_interactive: bool = False,
) -> list[DemoTopicEntry]:
    """Return ordered demos using one local directory and script inventory."""
    entries = (
        snapshot.demos.values() if snapshot is not None else
        (
            DemoTopicEntry(slug, path, _scan_scripts(path, demo=slug))
            for slug, path in _topic_dirs(DEMO_ROOT, demo_topic_slugs()).items()
        )
    )
    return [
        replace(entry, scripts=tuple(_select_scripts(
            entry.scripts, include_interactive=include_interactive,
        ))) for entry in entries
    ]


def discover_scripts(
    chapter: int, *, include_animations: bool = True,
    include_visualizations: bool = True, include_interactive: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> list[ScriptEntry]:
    """Select a chapter's scripts from a captured or freshly read inventory."""
    if snapshot is None:
        scripts = _scan_scripts(_chapter_dirs()[chapter], chapter=chapter)
    else:
        scripts = snapshot.chapters[chapter].scripts
    return _select_scripts(
        scripts, include_animations=include_animations,
        include_visualizations=include_visualizations, include_interactive=include_interactive,
    )


def discover_extra_scripts(
    topic: str, *, include_animations: bool = True,
    include_visualizations: bool = True, include_interactive: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> list[ScriptEntry]:
    """Select an extras topic's scripts without rebuilding shared state."""
    if snapshot is None:
        scripts = _scan_scripts(_topic_dirs(EXTRAS_ROOT, extra_topic_slugs())[topic], topic=topic)
    else:
        scripts = snapshot.extras[topic].scripts
    return _select_scripts(
        scripts, include_animations=include_animations,
        include_visualizations=include_visualizations, include_interactive=include_interactive,
    )


def discover_demo_scripts(
    slug: str, *, include_animations: bool = True,
    include_visualizations: bool = True, include_interactive: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> list[ScriptEntry]:
    """Select a demo's scripts without rebuilding shared state."""
    if snapshot is None:
        scripts = _scan_scripts(_topic_dirs(DEMO_ROOT, demo_topic_slugs())[slug], demo=slug)
    else:
        scripts = snapshot.demos[slug].scripts
    return _select_scripts(
        scripts, include_animations=include_animations,
        include_visualizations=include_visualizations, include_interactive=include_interactive,
    )


def _build_env(extra_path: Iterable[Path] | None = None) -> dict[str, str]:
    """Build the subprocess environment for headless chapter execution."""
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
    snapshot: DiscoverySnapshot | None = None,
) -> list[tuple[ScriptEntry, int]]:
    """Run every script in one chapter, returning ``(script, returncode)`` pairs."""
    results: list[tuple[ScriptEntry, int]] = []
    scripts = discover_scripts(
        chapter,
        include_animations=include_animations,
        include_visualizations=include_visualizations,
        snapshot=snapshot,
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


def run_extra_topic(
    topic: str,
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> list[tuple[ScriptEntry, int]]:
    """Run every script in one extras topic, returning ``(script, returncode)`` pairs."""
    results: list[tuple[ScriptEntry, int]] = []
    scripts = discover_extra_scripts(
        topic,
        include_animations=include_animations,
        include_visualizations=include_visualizations,
        snapshot=snapshot,
    )
    if not quiet:
        print(f"\n=== Extra: {topic} ({len(scripts)} scripts) ===", flush=True)
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


def run_demo(
    slug: str,
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> list[tuple[ScriptEntry, int]]:
    """Run every script in one demo topic, returning ``(script, returncode)`` pairs."""
    results: list[tuple[ScriptEntry, int]] = []
    scripts = discover_demo_scripts(
        slug,
        include_animations=include_animations,
        include_visualizations=include_visualizations,
        snapshot=snapshot,
    )
    if not quiet:
        print(f"\n=== Demo: {slug} ({len(scripts)} scripts) ===", flush=True)
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
    snapshot: DiscoverySnapshot | None = None,
) -> dict[int, list[tuple[ScriptEntry, int]]]:
    """Run every chapter, returning a per-chapter results dict."""
    out: dict[int, list[tuple[ScriptEntry, int]]] = {}
    snapshot = discover_snapshot() if snapshot is None else snapshot
    for entry in discover_chapters(snapshot=snapshot):
        out[entry.number] = run_chapter(
            entry.number,
            save=save,
            keep_going=keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            quiet=quiet,
            snapshot=snapshot,
        )
        first_failure = next(
            (rc for _, rc in out[entry.number] if rc != 0), 0
        )
        if first_failure and not keep_going:
            break
    return out


def run_all_extras(
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> dict[str, list[tuple[ScriptEntry, int]]]:
    """Run every extras topic, returning a per-topic results dict."""
    out: dict[str, list[tuple[ScriptEntry, int]]] = {}
    snapshot = discover_snapshot() if snapshot is None else snapshot
    for entry in discover_extras(snapshot=snapshot):
        out[entry.slug] = run_extra_topic(
            entry.slug,
            save=save,
            keep_going=keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            quiet=quiet,
            snapshot=snapshot,
        )
        first_failure = next((rc for _, rc in out[entry.slug] if rc != 0), 0)
        if first_failure and not keep_going:
            break
    return out


def run_all_demos(
    *,
    save: bool = True,
    keep_going: bool = False,
    include_animations: bool = True,
    include_visualizations: bool = True,
    quiet: bool = False,
    snapshot: DiscoverySnapshot | None = None,
) -> dict[str, list[tuple[ScriptEntry, int]]]:
    """Run every demo topic, returning a per-topic results dict."""
    out: dict[str, list[tuple[ScriptEntry, int]]] = {}
    snapshot = discover_snapshot() if snapshot is None else snapshot
    for entry in discover_demos(snapshot=snapshot):
        out[entry.slug] = run_demo(
            entry.slug,
            save=save,
            keep_going=keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            quiet=quiet,
            snapshot=snapshot,
        )
        first_failure = next((rc for _, rc in out[entry.slug] if rc != 0), 0)
        if first_failure and not keep_going:
            break
    return out
