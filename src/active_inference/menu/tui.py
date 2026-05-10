"""Stdlib-only text UI for the chapter orchestrator runner.

Exposes :func:`main` as the module entry point (invoked by ``run.sh`` and
``python -m active_inference.menu``). The TUI is deliberately minimal — no
curses, no rich, no readline — so the same code path works under SSH,
Docker exec, CI logs, and Windows terminals.

Non-interactive flags:

* ``--all``                   run every chapter in order
* ``--chapter N``             run a single chapter
* ``--script PATH``           run one orchestrator
* ``--list``                  print the discovered menu and exit
* ``--no-animations``         skip slow GIF renderers
* ``--keep-going``            don't abort on the first failed script
* ``--no-save``               don't pass ``--save`` (interactive windows)

With no flags the menu enters interactive mode.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .runner import (
    CHAPTER_DIRS,
    REPO_ROOT,
    ChapterEntry,
    ScriptEntry,
    discover_chapters,
    run_all_chapters,
    run_chapter,
    run_script,
)

BANNER = r"""
======================================================================
 Fundamentals of Active Inference - Chapter Runner
 (Python companion to Namjoshi, MIT Press 2026)
======================================================================
"""


def _term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


def render_menu(chapters: Sequence[ChapterEntry]) -> str:
    """Return the printable menu string (no I/O)."""
    width = max(60, min(_term_width(), 100))
    rule = "-" * width
    lines: list[str] = [BANNER.rstrip(), ""]
    lines.append(f"Repository root: {REPO_ROOT}")
    lines.append(rule)
    lines.append("Available chapters:")
    lines.append("")
    if not chapters:
        lines.append("  (no chapters discovered — is `chapters/` missing?)")
    for entry in chapters:
        lines.append(
            f"  [{entry.number:>2}]  {entry.title:<14}  "
            f"{len(entry.scripts):>2} script(s)   {entry.relative}"
        )
    lines.append("")
    lines.append("Bulk actions:")
    lines.append("   [a]  Run ALL chapters (every script with --save)")
    lines.append("   [l]  List every script in every chapter")
    lines.append("   [s]  Run a single script (by path)")
    lines.append("   [w]  Launch the local web UI in your browser")
    lines.append("   [h]  Print this menu again")
    lines.append("   [q]  Quit")
    lines.append(rule)
    return "\n".join(lines)


def _print_scripts(chapter: ChapterEntry) -> None:
    print(f"\n{chapter.title} — {len(chapter.scripts)} script(s)")
    for i, script in enumerate(chapter.scripts, start=1):
        marker = {
            "example": "ex ",
            "animation": "gif",
            "visualize": "viz",
            "concept": "cnp",
            "interactive": "int",
            "other": "   ",
        }.get(script.kind, "   ")
        print(f"  [{i:>2}] ({marker}) {script.name}")
    print()


def _summarize(results: dict[int, list[tuple[ScriptEntry, int]]]) -> int:
    """Print a final summary and return an exit code (0 = all green)."""
    total = sum(len(v) for v in results.values())
    failures = [
        (s, rc) for chapter_results in results.values()
        for s, rc in chapter_results if rc != 0
    ]
    print("\n" + "=" * 60)
    print(f"Ran {total} script(s) across {len(results)} chapter(s).")
    if not failures:
        print("All scripts succeeded.")
        return 0
    print(f"{len(failures)} script(s) failed:")
    for script, rc in failures:
        print(f"  - {script.relative} (returncode={rc})")
    return 1


def _resolve_script_path(raw: str) -> Path | None:
    """Map a user-supplied path or chapter/example fragment to a real file."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.is_file():
        return candidate
    # Fallback: fuzzy match by filename in any chapter directory.
    matches: list[Path] = []
    for chapter_dir in CHAPTER_DIRS.values():
        for path in chapter_dir.glob("*.py"):
            if raw in path.name:
                matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if matches:
        print(f"  Ambiguous script {raw!r} — matched {len(matches)}:")
        for m in matches:
            print(f"    - {m.relative_to(REPO_ROOT)}")
    return None


def prompt_menu(
    *,
    keep_going: bool = False,
    include_animations: bool = True,
    save: bool = True,
) -> int:
    """Interactive loop. Returns a process exit code."""
    chapters = discover_chapters()
    print(render_menu(chapters))
    exit_code = 0
    while True:
        try:
            raw = input("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return exit_code

        if raw in {"q", "quit", "exit"}:
            return exit_code
        if raw in {"h", "?", "help", "menu"}:
            print(render_menu(chapters))
            continue
        if raw == "l":
            for entry in chapters:
                _print_scripts(entry)
            continue
        if raw == "a":
            results = run_all_chapters(
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
            )
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw == "w":
            try:
                from ..web import run_server
            except ImportError as exc:  # pragma: no cover - defensive
                print(f"  Web UI unavailable: {exc}")
                continue
            print("  Starting local web UI — press Ctrl+C in this terminal to stop.")
            try:
                run_server(block=True)
            except KeyboardInterrupt:
                print("  Web UI stopped; returning to menu.")
            continue
        if raw == "s":
            path_input = input("  script path or filename fragment: ").strip()
            if not path_input:
                continue
            resolved = _resolve_script_path(path_input)
            if resolved is None:
                print(f"  No script found matching {path_input!r}")
                continue
            completed = run_script(resolved, save=save)
            exit_code = max(exit_code, completed.returncode)
            continue
        if raw.isdigit():
            number = int(raw)
            if number not in CHAPTER_DIRS:
                print(f"  Unknown chapter {number!r}. Available: "
                      f"{sorted(CHAPTER_DIRS)}")
                continue
            results = {number: run_chapter(
                number,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
            )}
            exit_code = max(exit_code, _summarize(results))
            continue
        print(f"  Unrecognized choice: {raw!r}. Press 'h' for help, 'q' to quit.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="active_inference.menu",
        description="Text menu for running chapter orchestrator scripts.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="Run every script in every chapter and exit.")
    g.add_argument("--chapter", type=int, metavar="N",
                   help="Run every script in chapter N and exit.")
    g.add_argument("--script", metavar="PATH",
                   help="Run a single script (path or filename fragment) and exit.")
    g.add_argument("--list", action="store_true",
                   help="Print the discovered menu and exit.")
    p.add_argument("--no-animations", action="store_true",
                   help="Skip animation_*.py scripts (faster).")
    p.add_argument("--no-visualizations", action="store_true",
                   help="Skip visualize_*.py diagnostic scripts.")
    p.add_argument("--keep-going", action="store_true",
                   help="Don't abort on the first failing script.")
    p.add_argument("--no-save", action="store_true",
                   help="Don't pass --save to chapter scripts (opens GUI windows).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    save = not args.no_save
    include_animations = not args.no_animations
    include_visualizations = not args.no_visualizations

    if args.list:
        chapters = discover_chapters()
        print(render_menu(chapters))
        for entry in chapters:
            _print_scripts(entry)
        return 0

    if args.all:
        results = run_all_chapters(
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
        )
        return _summarize(results)

    if args.chapter is not None:
        if args.chapter not in CHAPTER_DIRS:
            print(f"Unknown chapter {args.chapter!r}. Available: "
                  f"{sorted(CHAPTER_DIRS)}", file=sys.stderr)
            return 2
        results = {args.chapter: run_chapter(
            args.chapter,
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
        )}
        return _summarize(results)

    if args.script is not None:
        resolved = _resolve_script_path(args.script)
        if resolved is None:
            print(f"No script found matching {args.script!r}", file=sys.stderr)
            return 2
        completed = run_script(resolved, save=save)
        return completed.returncode

    # Interactive default.
    return prompt_menu(
        keep_going=args.keep_going,
        include_animations=include_animations,
        save=save,
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
