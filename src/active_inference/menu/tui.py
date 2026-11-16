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
    REPO_ROOT,
    ChapterEntry,
    DiscoverySnapshot,
    DemoTopicEntry,
    ExtraTopicEntry,
    ScriptEntry,
    discover_chapters,
    discover_snapshot,
    discover_demos,
    discover_extras,
    run_all_chapters,
    run_all_demos,
    run_all_extras,
    run_chapter,
    run_demo,
    run_extra_topic,
    run_script,
)

BANNER = r"""
======================================================================
 Fundamentals of Active Inference - Chapter Runner
 (Python companion to Namjoshi, MIT Press 2026)
======================================================================
"""


def _term_width(default: int = 80) -> int:
    """Return the terminal width used for text-menu layout."""
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


def render_menu(
    chapters: Sequence[ChapterEntry],
    extras: Sequence[ExtraTopicEntry] | None = None,
    demos: Sequence[DemoTopicEntry] | None = None,
) -> str:
    """Return the printable menu string (no I/O)."""
    extras = discover_extras() if extras is None else extras
    demos = discover_demos() if demos is None else demos
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
    lines.append("Extras:")
    if not extras:
        lines.append("  (no extras discovered - is `extras/` missing?)")
    last_family = None
    for entry in extras:
        if entry.family != last_family:
            lines.append(f"  {entry.family}:")
            last_family = entry.family
        lines.append(
            f"    [extra:{entry.slug}]  {entry.title:<30}  "
            f"{len(entry.scripts):>2} script(s)   {entry.relative}"
        )
    lines.append("")
    lines.append("Demos:")
    if not demos:
        lines.append("  (no demos discovered - is `demo/` missing?)")
    for entry in demos:
        lines.append(
            f"    [demo:{entry.slug}]  {entry.title:<30}  "
            f"{len(entry.scripts):>2} script(s)   {entry.relative}"
        )
    lines.append("")
    lines.append("Bulk actions:")
    lines.append("   [a]  Run ALL chapters (every script with --save)")
    lines.append("   [x]  Run ALL extras (every extras script with --save)")
    lines.append("   [d]  Run ALL demos (every demo script with --save)")
    lines.append("   [l]  List every script in every chapter")
    lines.append("   [e]  Run one extras topic")
    lines.append("   [m]  Run one demo topic")
    lines.append("   [s]  Run a single script (by path)")
    lines.append("   [w]  Launch the local web UI in your browser")
    lines.append("   [h]  Print this menu again")
    lines.append("   [q]  Quit")
    lines.append(rule)
    return "\n".join(lines)


def _print_scripts(chapter: ChapterEntry | ExtraTopicEntry | DemoTopicEntry) -> None:
    """Print a formatted section of discovered chapter scripts."""
    print(f"\n{chapter.title} — {len(chapter.scripts)} script(s)")
    for i, script in enumerate(chapter.scripts, start=1):
        marker = {
            "example": "ex ",
            "animation": "gif",
            "visualize": "viz",
            "simulate": "sim",
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


def _resolve_script_path(raw: str, *, snapshot: DiscoverySnapshot | None = None) -> Path | None:
    """Map a user-supplied path or chapter/example fragment to a real file."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.is_file():
        return candidate
    snapshot = discover_snapshot() if snapshot is None else snapshot
    matches = [
        script.path
        for entry in (*snapshot.chapters.values(), *snapshot.extras.values(), *snapshot.demos.values())
        for script in entry.scripts if raw in script.name
    ]
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
    snapshot = discover_snapshot()
    chapters = discover_chapters(snapshot=snapshot)
    extras = discover_extras(snapshot=snapshot)
    demos = discover_demos(snapshot=snapshot)
    print(render_menu(chapters, extras, demos))
    exit_code = 0
    while True:
        try:
            raw = input("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return exit_code

        if raw in {"q", "quit", "exit"}:
            return exit_code
        snapshot = discover_snapshot()
        chapters = discover_chapters(snapshot=snapshot)
        extras = discover_extras(snapshot=snapshot)
        demos = discover_demos(snapshot=snapshot)
        if raw in {"h", "?", "help", "menu"}:
            print(render_menu(chapters, extras, demos))
            continue
        if raw == "l":
            for entry in chapters:
                _print_scripts(entry)
            for entry in extras:
                _print_scripts(entry)
            for entry in demos:
                _print_scripts(entry)
            continue
        if raw == "a":
            results = run_all_chapters(
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw == "x":
            results = run_all_extras(
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw == "d":
            results = run_all_demos(
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw == "e":
            topic = input("  extras topic slug: ").strip()
            if not topic:
                continue
            if topic not in snapshot.extras:
                print(f"  Unknown extras topic {topic!r}. Available: "
                      f"{sorted(snapshot.extras)}")
                continue
            results = {topic: run_extra_topic(
                topic,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )}
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw == "m":
            slug = input("  demo topic slug: ").strip()
            if not slug:
                continue
            if slug not in snapshot.demos:
                print(f"  Unknown demo topic {slug!r}. Available: "
                      f"{sorted(snapshot.demos)}")
                continue
            results = {slug: run_demo(
                slug,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )}
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw.startswith("demo:"):
            slug = raw.split(":", 1)[1]
            if slug not in snapshot.demos:
                print(f"  Unknown demo topic {slug!r}. Available: "
                      f"{sorted(snapshot.demos)}")
                continue
            results = {slug: run_demo(
                slug,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )}
            exit_code = max(exit_code, _summarize(results))
            continue
        if raw.startswith("extra:"):
            topic = raw.split(":", 1)[1]
            if topic not in snapshot.extras:
                print(f"  Unknown extras topic {topic!r}. Available: "
                      f"{sorted(snapshot.extras)}")
                continue
            results = {topic: run_extra_topic(
                topic,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )}
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
            resolved = _resolve_script_path(path_input, snapshot=snapshot)
            if resolved is None:
                print(f"  No script found matching {path_input!r}")
                continue
            completed = run_script(resolved, save=save)
            exit_code = max(exit_code, completed.returncode)
            continue
        if raw.isdigit():
            number = int(raw)
            if number not in snapshot.chapters:
                print(f"  Unknown chapter {number!r}. Available: "
                      f"{sorted(snapshot.chapters)}")
                continue
            results = {number: run_chapter(
                number,
                save=save,
                keep_going=keep_going,
                include_animations=include_animations,
                snapshot=snapshot,
            )}
            exit_code = max(exit_code, _summarize(results))
            continue
        print(f"  Unrecognized choice: {raw!r}. Press 'h' for help, 'q' to quit.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(
        prog="active_inference.menu",
        description="Text menu for running chapter orchestrator scripts.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="Run every script in every chapter and exit.")
    g.add_argument("--extras", action="store_true",
                   help="Run every script in every extras topic and exit.")
    g.add_argument("--demos", action="store_true",
                   help="Run every script in every demo topic and exit.")
    g.add_argument("--chapter", type=int, metavar="N",
                   help="Run every script in chapter N and exit.")
    g.add_argument("--extra", metavar="TOPIC",
                   help="Run every script in one extras topic and exit.")
    g.add_argument("--demo", metavar="SLUG",
                   help="Run every script in one demo topic and exit.")
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
    """Run the user-facing entry point for this interface."""
    args = _parse_args(argv)
    save = not args.no_save
    include_animations = not args.no_animations
    include_visualizations = not args.no_visualizations
    snapshot = discover_snapshot()

    if args.list:
        chapters = discover_chapters(snapshot=snapshot)
        extras = discover_extras(snapshot=snapshot)
        demos = discover_demos(snapshot=snapshot)
        print(render_menu(chapters, extras, demos))
        for entry in chapters:
            _print_scripts(entry)
        for entry in extras:
            _print_scripts(entry)
        for entry in demos:
            _print_scripts(entry)
        return 0

    if args.all:
        results = run_all_chapters(
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )
        return _summarize(results)

    if args.extras:
        results = run_all_extras(
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )
        return _summarize(results)

    if args.demos:
        results = run_all_demos(
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )
        return _summarize(results)

    if args.chapter is not None:
        if args.chapter not in snapshot.chapters:
            print(f"Unknown chapter {args.chapter!r}. Available: "
                  f"{sorted(snapshot.chapters)}", file=sys.stderr)
            return 2
        results = {args.chapter: run_chapter(
            args.chapter,
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )}
        return _summarize(results)

    if args.extra is not None:
        if args.extra not in snapshot.extras:
            print(f"Unknown extras topic {args.extra!r}. Available: "
                  f"{sorted(snapshot.extras)}", file=sys.stderr)
            return 2
        results = {args.extra: run_extra_topic(
            args.extra,
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )}
        return _summarize(results)

    if args.demo is not None:
        if args.demo not in snapshot.demos:
            print(f"Unknown demo topic {args.demo!r}. Available: "
                  f"{sorted(snapshot.demos)}", file=sys.stderr)
            return 2
        results = {args.demo: run_demo(
            args.demo,
            save=save,
            keep_going=args.keep_going,
            include_animations=include_animations,
            include_visualizations=include_visualizations,
            snapshot=snapshot,
        )}
        return _summarize(results)

    if args.script is not None:
        resolved = _resolve_script_path(args.script, snapshot=snapshot)
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
