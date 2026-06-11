#!/usr/bin/env python3
"""Validate the book-grounded extras topic coverage matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.extra_topics import EXTRA_TOPICS  # noqa: E402
from active_inference.source_spine import appendix_section_ids, chapter_numbers, has_section  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the coverage validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "docs" / "reference" / "book_topic_matrix.md",
        help="Coverage matrix Markdown file to validate.",
    )
    parser.add_argument(
        "--extras-root",
        type=Path,
        default=REPO_ROOT / "extras",
        help="Extras topic root to validate.",
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=REPO_ROOT / "output" / "figures" / "extras",
        help="Rendered extras figure root used with --require-rendered.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "output" / "data" / "extras",
        help="Rendered extras raw-data root used with --require-rendered.",
    )
    parser.add_argument(
        "--require-rendered",
        action="store_true",
        help="Also require declared rendered PNG/GIF artifacts and NPZ+JSON sidecars.",
    )
    return parser.parse_args(argv)


def _expected_scripts(slug: str, *, simulation: bool, animation: bool) -> list[str]:
    """Return script names required for one registry entry."""
    scripts = [f"visualize_{slug}.py"]
    if simulation:
        scripts.append(f"simulate_{slug}.py")
        scripts.append(f"interactive_{slug}.py")
    if animation:
        scripts.append(f"animation_{slug}.py")
    return scripts


def _expected_artifacts(slug: str, *, simulation: bool, animation: bool) -> list[tuple[str, str]]:
    """Return ``(stem, media_suffix)`` artifacts required for one topic."""
    artifacts = [(f"visualize_{slug}", ".png")]
    if simulation:
        artifacts.append((f"simulate_{slug}", ".png"))
    if animation:
        artifacts.append((f"animation_{slug}", ".gif"))
    return artifacts


def _require_nonempty_file(path: Path, errors: list[str], label: str) -> None:
    """Append a validation error when ``path`` is missing or empty."""
    if not path.is_file():
        errors.append(f"{path}: missing {label}")
        return
    if path.stat().st_size <= 0:
        errors.append(f"{path}: empty {label}")


def validate_coverage(
    matrix: Path,
    extras_root: Path,
    *,
    require_rendered: bool = False,
    figures_root: Path = REPO_ROOT / "output" / "figures" / "extras",
    data_root: Path = REPO_ROOT / "output" / "data" / "extras",
) -> list[str]:
    """Return human-readable coverage validation errors."""
    errors: list[str] = []
    if not matrix.is_file():
        return [f"{matrix}: missing coverage matrix"]
    text = matrix.read_text(encoding="utf-8")
    for spec in EXTRA_TOPICS:
        topic_dir = extras_root / spec.slug
        if not topic_dir.is_dir():
            errors.append(f"{topic_dir}: missing extras topic folder")
            continue
        readme = topic_dir / "README.md"
        if not readme.is_file():
            errors.append(f"{readme}: missing README")
        else:
            readme_text = readme.read_text(encoding="utf-8")
            for phrase in (
                f"output/figures/extras/{spec.slug}",
                f"output/data/extras/{spec.slug}",
            ):
                if phrase not in readme_text:
                    errors.append(f"{readme}: missing artifact path {phrase!r}")
        for script in _expected_scripts(
            spec.slug,
            simulation=spec.has_simulation,
            animation=spec.has_animation,
        ):
            script_path = topic_dir / script
            if not script_path.is_file():
                errors.append(f"{script_path}: missing declared topic script")
        if f"`{spec.slug}`" not in text:
            errors.append(f"{matrix}: missing coverage row for {spec.slug!r}")
        for section in spec.sections:
            if not has_section(section):
                errors.append(f"{spec.slug}: section {section!r} is not in the PDF source spine")
            if section not in text:
                errors.append(f"{matrix}: missing section {section!r} for {spec.slug!r}")
        if require_rendered:
            for stem, suffix in _expected_artifacts(
                spec.slug,
                simulation=spec.has_simulation,
                animation=spec.has_animation,
            ):
                _require_nonempty_file(
                    figures_root / spec.slug / f"{stem}{suffix}",
                    errors,
                    "rendered extras media",
                )
                _require_nonempty_file(
                    data_root / spec.slug / f"{stem}.npz",
                    errors,
                    "extras NPZ sidecar",
                )
                _require_nonempty_file(
                    data_root / spec.slug / f"{stem}.json",
                    errors,
                    "extras JSON sidecar",
                )
    for section in appendix_section_ids():
        if section not in text:
            errors.append(f"{matrix}: missing appendix source-spine section {section!r}")
    chapter_dirs = sorted(
        int(path.name.removeprefix("chapter_"))
        for path in (REPO_ROOT / "chapters").glob("chapter_*")
        if path.is_dir()
    )
    expected = list(chapter_numbers())
    if chapter_dirs != expected:
        errors.append(f"{REPO_ROOT / 'chapters'}: expected chapter directories {expected}, found {chapter_dirs}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run coverage validation and return a process exit code."""
    args = parse_args(argv)
    errors = validate_coverage(
        args.matrix,
        args.extras_root,
        require_rendered=args.require_rendered,
        figures_root=args.figures_root,
        data_root=args.data_root,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    rendered = " with rendered artifacts" if args.require_rendered else ""
    print(f"Validated {len(EXTRA_TOPICS)} extras topic coverage entries{rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
