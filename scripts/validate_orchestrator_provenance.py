#!/usr/bin/env python3
"""Validate that chapter and extras orchestrators route through library APIs."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.extra_topics import extra_topic_slugs, extra_topic_spec  # noqa: E402


ALLOWED_EXTERNAL_ROOTS = {
    "__future__",
    "argparse",
    "collections",
    "dataclasses",
    "math",
    "matplotlib",
    "numpy",
    "os",
    "pathlib",
    "scipy",
    "sys",
    "typing",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the orchestrator validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapters-root", type=Path, default=REPO_ROOT / "chapters")
    parser.add_argument("--extras-root", type=Path, default=REPO_ROOT / "extras")
    return parser.parse_args(argv)


def _python_files(root: Path) -> list[Path]:
    """Return local Python files below ``root`` excluding dunder cache files."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    )


def _imports(path: Path) -> list[tuple[int, str, bool]]:
    """Return ``(line, module, is_from_import)`` imports from one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name, False) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append((node.lineno, module, True))
    return imports


def _calls_active_inference(tree: ast.AST) -> bool:
    """Return whether a script references active_inference names or imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("active_inference"):
            return True
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("active_inference") for alias in node.names):
                return True
    return False


def _sibling_modules(directory: Path) -> set[str]:
    """Return importable sibling module stems inside one orchestrator folder."""
    return {
        path.stem
        for path in directory.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    }


def _validate_file(path: Path, *, sibling_modules: set[str]) -> list[str]:
    """Validate one orchestrator file and return human-readable errors."""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    if not _calls_active_inference(tree):
        errors.append(f"{path}: does not route through an active_inference API")

    for line, module, is_from in _imports(path):
        root = module.lstrip(".").split(".", 1)[0]
        if module.startswith("."):
            errors.append(f"{path}:{line}: sibling script import via relative import")
            continue
        if root in sibling_modules and is_from:
            errors.append(f"{path}:{line}: sibling script import {module!r}")
            continue
        if root in {"chapters", "extras"}:
            errors.append(f"{path}:{line}: cross-orchestrator import {module!r}")
            continue
        if root and root not in ALLOWED_EXTERNAL_ROOTS and not module.startswith("active_inference"):
            errors.append(f"{path}:{line}: unexpected non-library import {module!r}")
    return errors


def validate_orchestrators(chapters_root: Path, extras_root: Path) -> list[str]:
    """Validate chapter and extras orchestrator provenance contracts."""
    errors: list[str] = []
    for chapter_dir in sorted(chapters_root.glob("chapter_*")):
        if not chapter_dir.is_dir():
            continue
        siblings = _sibling_modules(chapter_dir)
        for path in _python_files(chapter_dir):
            errors.extend(_validate_file(path, sibling_modules=siblings))

    for slug in extra_topic_slugs():
        topic_dir = extras_root / slug
        if not topic_dir.is_dir():
            errors.append(f"{topic_dir}: missing registered extras topic folder")
            continue
        siblings = _sibling_modules(topic_dir)
        expected = {f"visualize_{slug}.py"}
        spec = extra_topic_spec(slug)
        if spec.has_simulation:
            expected.update({f"simulate_{slug}.py", f"interactive_{slug}.py"})
        if spec.has_animation:
            expected.add(f"animation_{slug}.py")
        present = {path.name for path in _python_files(topic_dir)}
        missing = sorted(expected - present)
        errors.extend(f"{topic_dir / name}: missing declared wrapper" for name in missing)
        for path in _python_files(topic_dir):
            errors.extend(_validate_file(path, sibling_modules=siblings))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run provenance validation and return a process exit code."""
    args = parse_args(argv)
    errors = validate_orchestrators(args.chapters_root, args.extras_root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Validated chapter and extras orchestrator provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
