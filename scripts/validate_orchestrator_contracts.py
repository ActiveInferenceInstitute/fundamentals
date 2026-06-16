#!/usr/bin/env python3
"""Validate chapter/extras orchestrator contracts.

Hard failures are architecture/discovery violations. Soft findings, such as
duplicate stems and long wrappers, are warnings by default so the gate remains
actionable without blocking deliberate transitional debt.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_EXTERNAL_ROOTS = {"matplotlib", "numpy", "scipy"}
CHAPTER_NAME_RE = re.compile(
    r"^(?:\d{2}_[A-Za-z0-9_]+|example_\d+_\d+_[A-Za-z0-9_]+|animation_\d+_\d+_[A-Za-z0-9_]+|"
    r"visualize_[A-Za-z0-9_]+|interactive_[A-Za-z0-9_]+|simulate_[A-Za-z0-9_]+|animation_[A-Za-z0-9_]+)\.py$"
)
EXTRAS_NAME_RE = re.compile(r"^(?:visualize|simulate|animation|interactive)_[A-Za-z0-9_]+\.py$")
SOFT_LINE_LIMIT = 120

try:
    STDLIB_MODULES = set(sys.stdlib_module_names)
except AttributeError:  # pragma: no cover - Python < 3.10 fallback
    STDLIB_MODULES = {
        "argparse",
        "collections",
        "dataclasses",
        "functools",
        "itertools",
        "math",
        "os",
        "pathlib",
        "re",
        "sys",
        "typing",
    }


@dataclass(frozen=True)
class Finding:
    """One validator finding."""

    path: Path
    message: str


def repo_root() -> Path:
    """Return repository root from this script location."""
    return Path(__file__).resolve().parents[1]


def orchestrator_files(root: Path) -> list[Path]:
    """Return chapter and extras Python orchestrator files."""
    paths = [
        *root.glob("chapters/chapter_*/*.py"),
        *root.glob("extras/*/*.py"),
    ]
    return sorted(path for path in paths if not path.name.startswith("_"))


def is_interactive(path: Path) -> bool:
    """Return whether a script is an interactive-only wrapper."""
    return path.name.startswith("interactive_")


def expected_name(path: Path, root: Path) -> bool:
    """Return whether ``path`` follows the discovery naming contract."""
    relative = path.relative_to(root)
    if relative.parts[0] == "chapters":
        return bool(CHAPTER_NAME_RE.fullmatch(path.name))
    if relative.parts[0] == "extras":
        return bool(EXTRAS_NAME_RE.fullmatch(path.name))
    return False


def parse_python(path: Path) -> ast.Module:
    """Parse one Python file with path-aware error reporting."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc}") from exc


def import_roots(tree: ast.Module) -> Iterable[tuple[str, ast.AST]]:
    """Yield import roots and their AST nodes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0], node
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield ".", node
            elif node.module:
                yield node.module.split(".", 1)[0], node


def has_argument(tree: ast.Module, flag: str) -> bool:
    """Return whether an argparse-style ``add_argument(flag)`` call exists."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if any(isinstance(arg, ast.Constant) and arg.value == flag for arg in node.args):
            return True
    return False


def text_uses_save_gate(path: Path) -> bool:
    """Return whether script text branches on the parsed ``--save`` flag."""
    text = path.read_text(encoding="utf-8")
    return "args.save" in text or "namespace.save" in text


def delegates_to_extras_cli(path: Path) -> bool:
    """Return whether an extras wrapper delegates to the shared CLI parser."""
    text = path.read_text(encoding="utf-8")
    return any(
        token in text
        for token in (
            "main_visualize(",
            "main_simulate(",
            "main_animation(",
        )
    )


def validate_file(path: Path, root: Path) -> tuple[list[Finding], list[Finding]]:
    """Return hard errors and soft warnings for one orchestrator file."""
    errors: list[Finding] = []
    warnings: list[Finding] = []
    relative = path.relative_to(root)
    try:
        tree = parse_python(path)
    except ValueError as exc:
        return [Finding(relative, str(exc))], []

    if not expected_name(path, root):
        errors.append(Finding(relative, "filename is not discoverable by chapter/extras naming conventions"))

    roots = list(import_roots(tree))
    imported_root_names = {root_name for root_name, _node in roots}
    if "." in imported_root_names:
        errors.append(Finding(relative, "relative imports are not allowed in orchestrator scripts"))
    if {"chapters", "extras"} & imported_root_names:
        errors.append(Finding(relative, "orchestrators must not import chapter/extras scripts"))
    if "active_inference" not in imported_root_names:
        errors.append(Finding(relative, "orchestrators must import reusable logic through active_inference"))

    for root_name, node in roots:
        if root_name == ".":
            continue
        if root_name == "active_inference":
            continue
        if root_name in ALLOWED_EXTERNAL_ROOTS or root_name in STDLIB_MODULES:
            continue
        errors.append(Finding(relative, f"disallowed import root {root_name!r} at line {getattr(node, 'lineno', '?')}"))

    if not is_interactive(path) and not delegates_to_extras_cli(path):
        if not has_argument(tree, "--save"):
            errors.append(Finding(relative, "non-interactive orchestrator is missing a --save CLI flag"))
        elif not text_uses_save_gate(path):
            errors.append(Finding(relative, "non-interactive orchestrator declares --save but does not branch on it"))

    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > SOFT_LINE_LIMIT:
        warnings.append(Finding(relative, f"wrapper has {line_count} lines; target is <= {SOFT_LINE_LIMIT}"))

    if "numpy" in imported_root_names and not has_argument(tree, "--seed"):
        text = path.read_text(encoding="utf-8")
        if "random" in text or "rng" in text:
            warnings.append(Finding(relative, "script appears stochastic but has no --seed flag"))

    return errors, warnings


def duplicate_stem_warnings(paths: Iterable[Path], root: Path) -> list[Finding]:
    """Return non-fatal warnings for identical stems across surfaces."""
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.stem, []).append(path.relative_to(root))
    warnings: list[Finding] = []
    for stem, matches in sorted(by_stem.items()):
        if len(matches) > 1:
            joined = ", ".join(match.as_posix() for match in matches)
            warnings.append(Finding(Path(stem), f"duplicate orchestrator stem appears at: {joined}"))
    return warnings


def main(argv: list[str] | None = None) -> int:
    """Run the orchestrator contract validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-warnings", action="store_true", help="treat soft warnings as failures")
    args = parser.parse_args(argv)

    root = repo_root()
    paths = orchestrator_files(root)
    errors: list[Finding] = []
    warnings: list[Finding] = duplicate_stem_warnings(paths, root)
    for path in paths:
        file_errors, file_warnings = validate_file(path, root)
        errors.extend(file_errors)
        warnings.extend(file_warnings)

    if warnings:
        print("Warnings:", file=sys.stderr)
        for finding in warnings:
            print(f"  {finding.path}: {finding.message}", file=sys.stderr)
    if errors:
        print("Errors:", file=sys.stderr)
        for finding in errors:
            print(f"  {finding.path}: {finding.message}", file=sys.stderr)
        return 1
    if args.strict_warnings and warnings:
        return 1
    print(f"OK: checked {len(paths)} orchestrator scripts ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
