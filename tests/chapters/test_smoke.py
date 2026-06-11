"""Headless smoke tests for every chapter orchestrator script.

Each script is launched via ``subprocess`` with ``--save`` so we exercise the
exact CLI path a user would. ``MPLBACKEND=Agg`` keeps the GIF/PNG renderers
off any display.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from active_inference.menu.runner import discover_chapters, discover_scripts


REPO_ROOT = Path(__file__).resolve().parents[2]   # tests/chapters/ -> repo root


def _is_interactive(p: Path) -> bool:
    """Skip GUI-only scripts in headless tests."""
    return "interactive" in p.name


def _chapter_number(script: Path) -> int:
    """Infer the numeric chapter directory that owns ``script``."""
    return int(script.parent.name.removeprefix("chapter_"))


def _fresh_data_pairs(data_dir: Path, started_ns: int) -> list[tuple[Path, Path]]:
    """Return JSON/NPZ sidecars written after ``started_ns``."""
    pairs: list[tuple[Path, Path]] = []
    for json_path in data_dir.glob("*.json"):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        if json_path.stat().st_mtime_ns >= started_ns or npz_path.stat().st_mtime_ns >= started_ns:
            pairs.append((npz_path, json_path))
    return pairs


def _run(
    script: Path,
    *extra: str,
    output_root: Path,
    env_extra: dict | None = None,
    timeout: int = 120,
) -> None:
    """Run one chapter script and require fresh figure-adjacent raw-data output."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "error"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ACTIVE_INFERENCE_OUTPUT_ROOT"] = str(output_root)
    if env_extra:
        env.update(env_extra)
    chapter = _chapter_number(script)
    data_dir = output_root / "data" / f"chapter_{chapter:02d}"
    started_ns = time.time_ns()
    cmd = [sys.executable, str(script), "--save", *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=timeout, cwd=output_root.parent)
    if result.returncode != 0:
        raise AssertionError(
            f"Script {script.name} failed.\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    pairs = _fresh_data_pairs(data_dir, started_ns)
    assert pairs, f"Script {script.name} did not write a fresh NPZ+JSON data export"


def _all_chapter_scripts() -> list[Path]:
    """Return every non-interactive chapter script discovered by the menu layer."""
    scripts: list[Path] = []
    for chapter in discover_chapters():
        scripts.extend(entry.path for entry in discover_scripts(chapter.number))
    return scripts


CHAPTER_ALL_SCRIPTS = _all_chapter_scripts()


def _timeout(script: Path) -> int:
    """Return a smoke-test timeout appropriate for one script kind."""
    if script.name.startswith("animation_") or script.name.startswith("visualize_"):
        return 240
    if script.parent.name in {"chapter_04", "chapter_05", "chapter_06", "chapter_07",
                              "chapter_08", "chapter_09", "chapter_10"}:
        return 240
    return 120


@pytest.mark.parametrize(
    "script",
    CHAPTER_ALL_SCRIPTS,
    ids=[str(s.relative_to(REPO_ROOT)) for s in CHAPTER_ALL_SCRIPTS],
)
def test_chapter_script_runs_and_exports_raw_data(script: Path, tmp_path: Path) -> None:
    """Every discovered chapter script runs headlessly and writes fresh raw-data sidecars."""
    _run(script, output_root=tmp_path / "output", timeout=_timeout(script))
