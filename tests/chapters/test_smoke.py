"""Headless smoke tests for every chapter orchestrator script.

Each script is launched via ``subprocess`` with ``--save`` so we exercise the
exact CLI path a user would. ``MPLBACKEND=Agg`` keeps the GIF/PNG renderers
off any display.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]   # tests/chapters/ → repo root
CHAPTER_DIRS = {
    1: REPO_ROOT / "chapters" / "chapter_01",
    2: REPO_ROOT / "chapters" / "chapter_02",
    3: REPO_ROOT / "chapters" / "chapter_03",
}


def _is_interactive(p: Path) -> bool:
    """Skip GUI-only scripts in headless tests."""
    return "interactive" in p.name


def _run(script: Path, *extra: str, env_extra: dict | None = None,
         timeout: int = 120) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, str(script), "--save", *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(
            f"Script {script.name} failed.\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


CHAPTER_1_SCRIPTS = sorted(CHAPTER_DIRS[1].glob("0*.py"))
CHAPTER_2_EXAMPLES = sorted(p for p in CHAPTER_DIRS[2].glob("example_*.py")
                            if not _is_interactive(p))
CHAPTER_2_AUX = [CHAPTER_DIRS[2] / "visualize_generative_model.py"]
CHAPTER_2_ANIMATIONS = sorted(CHAPTER_DIRS[2].glob("animation_*.py"))
CHAPTER_3_EXAMPLES = sorted(CHAPTER_DIRS[3].glob("example_*.py"))
CHAPTER_3_ANIMATIONS = sorted(CHAPTER_DIRS[3].glob("animation_*.py"))
CHAPTER_3_VISUALIZATIONS = sorted(
    p for p in CHAPTER_DIRS[3].glob("visualize_*.py")
    if not _is_interactive(p)
)


@pytest.mark.parametrize("script", CHAPTER_1_SCRIPTS,
                         ids=[s.name for s in CHAPTER_1_SCRIPTS])
def test_chapter_1_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_EXAMPLES,
                         ids=[s.name for s in CHAPTER_2_EXAMPLES])
def test_chapter_2_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_AUX,
                         ids=[s.name for s in CHAPTER_2_AUX])
def test_chapter_2_auxiliary_scripts(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_2_ANIMATIONS])
def test_chapter_2_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_3_EXAMPLES,
                         ids=[s.name for s in CHAPTER_3_EXAMPLES])
def test_chapter_3_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_3_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_3_ANIMATIONS])
def test_chapter_3_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_3_VISUALIZATIONS,
                         ids=[s.name for s in CHAPTER_3_VISUALIZATIONS])
def test_chapter_3_visualizations(script: Path) -> None:
    # Diagnostic visualizations run a small Monte Carlo loop; allow some headroom.
    _run(script, timeout=240)
