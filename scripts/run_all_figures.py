"""Render every figure for Chapters 1 and 2.

Run::

    python scripts/run_all_figures.py [--chapters 1 2] [--clean]

This is a convenience wrapper around the per-chapter orchestrators. It sets
``MPLBACKEND=Agg`` so it can be used in CI or on headless servers, then runs
every chapter script with ``--save``. Figures land in
``output/figures/chapter_<N>/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIRS = {
    1: REPO_ROOT / "chapters" / "chapter_01",
    2: REPO_ROOT / "chapters" / "chapter_02",
    3: REPO_ROOT / "chapters" / "chapter_03",
}
OUTPUT_DIR = REPO_ROOT / "output" / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters", nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3])
    p.add_argument("--clean", action="store_true",
                   help="Delete existing output/figures before running")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue on script failure")
    p.add_argument("--no-animations", action="store_true",
                   help="Skip animation_*.py scripts (faster)")
    return p.parse_args()


def chapter_scripts(ch: int, *, include_animations: bool = True) -> list[Path]:
    if ch not in CHAPTER_DIRS:
        raise ValueError(ch)
    base = CHAPTER_DIRS[ch]
    if ch == 1:
        return sorted(base.glob("0*.py"))
    out = sorted(p for p in base.glob("*.py")
                 if "interactive" not in p.name)
    if not include_animations:
        out = [p for p in out if not p.name.startswith("animation_")]
    return out


def run(script: Path) -> int:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(script), "--save"]
    print(f"  ▶ {script.relative_to(REPO_ROOT)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> int:
    args = parse_args()
    if args.clean and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned {OUTPUT_DIR}")

    failed: list[Path] = []
    for ch in args.chapters:
        print(f"\n=== Chapter {ch} ===")
        for s in chapter_scripts(ch, include_animations=not args.no_animations):
            rc = run(s)
            if rc != 0:
                failed.append(s)
                if not args.keep_going:
                    print(f"X {s.name} failed; aborting (use --keep-going to continue).")
                    return rc

    if failed:
        print("\nFailed scripts:")
        for s in failed:
            print(f"  - {s}")
        return 1
    print("\nAll figures rendered successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
