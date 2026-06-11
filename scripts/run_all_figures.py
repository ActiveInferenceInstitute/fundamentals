"""Render every figure for all discovered chapters.

Run::

    python scripts/run_all_figures.py [--chapters 1 2 3 4 5] [--clean]

This is a convenience wrapper around the per-chapter orchestrators. It sets
``MPLBACKEND=Agg`` so it can be used in CI or on headless servers, then runs
every chapter script with ``--save``. Figures land in
``output/figures/chapter_<N>/`` and raw data lands in
``output/data/chapter_<N>/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.menu.runner import (  # noqa: E402
    discover_chapters,
    discover_extra_scripts,
    discover_extras,
    discover_scripts,
)
OUTPUT_DIR = REPO_ROOT / "output" / "figures"
DATA_DIR = REPO_ROOT / "output" / "data"
GENERATED_SUFFIXES = {".gif", ".mp4", ".pdf", ".png", ".svg", ".webm"}
GENERATED_DATA_SUFFIXES = {".csv", ".json", ".npz"}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    discovered_chapters = [entry.number for entry in discover_chapters()]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters", nargs="+", type=int,
                   default=discovered_chapters,
                   choices=discovered_chapters)
    p.add_argument("--extras", nargs="*", default=None,
                   help="Extras topic slugs to render; pass without values for all extras")
    p.add_argument("--no-chapters", action="store_true",
                   help="Skip chapter scripts; useful with --extras for extras-only renders")
    p.add_argument("--clean", action="store_true",
                   help="Delete existing generated figure media and raw data before running")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue on script failure")
    p.add_argument("--no-animations", action="store_true",
                   help="Skip animation_*.py scripts (faster)")
    return p.parse_args()


def clean_output_dir(root: Path = OUTPUT_DIR) -> int:
    """Remove generated media while preserving hand-maintained docs.

    The output tree contains README files that explain generated artifacts.
    A clean render should replace stale figures, not erase that documentation.
    """

    if not root.exists():
        return 0

    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def clean_data_dir(root: Path = DATA_DIR) -> int:
    """Remove generated raw-data sidecars while preserving README/AGENTS docs."""
    if not root.exists():
        return 0

    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in GENERATED_DATA_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def chapter_scripts(ch: int, *, include_animations: bool = True) -> list[Path]:
    """Return non-interactive scripts for one discovered chapter."""
    out = [entry.path for entry in discover_scripts(ch)]
    if not include_animations:
        out = [p for p in out if not p.name.startswith("animation_")]
    return out


def extra_scripts(topic: str, *, include_animations: bool = True) -> list[Path]:
    """Return non-interactive scripts for one extras topic."""
    entries = discover_extra_scripts(
        topic,
        include_animations=include_animations,
        include_visualizations=True,
    )
    return [entry.path for entry in entries]


def run(script: Path) -> int:
    """Support this repository command-line validation or rendering script."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "error"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(script), "--save"]
    print(f"  ▶ {script.relative_to(REPO_ROOT)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> int:
    """Run this repository command-line tool and return its exit status."""
    args = parse_args()
    if args.clean and OUTPUT_DIR.exists():
        removed = clean_output_dir(OUTPUT_DIR)
        print(f"Cleaned {removed} generated artifacts from {OUTPUT_DIR}")
    if args.clean and DATA_DIR.exists():
        removed = clean_data_dir(DATA_DIR)
        print(f"Cleaned {removed} generated data files from {DATA_DIR}")

    failed: list[Path] = []
    if not args.no_chapters:
        for ch in args.chapters:
            print(f"\n=== Chapter {ch} ===")
            for s in chapter_scripts(ch, include_animations=not args.no_animations):
                rc = run(s)
                if rc != 0:
                    failed.append(s)
                    if not args.keep_going:
                        print(f"X {s.name} failed; aborting (use --keep-going to continue).")
                        return rc

    if args.extras is not None:
        all_topics = [entry.slug for entry in discover_extras()]
        requested = all_topics if args.extras == [] else list(args.extras)
        unknown = sorted(set(requested) - set(all_topics))
        if unknown:
            print(f"Unknown extras topics: {', '.join(unknown)}", file=sys.stderr)
            return 2
        for topic in requested:
            print(f"\n=== Extra: {topic} ===")
            for s in extra_scripts(topic, include_animations=not args.no_animations):
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
