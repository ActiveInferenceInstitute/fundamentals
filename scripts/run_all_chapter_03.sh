#!/usr/bin/env bash
# Render every Chapter 3 figure to output/figures/chapter_03/.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_all_figures.py --chapters 3 "$@"
