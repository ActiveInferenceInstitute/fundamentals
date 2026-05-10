#!/usr/bin/env bash
# Render every Chapter 1 figure to output/figures/chapter_01/.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_all_figures.py --chapters 1 "$@"
