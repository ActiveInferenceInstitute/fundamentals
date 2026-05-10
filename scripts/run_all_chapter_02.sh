#!/usr/bin/env bash
# Render every Chapter 2 figure to output/figures/chapter_02/.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_all_figures.py --chapters 2 "$@"
