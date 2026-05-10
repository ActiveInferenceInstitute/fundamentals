"""Bonus — Slider-driven explorer for the canonical linear-Gaussian agent.

Run::

    python chapters/chapter_02/interactive_explorer.py

This script opens a matplotlib window with sliders for the observation, prior
mean, prior variance, and likelihood variance. Drag any slider and watch the
prior, likelihood, and posterior update in real time.

A second window exposes a single ``log10(s2_x / sigma2_y)`` slider that
sweeps from "prior dominates" to "data dominates" — useful for building
intuition about precision-weighted inference.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import (
    interactive_inference,
    interactive_precision,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("full", "precision"), default="full")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        interactive_inference()
    else:
        interactive_precision()
    plt.show()


if __name__ == "__main__":
    main()
