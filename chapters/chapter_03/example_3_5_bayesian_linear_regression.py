"""Example 3.5 - Bayesian linear regression."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger
from active_inference.orchestrator_workflows import build_example_3_5_bayesian_linear_regression
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex5")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-n", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    result = build_example_3_5_bayesian_linear_regression(seed=args.seed, max_n=args.max_n)
    LOG.info(
        "Final posterior mean = %s, std = %s",
        np.round(result.summary["posterior_mean"], 3),
        np.round(result.summary["posterior_std"], 3),
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        for stem, fig in result.figures.items():
            save_or_show(fig, out / f"{stem}.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
