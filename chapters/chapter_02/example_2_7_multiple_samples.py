"""Example 2.7 - Bayesian inference from a batch of i.i.d. observations."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger
from active_inference.orchestrator_workflows import build_example_2_7_multiple_samples
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch2.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--x-true", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    result = build_example_2_7_multiple_samples(
        seed=args.seed,
        n_samples=args.n_samples,
        x_true=args.x_true,
    )
    LOG.info(
        "Sample mean = %.3f, std = %.3f (true mean = %.3f)",
        result.summary["sample_mean"],
        result.summary["sample_std"],
        result.summary["process_mean"],
    )
    LOG.info(
        "Batch posterior mode = %.4f, var = %.4g",
        result.summary["batch_mode"],
        result.summary["batch_variance"],
    )
    LOG.info("Sequential final mode = %.4f", result.summary["sequential_mode"])

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        for stem, fig in result.figures.items():
            save_or_show(fig, out / f"{stem}.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
