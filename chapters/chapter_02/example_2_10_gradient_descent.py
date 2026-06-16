"""Section 2.5.2 - Iterative MLE / MAP via gradient descent."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger
from active_inference.orchestrator_workflows import build_example_2_10_gradient_descent
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch2.ex10")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--x-true", type=float, default=2.5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Step size; require lr * Hessian < 2 for convergence.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    result = build_example_2_10_gradient_descent(
        seed=args.seed,
        n_samples=args.n_samples,
        x_true=args.x_true,
        lr=args.lr,
        max_iter=args.max_iter,
    )
    LOG.info("Sample mean = %.3f, std = %.3f", result.summary["sample_mean"], result.summary["sample_std"])
    LOG.info(
        "MLE closed-form = %.4f, gradient descent = %.4f, true x* = %.3f",
        result.summary["mle_closed"],
        result.summary["mle_iter"],
        args.x_true,
    )
    LOG.info(
        "MAP closed-form = %.4f, gradient descent = %.4f",
        result.summary["map_closed"],
        result.summary["map_iter"],
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        for stem, fig in result.figures.items():
            save_or_show(fig, out / f"{stem}.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
