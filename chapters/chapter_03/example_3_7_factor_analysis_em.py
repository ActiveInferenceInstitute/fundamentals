"""Example 3.7 - Factor analysis via expectation-maximization."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger
from active_inference.orchestrator_workflows import build_example_3_7_factor_analysis_em
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-factors", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    result = build_example_3_7_factor_analysis_em(
        seed=args.seed,
        n_samples=args.n_samples,
        n_factors=args.n_factors,
        max_iter=args.max_iter,
    )
    LOG.info("Converged=%s after %d iterations.", result.summary["converged"], result.summary["n_iterations"])
    LOG.info("Final marginal LL = %.3f", result.summary["final_log_likelihood"])
    LOG.info("Reconstruction RMSE = %.4f", result.summary["rmse"])

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        for stem, fig in result.figures.items():
            save_or_show(fig, out / f"{stem}.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
