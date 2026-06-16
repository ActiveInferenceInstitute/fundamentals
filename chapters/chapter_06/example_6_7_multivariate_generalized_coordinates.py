"""Example 6.7 - multivariate generalized filtering in generalized coordinates."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger, save_chapter_data
from active_inference.orchestrator_workflows import build_example_6_7_multivariate_generalized_coordinates
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch6.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Example 6.7."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Run Example 6.7 and render or save the educational figure."""
    args = parse_args()
    result = build_example_6_7_multivariate_generalized_coordinates(
        seed=args.seed,
        n_steps=args.n_steps,
        dt=args.dt,
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
    )
    LOG.info(
        "Example 6.7 tracking error: ordinary=%.4f | generalized=%.4f",
        result.summary["ordinary_error"],
        result.summary["generalized_error"],
    )

    fig = result.figures["example_6_7_multivariate_generalized_coordinates"]
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        figure = out / "example_6_7_multivariate_generalized_coordinates.png"
        save_or_show(fig, figure)
        save_chapter_data(6, figure.stem, result.arrays, result.metadata, figures=[figure])
        LOG.info("Saved Example 6.7 to %s", figure)
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
