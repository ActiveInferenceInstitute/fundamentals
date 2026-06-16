"""Animation - Chapter 7 section 7.5 multivariate active generalized filtering."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger, save_chapter_data
from active_inference.orchestrator_workflows import build_animation_7_5_multivariate_active_inference
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_animation

LOG = get_logger("ch7.anim5")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Chapter 7 section 7.5 animation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Create and optionally save the multivariate active-inference GIF."""
    args = parse_args()
    result = build_animation_7_5_multivariate_active_inference(
        seed=args.seed,
        n_steps=args.n_steps,
        dt=args.dt,
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
    )
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_07")
        figure = out / "animation_7_5_multivariate_active_inference.gif"
        save_animation(result.animation, figure, fps=12, dpi=110)
        save_chapter_data(7, figure.stem, result.arrays, result.metadata, figures=[figure])
        LOG.info("Saved Chapter 7 section 7.5 animation to %s", figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
