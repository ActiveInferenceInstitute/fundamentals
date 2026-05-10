"""Example 2.3 — Trust the data or trust the prior?

Run::

    python chapters/chapter_02/example_2_3_precision.py [--save]

Sweeping the prior variance and the likelihood variance shows how the
posterior smoothly shifts between "stubborn prior" and "credulous data".
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_precision_comparison, save_or_show

LOG = get_logger("ch2.ex3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=7.0)
    return p.parse_args()


def configurations() -> list[tuple[str, float, float]]:
    """Three labelled (sigma2_y, s2_x) settings to overlay."""
    return [
        ("balanced (s2_x = sigma2_y = 0.25)", 0.25, 0.25),
        ("trust prior  (s2_x = 0.1, sigma2_y = 1.5)", 1.5, 0.1),
        ("trust data   (s2_x = 1.5, sigma2_y = 0.1)", 0.1, 1.5),
    ]


def main() -> None:
    args = parse_args()
    x_grid = make_grid(0.0, 5.0, 500)

    results = []
    for label, sigma2_y, s2_x in configurations():
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
            m_x=4.0, s2_x=s2_x, prior_kind="gaussian",
        )
        res = GridBayesianInference(model, x_grid).infer(args.y_obs)
        LOG.info("%-44s mode=%.3f", label, res.posterior_mode)
        results.append((label, res))

    fig = plot_precision_comparison(
        results, title=f"Example 2.3 · precision sweep at y = {args.y_obs}",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig, out / "example_2_3_precision.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
