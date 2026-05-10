"""Example 2.4 — Nonlinear, deterministic generative model.

Run::

    python chapters/chapter_02/example_2_4_nonlinear_deterministic.py [--save]

Quadratic generator on a symmetric domain. With a uniform prior the posterior
is bi-modal; this is exactly the situation that motivates hierarchical priors
later in the book.
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
from active_inference.visualizations import (
    plot_generating_function,
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch2.ex4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=11.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = make_grid(-2.5, 2.5, 600)

    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=1e-3,
        prior_kind="uniform", uniform_low=-2.5, uniform_high=2.5,
        psi=lambda x: x ** 2,
    )
    res = GridBayesianInference(model, x_grid).infer(args.y_obs)

    expected = np.sqrt((args.y_obs - 3) / 2)
    LOG.info("Analytic ±sqrt((y-3)/2) = ±%.4f", expected)

    fig_curve = plot_generating_function(
        x_grid, model.predict_mean(x_grid),
        title="Example 2.4 · g(x) = 3 + 2 x^2",
    )
    fig_post = plot_prior_likelihood_posterior(
        res, title=f"Example 2.4 · y = {args.y_obs} (bi-modal posterior)",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_curve, out / "example_2_4_curve.png")
        save_or_show(fig_post, out / "example_2_4_posterior.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
