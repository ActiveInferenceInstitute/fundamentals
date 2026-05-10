"""Example 2.1 — Linear, deterministic generative model.

Run::

    python chapters/chapter_02/example_2_1_linear_deterministic.py [--save]

In the deterministic limit the likelihood collapses to a Dirac delta and
Bayesian inference reduces to inverting the generating function. We use a
narrow Gaussian as a proxy for the delta and a uniform prior so the posterior
mode is exactly ``g^{-1}(y)``.
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
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch2.ex1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=10.0)
    p.add_argument("--epsilon2", type=float, default=1e-4,
                   help="Variance of the dirac-like likelihood proxy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = make_grid(0.0, 5.0, 500)

    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=args.epsilon2,
        prior_kind="uniform", uniform_low=0.0, uniform_high=5.0,
    )
    res = GridBayesianInference(model, x_grid).infer(args.y_obs)

    expected = (args.y_obs - 3.0) / 2.0
    LOG.info("Analytic g^{-1}(%.2f) = %.4f, posterior mode = %.4f",
             args.y_obs, expected, res.posterior_mode)

    fig = plot_prior_likelihood_posterior(
        res, title=f"Example 2.1 · deterministic Bayesian inversion (y = {args.y_obs})",
        truth=expected,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig, out / "example_2_1_linear_deterministic.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
