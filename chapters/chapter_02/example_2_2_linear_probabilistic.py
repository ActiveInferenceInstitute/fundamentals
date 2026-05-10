"""Example 2.2 — Linear, probabilistic generative model.

Run::

    python chapters/chapter_02/example_2_2_linear_probabilistic.py [--save]

The standard textbook setup: Gaussian likelihood, Gaussian prior, single
observation. The posterior is Gaussian with a mean that lies between the
prior mean and the analytic inverse of ``y``.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    LinearGaussianProcess,
    get_logger,
    make_grid,
    map_analytic_linear,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_generating_function,
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch2.ex2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--x-true", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng)
    y_obs = float(process.sample(args.x_true, n=1)[0])
    LOG.info("Sampled observation y = %.4f from x* = %.3f", y_obs, args.x_true)

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.25,
        m_x=4.0, s2_x=0.25, prior_kind="gaussian",
    )
    res = GridBayesianInference(model, x_grid).infer(y_obs)

    map_estimate = map_analytic_linear(
        np.array([y_obs]), beta0=3.0, beta1=2.0,
        sigma2_y=0.25, m_x=4.0, s2_x=0.25,
    )
    LOG.info("Posterior mode (grid) = %.4f, analytic MAP = %.4f",
             res.posterior_mode, map_estimate)
    LOG.info("95%% credible interval = (%.3f, %.3f)",
             *res.credible_interval(0.95))

    fig_curve = plot_generating_function(
        x_grid, model.predict_mean(x_grid),
        samples_x=np.array([args.x_true]),
        samples_y=np.array([y_obs]),
        title="Example 2.2 · noisy linear sensor",
    )
    fig_post = plot_prior_likelihood_posterior(
        res, title=f"Example 2.2 · y = {y_obs:.3f}",
        truth=args.x_true,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_curve, out / "example_2_2_curve.png")
        save_or_show(fig_post, out / "example_2_2_posterior.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
