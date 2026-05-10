"""Example 2.5 — Nonlinear, probabilistic generative model.

Run::

    python chapters/chapter_02/example_2_5_nonlinear_probabilistic.py [--save]

Quadratic generator with Gaussian noise. A Gaussian state prior centered near
the positive root resolves the inverse-problem ambiguity from Example 2.4.
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
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_generating_function,
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch2.ex5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--x-true", type=float, default=np.sqrt(2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    psi = lambda x: x ** 2
    process = LinearGaussianProcess(
        beta0=1.0, beta1=1.0, sigma2_y=0.25, rng=rng, nonlinear=psi,
    )
    y_obs = float(process.sample(args.x_true, n=1)[0])
    LOG.info("y = %.4f sampled at x* = %.4f", y_obs, args.x_true)

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=1.0, beta1=1.0, sigma2_y=0.25,
        m_x=2.0, s2_x=0.25, prior_kind="gaussian",
        psi=psi,
    )
    res = GridBayesianInference(model, x_grid).infer(y_obs)
    LOG.info("Posterior mode = %.4f, true x* = %.4f", res.posterior_mode, args.x_true)

    fig_curve = plot_generating_function(
        x_grid, model.predict_mean(x_grid),
        samples_x=np.array([args.x_true]),
        samples_y=np.array([y_obs]),
        title="Example 2.5 · g(x) = 1 + x^2 with noise",
    )
    fig_post = plot_prior_likelihood_posterior(
        res, title=f"Example 2.5 · y = {y_obs:.3f}", truth=args.x_true,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_curve, out / "example_2_5_curve.png")
        save_or_show(fig_post, out / "example_2_5_posterior.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
