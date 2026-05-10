"""Bonus — animated Bayesian linear regression posterior tightening.

Run::

    python chapters/chapter_03/animation_blr_tightening.py [--save] [--n 60]

Each frame represents one additional observation. The 1- and 2-σ ellipses
of the posterior over (β₀, β₁) shrink onto the true coefficients as N grows.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    BayesianLinearRegression,
    LinearGaussianProcess,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_2d_posterior,
    save_animation,
)

LOG = get_logger("ch3.blr_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)
    x_pool = rng.uniform(0, 5, size=args.n)
    y_pool = np.array([process.sample(float(x), n=1)[0] for x in x_pool])

    blr = BayesianLinearRegression(
        prior_mean=np.zeros(2),
        prior_cov=np.eye(2) * 4.0,
        sigma2_y=1.0,
    )

    means = np.empty((args.n, 2))
    covs = np.empty((args.n, 2, 2))
    for i, post in blr.fit_sequential(x_pool, y_pool):
        means[i - 1] = post.mean
        covs[i - 1] = post.cov

    LOG.info("Final posterior mean = %s, std = %s",
             np.round(means[-1], 3), np.round(np.sqrt(np.diag(covs[-1])), 3))

    anim = animate_2d_posterior(
        means, covs,
        truth=np.array([3.0, 2.0]),
        prior_mean=blr.prior_mean,
        prior_cov=blr.prior_cov,
        xlim=(-1, 5), ylim=(-1, 5),
        labels=(r"$\beta_0$", r"$\beta_1$"),
        title="Chapter 3 · Bayesian linear regression posterior tightening",
        interval_ms=120,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_blr_tightening.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
