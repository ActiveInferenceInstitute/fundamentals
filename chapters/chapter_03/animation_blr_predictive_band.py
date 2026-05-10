"""Bonus animation — Bayesian linear regression predictive band collapse.

Run::

    python chapters/chapter_03/animation_blr_predictive_band.py [--save]

The 95 % predictive envelope around the BLR mean line shrinks toward
the true regression line as new ``(x, y)`` pairs are assimilated.
This is the visual companion to ``animation_blr_tightening.py``: rather
than tracking the parameter posterior, this animation tracks the
*observable* predictive uncertainty.
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
    animate_blr_predictive_band,
    save_animation,
)

LOG = get_logger("ch3.blr_band_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sigma2_y = 1.0
    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y, rng=rng,
    )
    x_data = rng.uniform(0.0, 5.0, size=args.n)
    y_data = np.array([process.sample(float(x), n=1)[0] for x in x_data])

    blr = BayesianLinearRegression(
        prior_mean=np.zeros(2), prior_cov=np.eye(2) * 4.0,
        sigma2_y=sigma2_y,
    )
    means = np.empty((args.n, 2))
    covs = np.empty((args.n, 2, 2))
    for i, post in blr.fit_sequential(x_data, y_data):
        means[i - 1] = post.mean
        covs[i - 1] = post.cov

    LOG.info("Final posterior mean = %s, std = %s",
             np.round(means[-1], 3), np.round(np.sqrt(np.diag(covs[-1])), 3))

    x_grid = np.linspace(0.0, 5.0, 200)
    anim = animate_blr_predictive_band(
        x_grid, means, covs, sigma2_y,
        x_data=x_data, y_data=y_data,
        truth_line=(3.0, 2.0),
        intercept=True,
        title="Chapter 3 · BLR predictive band collapsing onto the truth",
        interval_ms=110,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_blr_predictive_band.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
