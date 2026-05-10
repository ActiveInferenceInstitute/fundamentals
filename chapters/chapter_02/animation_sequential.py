"""Bonus — animated sequential Bayesian update for Chapter 2.

Run::

    python chapters/chapter_02/animation_sequential.py [--save] [--n 50]

Watches the posterior tighten frame-by-frame as i.i.d. observations arrive.
A non-trivial demonstration of the building blocks from Examples 2.2 / 2.7.
The rendered animation is saved as a GIF when ``--save`` is passed.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianModel,
    LinearGaussianProcess,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_sequential_posterior,
    save_animation,
)

LOG = get_logger("ch2.anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--x-true", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.4, rng=rng)
    samples = process.sample(args.x_true, n=args.n).flatten()

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.4,
        m_x=4.0, s2_x=1.0, prior_kind="gaussian",
    )

    log_state = model.log_prior(x_grid).copy()
    prior_density = np.exp(log_state - np.max(log_state))
    prior_density /= np.trapezoid(prior_density, x_grid)

    posteriors = []
    for y in samples:
        log_state = log_state + model.log_likelihood(float(y), x_grid)
        density = np.exp(log_state - np.max(log_state))
        density /= np.trapezoid(density, x_grid)
        posteriors.append(density)

    LOG.info("Final posterior mode after %d samples: %.4f",
             args.n, x_grid[int(np.argmax(posteriors[-1]))])

    anim = animate_sequential_posterior(
        x_grid, posteriors,
        truth=args.x_true, prior=prior_density,
        title="Chapter 2 · sequential posterior update",
        interval_ms=80,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        path = save_animation(anim, out / "animation_sequential.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
