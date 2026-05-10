"""Diagnostic animation — running posterior mean / std / KL[post||prior] vs N.

Run::

    python chapters/chapter_03/animate_sufficient_stats.py [--save]

This animation shows what is *actually* changing as observations arrive: the
posterior mean walking toward the truth, the posterior std contracting, and
the KL from the prior climbing as the data dominates. It complements the
shape-evolution animation in ``animation_sequential.py`` by surfacing
numerical statistics rather than just the density curves.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianModel,
    LinearGaussianProcess,
    get_logger,
    grid_kl_divergence,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_sufficient_statistics,
    save_animation,
)

LOG = get_logger("ch3.sufficient_stats")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--x-true", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=0.4, rng=rng,
    )
    samples = process.sample(args.x_true, n=args.n).flatten()

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.4,
        m_x=4.0, s2_x=1.0, prior_kind="gaussian",
    )

    log_prior = model.log_prior(x_grid).copy()
    prior_density = np.exp(log_prior - np.max(log_prior))
    prior_density /= np.trapezoid(prior_density, x_grid)

    log_state = log_prior.copy()
    means = np.empty(args.n)
    stds = np.empty(args.n)
    kls = np.empty(args.n)
    for i, y in enumerate(samples):
        log_state = log_state + model.log_likelihood(float(y), x_grid)
        density = np.exp(log_state - np.max(log_state))
        density /= np.trapezoid(density, x_grid)
        m = float(np.trapezoid(x_grid * density, x_grid))
        v = float(np.trapezoid((x_grid - m) ** 2 * density, x_grid))
        means[i] = m
        stds[i] = float(np.sqrt(max(v, 0.0)))
        kls[i] = grid_kl_divergence(density, prior_density, x_grid)

    LOG.info(
        "Final mean = %.4f, std = %.4f, KL = %.4f (true x* = %.3f)",
        means[-1], stds[-1], kls[-1], args.x_true,
    )

    anim = animate_sufficient_statistics(
        np.arange(1, args.n + 1),
        means, stds, kls,
        truth=args.x_true,
        title="Chapter 3 · Running posterior statistics",
        interval_ms=70,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_sufficient_statistics.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
