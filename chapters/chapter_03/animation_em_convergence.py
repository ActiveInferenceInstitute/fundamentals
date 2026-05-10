"""Bonus — animated EM convergence for factor analysis.

Run::

    python chapters/chapter_03/animation_em_convergence.py [--save]

Renders a side-by-side animation of the marginal log-likelihood and the
loadings matrix Θ as EM iterations progress.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    diagonal_cov,
    fit_factor_analysis,
    get_logger,
    mvn_sample,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_em_convergence,
    save_animation,
)

LOG = get_logger("ch3.em_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-factors", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    true_Theta = np.array([
        [1.2, 0.4],
        [0.7, -0.2],
        [-0.3, 1.0],
        [0.5, 0.6],
        [0.1, 0.9],
    ])
    true_diag = np.array([0.10, 0.20, 0.05, 0.30, 0.15])
    n_obs = true_Theta.shape[0]

    X_latent = mvn_sample(np.zeros(args.n_factors),
                          np.eye(args.n_factors), n=args.n_samples, rng=rng)
    noise = mvn_sample(np.zeros(n_obs), diagonal_cov(true_diag),
                       n=args.n_samples, rng=rng)
    Y = X_latent @ true_Theta.T + noise

    result = fit_factor_analysis(
        Y, n_factors=args.n_factors,
        max_iter=args.max_iter, tol=0,  # force full iteration count for animation
        rng=np.random.default_rng(args.seed + 1),
    )
    LOG.info("Recorded %d EM iterations.", result.n_iterations)

    anim = animate_em_convergence(
        result.log_likelihoods,
        result.history["Theta"],
        title="Chapter 3 · EM convergence for factor analysis",
        interval_ms=130,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_em_convergence.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
