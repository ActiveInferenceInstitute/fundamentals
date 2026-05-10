"""Bonus animation — alternating E and M steps in factor-analysis EM.

Run::

    python chapters/chapter_03/animation_em_steps.py [--save]

Renders three synchronized panels:

- **E-step:** scatter of the latent posterior means (factor-1 vs factor-2).
- **M-step:** heat-map of the loadings matrix Θ being re-estimated.
- **Bottom inset:** marginal log p(Y) — must be monotone non-decreasing.

The point of the animation is to make the E-step → M-step alternation
*visible* (not just a converged endpoint) and to show the monotonicity
of the marginal log-likelihood.
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
from active_inference.estimators.em import (
    factor_analysis_e_step,
    incomplete_log_likelihood,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_em_steps, save_animation

LOG = get_logger("ch3.em_steps_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--max-iter", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    true_Theta = np.array([
        [1.2, 0.4], [0.7, -0.2], [-0.3, 1.0], [0.5, 0.6], [0.1, 0.9],
    ])
    true_diag = np.array([0.10, 0.20, 0.05, 0.30, 0.15])

    X_latent = mvn_sample(np.zeros(2), np.eye(2), n=args.n_samples, rng=rng)
    noise = mvn_sample(np.zeros(5), diagonal_cov(true_diag),
                       n=args.n_samples, rng=rng)
    Y = X_latent @ true_Theta.T + noise

    result = fit_factor_analysis(
        Y, n_factors=2, max_iter=args.max_iter, tol=0,
        rng=np.random.default_rng(args.seed + 1),
    )
    LOG.info("Recorded %d EM iterations.", result.n_iterations)

    # Recompute per-iteration E-step posterior means using the per-iter Theta
    # and diag(cov_y) trajectory, so the animation actually shows the E-step
    # updating in lockstep with the M-step.
    e_step_means = []
    log_likelihoods = []
    for theta, diag in zip(result.history["Theta"], result.history["cov_y_diag"]):
        cov_y = np.diag(diag)
        mu, _ = factor_analysis_e_step(Y, theta, cov_y)
        e_step_means.append(mu)
        log_likelihoods.append(incomplete_log_likelihood(Y, theta, cov_y))

    anim = animate_em_steps(
        e_step_means,
        result.history["Theta"],
        np.asarray(log_likelihoods),
        title="Chapter 3 · EM · alternating E and M steps",
        interval_ms=220,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_em_steps.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
