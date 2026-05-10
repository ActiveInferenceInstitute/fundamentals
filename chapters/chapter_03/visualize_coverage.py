"""Diagnostic — empirical coverage of a 95% credible interval as N grows.

Run::

    python chapters/chapter_03/visualize_coverage.py [--save] [--seed N]

For each sample size in a sweep, repeats the LGS sensor-fusion experiment
many times and records what fraction of the trials' 95% Mahalanobis
credible regions actually contained the truth. The coverage curve should
hover around 0.95 once N is large enough — otherwise the model is mis-
calibrated.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianMVProcess,
    LinearGaussianSystem,
    get_logger,
    isotropic_cov,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_coverage_curve, save_or_show

LOG = get_logger("ch3.coverage")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sample_sizes = np.array([3, 5, 10, 25, 50, 100, 200])
    cov_y = 0.04 * np.eye(2)
    Theta = np.eye(2)
    chi2_95 = float(-2.0 * np.log(0.05))   # χ²₂ at 0.95 ≈ 5.991

    coverages = np.empty(sample_sizes.size, dtype=float)
    for i, n in enumerate(sample_sizes):
        contains = np.empty(args.n_trials, dtype=bool)
        for t in range(args.n_trials):
            x_star = rng.uniform(0.0, 1.0, size=2)
            proc = LinearGaussianMVProcess(
                Theta=Theta, cov_y=cov_y,
                rng=np.random.default_rng(args.seed + 1000 * (i + 1) + t),
            )
            Y = proc.sample(x_star, n=int(n)).reshape(int(n), 2)
            lgs = LinearGaussianSystem(
                Theta=Theta, cov_y=cov_y,
                mx=np.array([0.5, 0.5]), cov_x=isotropic_cov(2, 4.0),
            )
            post = lgs.posterior_batch(Y)
            diff = x_star - post.mean
            mahal = float(diff @ np.linalg.inv(post.cov) @ diff)
            contains[t] = mahal <= chi2_95
        coverages[i] = float(np.mean(contains))
        LOG.info("N = %3d → empirical coverage = %.3f", n, coverages[i])

    fig = plot_coverage_curve(
        sample_sizes, coverages,
        nominal=0.95,
        title=f"95% credible region coverage · LGS sensor fusion · {args.n_trials} trials each",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "diagnostic_coverage.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
