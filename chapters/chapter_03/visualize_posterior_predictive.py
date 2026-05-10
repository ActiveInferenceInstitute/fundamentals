"""Diagnostic — posterior predictive check for the BLR fit.

Run::

    python chapters/chapter_03/visualize_posterior_predictive.py [--save]

After fitting BLR on real data, we draw ``M`` parameter samples from the
posterior, generate replicated data sets, compute several test statistics
(mean, std, range), and report the two-sided posterior-predictive p-values
along with the histograms of the replicated statistic vs the observed
value.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    BayesianLinearRegression,
    LinearGaussianProcess,
    get_logger,
    posterior_predictive_check,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_posterior_predictive_check,
    save_or_show,
)

LOG = get_logger("ch3.ppc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--m-replicates", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sigma2_y = 0.25
    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y, rng=rng,
    )
    x_pool = rng.uniform(0.0, 5.0, size=args.n)
    y_pool = np.array([process.sample(float(x), n=1)[0] for x in x_pool])

    blr = BayesianLinearRegression(
        prior_mean=np.zeros(2), prior_cov=np.eye(2) * 4.0,
        sigma2_y=sigma2_y,
    )
    posterior = blr.fit(x_pool, y_pool)
    LOG.info("Posterior θ mean = %s", np.round(posterior.mean, 3).tolist())

    # Draw parameter samples and simulate replicate data sets.
    theta_samples = posterior.sample(n=args.m_replicates, rng=rng)
    replicated = np.empty((args.m_replicates, args.n))
    for i, theta in enumerate(theta_samples):
        replicated[i] = (
            theta[0] + theta[1] * x_pool
            + rng.normal(scale=np.sqrt(sigma2_y), size=args.n)
        )

    statistics = {
        "mean": np.mean,
        "std": np.std,
        "range": lambda a: float(np.max(a) - np.min(a)),
    }

    figs = []
    for label, stat in statistics.items():
        check = posterior_predictive_check(y_pool, replicated, statistic=stat)
        LOG.info("PPC[%-5s] observed = %.4f, p = %.3f",
                 label, check.observed, check.p_value)
        fig = plot_posterior_predictive_check(
            check, label=f"replicated {label}",
            title=f"Posterior predictive check · statistic = {label}",
        )
        figs.append((label, fig))

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        for label, fig in figs:
            save_or_show(fig, out / f"diagnostic_ppc_{label}.png")
        LOG.info("Saved PPC figures to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
