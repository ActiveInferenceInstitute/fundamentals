"""Example 3.3 — Multiple linear regression via the normal equation.

Run::

    python chapters/chapter_03/example_3_3_multiple_regression.py [--save]

Builds a synthetic ``y = β₀ + Σ βᵢ xᵢ + ε`` dataset with ``C`` predictors,
solves for ``θ`` in closed form, and compares the recovered coefficients to
the truth as a function of N.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import get_logger, mle_linear_regression
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-features", type=int, default=4)
    p.add_argument("--max-n", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    true_theta = rng.normal(size=args.n_features + 1)
    LOG.info("True θ: %s", np.round(true_theta, 3))

    sample_sizes = np.unique(np.geomspace(args.n_features + 5,
                                          args.max_n, 12).astype(int))
    errors = []
    last_estimate = None
    for n in sample_sizes:
        X = rng.normal(size=(n, args.n_features))
        noise = rng.normal(scale=0.5, size=n)
        y = true_theta[0] + X @ true_theta[1:] + noise
        theta_hat = mle_linear_regression(X, y)
        errors.append(np.linalg.norm(theta_hat - true_theta))
        last_estimate = theta_hat

    LOG.info("Final estimate at N=%d: %s",
             sample_sizes[-1], np.round(last_estimate, 3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].loglog(sample_sizes, errors, "o-", color="#1f77b4", lw=2)
    axes[0].set_xlabel("number of samples N")
    axes[0].set_ylabel(r"$\|\hat{\theta} - \theta^*\|_2$")
    axes[0].set_title("Estimation error decays with N")
    axes[0].grid(alpha=0.3, which="both")

    width = 0.35
    idx = np.arange(true_theta.size)
    axes[1].bar(idx - width / 2, true_theta, width=width,
                color="#1f77b4", label="true")
    axes[1].bar(idx + width / 2, last_estimate, width=width,
                color="#2ca02c", label=f"estimate (N={sample_sizes[-1]})")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([r"$\beta_0$"] +
                             [rf"$\beta_{{{i + 1}}}$"
                              for i in range(args.n_features)])
    axes[1].set_title("Coefficient recovery")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_3_multiple_regression.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
