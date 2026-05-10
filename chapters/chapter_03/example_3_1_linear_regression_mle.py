"""Example 3.1 — Closed-form linear regression at low sample sizes.

Run::

    python chapters/chapter_03/example_3_1_linear_regression_mle.py [--save]

Demonstrates that the analytic ``θ = (XᵀX)⁻¹ Xᵀy`` is correct on average but
that individual fits at small N can wander widely. Repeats the regression
``--n-trials`` times with fresh draws and plots all candidate fits, then
overlays the cloud of estimates against the true (β₀, β₁).
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    LinearGaussianProcess,
    get_logger,
    mle_linear_regression,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-trial", type=int, default=8)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--x-true", type=float, default=2.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)

    # x range from which inputs are drawn; we want the agent to see *both*
    # endpoints of the line, otherwise extrapolation dominates.
    x_pool = np.linspace(0.0, 5.0, 200)

    estimates = np.empty((args.n_trials, 2))
    inferred = np.empty(args.n_trials)
    for trial in range(args.n_trials):
        x_trial = rng.choice(x_pool, size=args.n_per_trial, replace=False)
        y_trial = np.array([process.sample(float(x), n=1)[0] for x in x_trial])
        theta = mle_linear_regression(x_trial, y_trial)
        estimates[trial] = theta
        # Hidden-state estimate using the trial's parameters: x = (mean(y) - b0) / b1
        ys_for_inference = process.sample(args.x_true, n=8).flatten()
        inferred[trial] = (ys_for_inference.mean() - theta[0]) / theta[1]

    LOG.info("Mean estimate: β0=%.3f, β1=%.3f (true 3.0, 2.0)",
             estimates[:, 0].mean(), estimates[:, 1].mean())
    LOG.info("Std  estimate: β0=%.3f, β1=%.3f",
             estimates[:, 0].std(), estimates[:, 1].std())
    LOG.info("Inferred x mean ± std: %.3f ± %.3f (true %.3f)",
             inferred.mean(), inferred.std(), args.x_true)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    grid = np.linspace(0.0, 5.0, 200)
    for theta in estimates[:30]:  # only plot a subset of fits
        axes[0].plot(grid, theta[0] + theta[1] * grid, color="#2ca02c",
                     lw=0.7, alpha=0.5)
    axes[0].plot(grid, 3.0 + 2.0 * grid, color="red", lw=2, label="true line")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"30 of {args.n_trials} regression fits at N = {args.n_per_trial}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    sc = axes[1].scatter(estimates[:, 0], estimates[:, 1],
                         c=np.abs(inferred - args.x_true),
                         cmap="viridis", s=18)
    axes[1].scatter(3.0, 2.0, marker="x", color="red", s=120, lw=2,
                    label="true (3, 2)")
    axes[1].set_xlabel(r"$\hat{\beta}_0$")
    axes[1].set_ylabel(r"$\hat{\beta}_1$")
    axes[1].set_title("Cloud of MLE estimates")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.colorbar(sc, ax=axes[1], label=r"$|\hat{x} - x^*|$")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_1_mle.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
