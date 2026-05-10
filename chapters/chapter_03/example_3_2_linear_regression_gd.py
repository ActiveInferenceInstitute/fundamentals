"""Example 3.2 — Linear regression by gradient descent.

Run::

    python chapters/chapter_03/example_3_2_linear_regression_gd.py [--save]

Visualizes the (β₀, β₁) loss surface for the squared error and overlays the
gradient-descent trajectory from a random initialization. Verifies the final
iterate matches the analytic normal-equation solution.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    LinearGaussianProcess,
    gd_linear_regression,
    get_logger,
    mle_linear_regression,
)
from active_inference.estimators.linear_regression import (
    add_intercept,
    squared_loss,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)
    x_pool = np.linspace(0.0, 5.0, args.n)
    y_pool = np.array([process.sample(float(x), n=1)[0] for x in x_pool])

    # Random initialization far from the optimum to make the trajectory visible.
    theta0 = rng.uniform(low=-50, high=50, size=2)
    result = gd_linear_regression(
        x_pool, y_pool,
        learning_rate=args.lr,
        max_iter=args.max_iter,
        theta0=theta0,
    )
    analytic = mle_linear_regression(x_pool, y_pool)
    LOG.info("Analytic θ = %s", np.round(analytic, 4))
    LOG.info("GD θ       = %s after %d iters (converged=%s)",
             np.round(result.theta, 4), result.n_iterations, result.converged)
    assert np.allclose(result.theta, analytic, atol=5e-3)

    # Build a (β₀, β₁) loss surface for visualization.
    b0_range = np.linspace(min(theta0[0], analytic[0]) - 5,
                           max(theta0[0], analytic[0]) + 5, 80)
    b1_range = np.linspace(min(theta0[1], analytic[1]) - 5,
                           max(theta0[1], analytic[1]) + 5, 80)
    B0, B1 = np.meshgrid(b0_range, b1_range)
    Z = np.empty_like(B0)
    X_aug = add_intercept(x_pool)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            theta_ij = np.array([B0[i, j], B1[i, j]])
            residuals = X_aug @ theta_ij - y_pool
            Z[i, j] = 0.5 * residuals @ residuals

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    cs = axes[0].contour(B0, B1, np.log(Z + 1), levels=20, cmap="viridis")
    axes[0].plot(result.history[:, 0], result.history[:, 1],
                 "o-", color="#d62728", ms=3, lw=1, label="iterate")
    axes[0].plot(*analytic, "x", color="green", ms=12, mew=2, label="analytic")
    axes[0].plot(*theta0, "o", color="black", ms=8, label="init")
    axes[0].set_xlabel(r"$\beta_0$")
    axes[0].set_ylabel(r"$\beta_1$")
    axes[0].set_title("Log loss surface + GD path")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    fig.colorbar(cs, ax=axes[0], label="log(loss + 1)")

    axes[1].plot(result.losses, color="#1f77b4", lw=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss vs iteration (log scale)")
    axes[1].grid(alpha=0.3, which="both")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_2_gd.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
