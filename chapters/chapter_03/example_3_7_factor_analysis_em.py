"""Example 3.7 — Factor analysis via expectation–maximization.

Run::

    python chapters/chapter_03/example_3_7_factor_analysis_em.py [--save]

Synthetic data: ``Y = X Θ_true^T + diag(σ²) noise`` with a 2-D latent space
and a 5-D observation space. EM recovers ``Θ`` and the diagonal of the noise
covariance up to a rotation, while the marginal log-likelihood increases
monotonically.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    diagonal_cov,
    fit_factor_analysis,
    get_logger,
    mvn_sample,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch3.ex7")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-factors", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Ground truth model.
    true_Theta = np.array([
        [1.0, 0.5],
        [0.7, -0.3],
        [-0.2, 1.0],
        [0.4, 0.4],
        [0.0, 0.9],
    ])
    true_diag = np.array([0.10, 0.20, 0.05, 0.30, 0.15])
    n_obs = true_Theta.shape[0]

    # Sample latent x ~ N(0, I), observe y = Theta x + noise.
    X_latent = mvn_sample(np.zeros(args.n_factors),
                          np.eye(args.n_factors),
                          n=args.n_samples, rng=rng)
    noise = mvn_sample(np.zeros(n_obs), diagonal_cov(true_diag),
                       n=args.n_samples, rng=rng)
    Y = X_latent @ true_Theta.T + noise

    result = fit_factor_analysis(
        Y, n_factors=args.n_factors,
        max_iter=args.max_iter, tol=1e-6,
        rng=np.random.default_rng(args.seed + 1),
    )
    LOG.info("Converged=%s after %d iterations.",
             result.converged, result.n_iterations)
    LOG.info("Final marginal LL = %.3f", result.log_likelihoods[-1])

    # Reconstruct y from posterior latent means.
    Y_hat = result.posterior_means @ result.Theta.T
    rmse = float(np.sqrt(((Y - Y_hat) ** 2).mean()))
    LOG.info("Reconstruction RMSE = %.4f", rmse)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axes[0, 0].plot(result.log_likelihoods, color="#1f77b4", lw=2)
    axes[0, 0].set_xlabel("EM iteration")
    axes[0, 0].set_ylabel("incomplete log p(Y)")
    axes[0, 0].set_title("Marginal log-likelihood (monotone ↑)")
    axes[0, 0].grid(alpha=0.3)

    width = 0.35
    idx = np.arange(true_diag.size)
    axes[0, 1].bar(idx - width / 2, true_diag, width=width,
                   color="#1f77b4", label="true")
    axes[0, 1].bar(idx + width / 2, np.diag(result.cov_y), width=width,
                   color="#2ca02c", label="EM estimate")
    axes[0, 1].set_xticks(idx)
    axes[0, 1].set_xticklabels([f"y_{i}" for i in idx])
    axes[0, 1].set_ylabel("noise variance")
    axes[0, 1].set_title("Per-channel noise (recovered)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis="y")

    vmax = max(np.abs(true_Theta).max(), np.abs(result.Theta).max())
    im0 = axes[1, 0].imshow(true_Theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            aspect="auto")
    axes[1, 0].set_title("True loadings  Θ*")
    fig.colorbar(im0, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_xlabel("factor")
    axes[1, 0].set_ylabel("output dim")

    im1 = axes[1, 1].imshow(result.Theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            aspect="auto")
    axes[1, 1].set_title(f"EM estimate  Θ̂  (rmse = {rmse:.3f})")
    fig.colorbar(im1, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_xlabel("factor")
    axes[1, 1].set_ylabel("output dim")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_7_factor_analysis_em.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
