"""Example 2.8 — Closed-form maximum likelihood estimation.

Run::

    python chapters/chapter_02/example_2_8_mle_analytic.py [--save]

Compares the analytic MLE ``(mean(y) - beta0) / beta1`` against the posterior
mode obtained from a uniform-prior grid Bayesian inference. They should agree
up to grid resolution.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    LinearGaussianProcess,
    get_logger,
    make_grid,
    mle_analytic_linear,
    mle_loss,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch2.ex8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--x-true", type=float, default=2.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng)
    samples = process.sample(args.x_true, n=args.n_samples).flatten()

    x_mle = mle_analytic_linear(samples, beta0=3.0, beta1=2.0)

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.25,
        prior_kind="uniform", uniform_low=0.0, uniform_high=5.0,
    )
    res = GridBayesianInference(model, x_grid).infer(samples)
    LOG.info("Analytic MLE = %.4f, grid posterior mode = %.4f, true x* = %.4f",
             x_mle, res.posterior_mode, args.x_true)

    nll = mle_loss(x_grid, samples, beta0=3.0, beta1=2.0, sigma2_y=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x_grid, nll, color="#1f77b4", lw=2, label="−ℓ(x)")
    axes[0].axvline(x_mle, color="red", ls="--", label=f"x_MLE = {x_mle:.3f}")
    axes[0].axvline(args.x_true, color="green", ls=":", label=f"x* = {args.x_true}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("negative log-likelihood")
    axes[0].set_title("MLE objective")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_grid, res.posterior, color="#2ca02c", lw=2,
                 label="grid posterior (uniform prior)")
    axes[1].axvline(x_mle, color="red", ls="--", label=f"analytic MLE = {x_mle:.3f}")
    axes[1].axvline(res.posterior_mode, color="black", ls=":",
                    label=f"grid mode = {res.posterior_mode:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density")
    axes[1].set_title("Analytic MLE matches grid mode")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig, out / "example_2_8_mle.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
