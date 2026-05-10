"""Example 2.9 — Closed-form maximum a posteriori estimation.

Run::

    python chapters/chapter_02/example_2_9_map_analytic.py [--save]

The MAP estimator is a precision-weighted average of the data-driven MLE and
the prior mean. We verify the analytic formula against the grid Bayesian
posterior mode, then sweep ``s2_x`` to show the smooth interpolation between
those two extremes.
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
    map_analytic_linear,
    mle_analytic_linear,
    map_loss,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch2.ex9")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--x-true", type=float, default=2.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng)
    samples = process.sample(args.x_true, n=args.n_samples).flatten()

    x_mle = mle_analytic_linear(samples, beta0=3.0, beta1=2.0)
    x_map = map_analytic_linear(
        samples, beta0=3.0, beta1=2.0, sigma2_y=0.5, m_x=4.0, s2_x=0.25
    )

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.5,
        m_x=4.0, s2_x=0.25, prior_kind="gaussian",
    )
    res = GridBayesianInference(model, x_grid).infer(samples)
    LOG.info(
        "MLE = %.4f, MAP (analytic) = %.4f, grid mode = %.4f, true x* = %.3f",
        x_mle, x_map, res.posterior_mode, args.x_true,
    )

    nll_map = map_loss(
        x_grid, samples, beta0=3.0, beta1=2.0, sigma2_y=0.5,
        m_x=4.0, s2_x=0.25,
    )

    # Sweep prior variance from "stubborn" to "ignored".
    s2_grid = np.geomspace(0.05, 50.0, 40)
    map_traj = [
        map_analytic_linear(samples, 3.0, 2.0, 0.5, 4.0, float(s2)) for s2 in s2_grid
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x_grid, nll_map, color="#1f77b4", lw=2,
                 label="−log p(x|y) (up to const.)")
    axes[0].axvline(x_map, color="red", ls="--", label=f"MAP = {x_map:.3f}")
    axes[0].axvline(x_mle, color="orange", ls=":", label=f"MLE = {x_mle:.3f}")
    axes[0].axvline(args.x_true, color="green", ls=":", label=f"x* = {args.x_true}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("MAP loss")
    axes[0].set_title("MAP objective and estimators")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].semilogx(s2_grid, map_traj, color="#2ca02c", lw=2)
    axes[1].axhline(x_mle, color="orange", ls=":", label=f"MLE limit = {x_mle:.3f}")
    axes[1].axhline(4.0, color="red", ls=":", label="prior mean = 4")
    axes[1].axhline(args.x_true, color="green", ls="--", label=f"x* = {args.x_true}")
    axes[1].set_xlabel("prior variance s2_x  (log scale)")
    axes[1].set_ylabel("MAP estimate")
    axes[1].set_title("Prior strength sweep")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig, out / "example_2_9_map.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
