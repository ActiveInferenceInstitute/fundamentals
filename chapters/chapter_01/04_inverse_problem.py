"""Chapter 1 · The inverse problem becomes ill-posed for non-injective generators.

Run::

    python chapters/chapter_01/04_inverse_problem.py [--save]

A quadratic generating function ``y = beta0 + beta1 * x**2`` produces the same
observation for ``+x`` and ``-x``. With a uniform prior over a symmetric domain,
the posterior is bi-modal — a clean illustration of why the inverse problem
matters.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_generating_function,
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch1.inverse")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    x_grid = make_grid(-2.5, 2.5, 600)

    # Quadratic generator with deterministic-ish likelihood (uniform prior).
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.05,
        prior_kind="uniform", uniform_low=-2.5, uniform_high=2.5,
        psi=lambda x: x ** 2,
    )

    res = GridBayesianInference(model, x_grid).infer(args.y_obs)

    fig_curve = plot_generating_function(
        x_grid, model.predict_mean(x_grid),
        title="Non-injective generator g(x) = 3 + 2 x^2",
        ylabel="observation y",
    )

    fig_post = plot_prior_likelihood_posterior(
        res, title=f"Bi-modal posterior for y = {args.y_obs}",
    )

    # Pick out the two posterior modes by splitting the grid at zero.
    left = x_grid < 0
    right = x_grid >= 0
    mode_left = float(x_grid[left][np.argmax(res.posterior[left])])
    mode_right = float(x_grid[right][np.argmax(res.posterior[right])])
    LOG.info("Posterior modes at x = %.3f and x = %.3f", mode_left, mode_right)

    # A small overlay that annotates the two modes.
    fig_overlay, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(x_grid, res.posterior, color="#2ca02c", lw=2)
    ax.fill_between(x_grid, res.posterior, alpha=0.25, color="#2ca02c")
    for m, label in [(mode_left, "left mode"), (mode_right, "right mode")]:
        ax.axvline(m, color="black", ls="--", lw=1)
        ax.annotate(f"{label}\n{m:.3f}", xy=(m, np.max(res.posterior) * 0.9),
                    ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))
    ax.set_xlabel("x")
    ax.set_ylabel("posterior density")
    ax.set_title("Inverse problem: two states explain the same observation")
    ax.grid(alpha=0.3)

    if args.save:
        out_dir = ensure_dir(default_figure_dir() / "chapter_01")
        save_or_show(fig_curve, out_dir / "04_inverse_curve.png")
        save_or_show(fig_post, out_dir / "04_inverse_posterior.png")
        save_or_show(fig_overlay, out_dir / "04_inverse_overlay.png")
        LOG.info("Saved figures to %s", out_dir)
    else:
        plt.show()


if __name__ == "__main__":
    main()
