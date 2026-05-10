"""Chapter 1 · Walking through Bayes' theorem one factor at a time.

Run::

    python chapters/chapter_01/03_bayes_intuition.py [--save]

Three panels are produced:

1. The four ingredients (prior, likelihood, evidence, posterior) plotted on a
   shared x-axis so their relative shapes are obvious at a glance.
2. The likelihood "in reverse" — fixing y and varying x to read off the
   credibility of each candidate hidden state.
3. The four-panel breakdown of how evidence normalizes the unnormalized
   posterior.
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
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch1.bayes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--y-obs", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_grid = make_grid(0.0, 5.0, 500)
    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.25,
        m_x=4.0, s2_x=0.25, prior_kind="gaussian",
    )
    inferer = GridBayesianInference(model, x_grid)
    res = inferer.infer(args.y_obs)
    LOG.info("Posterior mode for y=%.2f: %.3f (true inverse = %.3f)",
             args.y_obs, res.posterior_mode, (args.y_obs - 3) / 2)

    # Panel A: the textbook three-panel figure.
    fig_a = plot_prior_likelihood_posterior(
        res, title=f"Bayesian update for y = {args.y_obs}",
        truth=(args.y_obs - 3) / 2,
    )

    # Panel B: a single overlay showing how the posterior compromises.
    fig_b, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    norm_lik = res.likelihood / np.max(res.likelihood)
    norm_prior = res.prior / np.max(res.prior)
    norm_post = res.posterior / np.max(res.posterior)
    ax.plot(x_grid, norm_prior, lw=2, label="prior", color="#1f77b4")
    ax.plot(x_grid, norm_lik, lw=2, label="likelihood", color="#d62728")
    ax.plot(x_grid, norm_post, lw=2, label="posterior", color="#2ca02c")
    ax.axvline(res.posterior_mode, color="black", ls="--",
               label=f"posterior mode = {res.posterior_mode:.2f}")
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density (peak = 1)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Posterior is a precision-weighted compromise")

    # Panel C: highlight that the likelihood-as-a-function-of-x is unnormalized.
    fig_c, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].plot(x_grid, res.likelihood, color="#d62728", lw=2)
    axes[0].set_title("Likelihood (function of x; not normalized)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("credibility")
    axes[0].grid(alpha=0.3)

    unnormalized = np.exp(model.log_likelihood(args.y_obs, x_grid)
                          + model.log_prior(x_grid))
    evidence = np.trapezoid(unnormalized, x_grid)
    axes[1].plot(x_grid, unnormalized / evidence, color="#2ca02c", lw=2,
                 label="posterior")
    axes[1].fill_between(x_grid, unnormalized / evidence, alpha=0.25,
                         color="#2ca02c")
    axes[1].set_title(f"Posterior (evidence p(y) = {evidence:.4g})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if args.save:
        out_dir = ensure_dir(default_figure_dir() / "chapter_01")
        save_or_show(fig_a, out_dir / "03_bayes_three_panel.png")
        save_or_show(fig_b, out_dir / "03_bayes_overlay.png")
        save_or_show(fig_c, out_dir / "03_bayes_evidence.png")
        LOG.info("Saved figures to %s", out_dir)
    else:
        plt.show()


if __name__ == "__main__":
    main()
