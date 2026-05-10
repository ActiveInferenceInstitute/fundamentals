"""Example 2.7 — Bayesian inference from a batch of i.i.d. observations.

Run::

    python chapters/chapter_02/example_2_7_multiple_samples.py [--save]

We compare two equivalent strategies:

A) Sequential update — each observation's posterior becomes the next prior.
B) Batch update — sum log-likelihoods first, then divide by evidence.

Both yield the same final posterior thanks to exchangeability. The script also
plots a per-sample ridge of likelihoods (figure 2.3.2 in the book) and a
convergence curve showing how the posterior tightens with N.
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
    plot_likelihood_ridge,
    plot_prior_likelihood_posterior,
    save_or_show,
)

LOG = get_logger("ch2.ex7")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--x-true", type=float, default=2.0)
    return p.parse_args()


def sequential_inference(
    model: LinearGaussianModel,
    x_grid: np.ndarray,
    samples: np.ndarray,
):
    """Yield (i, posterior) after assimilating the first i+1 samples one by one.

    Implemented in log-space directly so we can keep using the same
    LinearGaussianModel without mutating it.
    """
    log_state = model.log_prior(x_grid).copy()
    for i, y in enumerate(samples):
        log_state = log_state + model.log_likelihood(float(y), x_grid)
        normed = np.exp(log_state - np.max(log_state))
        normed /= np.trapezoid(normed, x_grid)
        yield i, normed


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng)
    samples = process.sample(args.x_true, n=args.n_samples).flatten()
    LOG.info("Sample mean = %.3f, std = %.3f (true mean = %.3f)",
             samples.mean(), samples.std(), process.mean(args.x_true))

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.25,
        m_x=4.0, s2_x=0.25, prior_kind="gaussian",
    )

    # Batch inference (single shot)
    res_batch = GridBayesianInference(model, x_grid).infer(samples)
    LOG.info("Batch posterior mode = %.4f, var = %.4g",
             res_batch.posterior_mode, res_batch.posterior_variance)

    # Sequential inference, recording the trajectory of mode + 95% CI
    modes = []
    los = []
    his = []
    for i, post in sequential_inference(model, x_grid, samples):
        modes.append(float(x_grid[int(np.argmax(post))]))
        cdf = np.concatenate(
            ([0.0],
             np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(x_grid)))
        )
        cdf /= cdf[-1]
        los.append(float(np.interp(0.025, cdf, x_grid)))
        his.append(float(np.interp(0.975, cdf, x_grid)))

    final_post = post  # last yielded posterior
    LOG.info("Sequential final mode = %.4f (matches batch within tol)",
             modes[-1])
    assert np.allclose(modes[-1], res_batch.posterior_mode, atol=2e-3)

    # Per-sample likelihoods for the ridge plot.
    per_sample_lik = [np.exp(model.log_likelihood(float(y), x_grid)) for y in samples[:9]]

    fig_ridge = plot_likelihood_ridge(
        x_grid,
        per_sample_lik,
        labels=[f"y_{i + 1} = {y:.2f}" for i, y in enumerate(samples[:9])],
        title="Example 2.7 · per-sample likelihoods (first 9 of N)",
    )

    fig_post = plot_prior_likelihood_posterior(
        res_batch, title=f"Example 2.7 · posterior after N = {args.n_samples}",
        truth=args.x_true,
    )

    fig_conv, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    n_axis = np.arange(1, args.n_samples + 1)
    ax.plot(n_axis, modes, color="#2ca02c", lw=2, label="posterior mode")
    ax.fill_between(n_axis, los, his, alpha=0.25, color="#2ca02c",
                    label="95% credible interval")
    ax.axhline(args.x_true, color="red", ls=":", lw=1.5, label="true x*")
    ax.set_xlabel("samples assimilated")
    ax.set_ylabel("estimate")
    ax.set_title("Sequential Bayesian update converges to the truth")
    ax.legend()
    ax.grid(alpha=0.3)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_ridge, out / "example_2_7_ridge.png")
        save_or_show(fig_post, out / "example_2_7_posterior.png")
        save_or_show(fig_conv, out / "example_2_7_convergence.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
