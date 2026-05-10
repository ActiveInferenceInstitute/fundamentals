"""Chapter 1 · Three perspectives on what a "model" is.

Run::

    python chapters/chapter_01/02_three_perspectives.py [--save]

Side-by-side panels:

A) *Scientific* — a researcher's external model of an unknown world.
B) *Hypothesis-testing brain* — predictions, errors, and corrections.
C) *Statistical* — a generative process and the agent's generative model.

Each panel uses the same toy environment (linear-Gaussian sensor) but
illustrates a different conceptual emphasis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    LinearGaussianModel,
    LinearGaussianProcess,
    GridBayesianInference,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show

LOG = get_logger("ch1.perspectives")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.4, rng=rng)
    x_grid = make_grid(0.0, 5.0, 400)
    f_x = process.mean(x_grid)
    x_true = 2.5
    samples = process.sample(x_star=x_true, n=80).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    # A — Scientific: unknown function, scientist plots their hypothesized fit.
    axes[0].scatter(np.full_like(samples, x_true), samples,
                    s=14, alpha=0.6, color="#d2691e", label="observations")
    poly = np.poly1d(np.polyfit([x_true] * samples.size + [0.0],
                                list(samples) + [3.0], 1))
    axes[0].plot(x_grid, poly(x_grid), color="#888", ls="--",
                 label="scientist's fit")
    axes[0].plot(x_grid, f_x, color="#333", lw=2, label="true (unknown) g(x)")
    axes[0].set_title("A · Scientific modeling\n(\"reality is a target system\")")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # B — Hypothesis-testing brain: predictions and prediction errors.
    pred_mean = float(np.mean(samples))
    errors = samples - pred_mean
    axes[1].axhline(pred_mean, color="#1f77b4", lw=2, label="prediction")
    axes[1].vlines(np.arange(samples.size), pred_mean, samples,
                   colors="#d62728", alpha=0.5, label="prediction error")
    axes[1].scatter(np.arange(samples.size), samples, s=10, color="black")
    axes[1].set_title("B · Hypothesis-testing brain\n(prediction → error → update)")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    LOG.info("Mean prediction error magnitude: %.3f", np.mean(np.abs(errors)))

    # C — Statistical: posterior over the hidden state given the stream.
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.4,
        m_x=2.0, s2_x=2.0, prior_kind="gaussian",
    )
    res = GridBayesianInference(model, x_grid).infer(samples)
    axes[2].plot(x_grid, res.prior, color="#1f77b4", label="prior")
    axes[2].plot(x_grid, res.posterior, color="#2ca02c", lw=2, label="posterior")
    axes[2].axvline(x_true, color="red", ls=":", label=f"x* = {x_true}")
    axes[2].axvline(res.posterior_mode, color="black", ls="--",
                    label=f"mode = {res.posterior_mode:.3f}")
    axes[2].set_title("C · Statistical\n(generative model inverts the process)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("density")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    LOG.info("Posterior mode after %d samples: %.3f", samples.size,
             res.posterior_mode)

    fig.suptitle("Three perspectives on the same environment", fontsize=12)

    if args.save:
        out_dir = ensure_dir(default_figure_dir() / "chapter_01")
        save_or_show(fig, out_dir / "02_three_perspectives.png")
        LOG.info("Saved figure to %s", out_dir)
    else:
        plt.show()


if __name__ == "__main__":
    main()
