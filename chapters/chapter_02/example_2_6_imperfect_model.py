"""Example 2.6 — Mismatched generative model and generative process.

Run::

    python chapters/chapter_02/example_2_6_imperfect_model.py [--save]

The environment is quadratic; the agent assumes it is linear. We sweep the
true hidden state over a range and watch how badly the agent's posterior
shifts away from the truth as we leave the agent's "linear comfort zone".
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
from active_inference.visualizations import save_or_show

LOG = get_logger("ch2.ex6")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    psi = lambda x: x ** 2
    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng, nonlinear=psi,
    )

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(  # agent assumes linear
        beta0=3.0, beta1=2.0, sigma2_y=0.25,
        m_x=4.0, s2_x=0.25, prior_kind="gaussian",
    )
    inferer = GridBayesianInference(model, x_grid)

    truths = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    rows = []
    for x_true in truths:
        ys = process.sample(x_true, n=args.n_samples).flatten()
        res = inferer.infer(ys)
        rows.append((x_true, res.posterior_mode, res.posterior_mode - x_true))
    LOG.info("x*       posterior mode   bias")
    for r in rows:
        LOG.info("%.2f     %8.4f       %+8.4f", *r)

    # Two-panel figure: true generator vs assumed generator + posterior modes vs truth.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(x_grid, process.mean(x_grid), color="#d62728", lw=2,
                 label="true (quadratic)")
    axes[0].plot(x_grid, model.predict_mean(x_grid), color="#1f77b4", lw=2,
                 label="agent (linear)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Generator mismatch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    truth_arr = np.array([r[0] for r in rows])
    mode_arr = np.array([r[1] for r in rows])
    axes[1].plot(truth_arr, truth_arr, color="black", ls="--", label="ideal")
    axes[1].plot(truth_arr, mode_arr, "o-", color="#2ca02c",
                 label="posterior mode")
    for x_t, m in zip(truth_arr, mode_arr):
        axes[1].annotate(f"{m - x_t:+.2f}", xy=(x_t, m), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_xlabel("true x*")
    axes[1].set_ylabel("posterior mode")
    axes[1].set_title("Bias of the linear agent")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig, out / "example_2_6_imperfect_model.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
