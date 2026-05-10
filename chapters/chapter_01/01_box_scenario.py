"""Chapter 1 · Box scenario — an agent receives a stream of noisy sensor readings.

Run::

    python chapters/chapter_01/01_box_scenario.py [--save] [--seed N]

This is a thin orchestrator: a single hidden environmental state generates
multiple noisy observations through a linear-Gaussian process. The script
plots the noise-free generator, the sampled stream, and a histogram of the
sensor values to convey "the agent only sees this stream — it must reconstruct
the world from it."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from active_inference import LinearGaussianProcess, get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_generating_function, save_or_show

LOG = get_logger("ch1.box")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true", help="save figures instead of showing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--x-true", type=float, default=2.5,
                   help="The single hidden state of the environment")
    p.add_argument("--sigma2-y", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=args.sigma2_y, rng=rng,
    )
    LOG.info("Generative process: %s", process)

    # Stream of n noisy observations from the same hidden state.
    samples = process.sample(x_star=args.x_true, n=args.n_samples).flatten()
    LOG.info(
        "Stream stats: mean=%.3f  std=%.3f  (true mean=%.3f)",
        samples.mean(), samples.std(),
        process.mean(args.x_true),
    )

    # Visualization 1: generating function with samples scattered along x = x_true
    x_grid = np.linspace(0.0, 5.0, 400)
    f_x = process.mean(x_grid)
    fig1 = plot_generating_function(
        x_grid, f_x,
        samples_x=np.full_like(samples, args.x_true),
        samples_y=samples,
        title="Box scenario · noisy linear sensor",
        xlabel="external state x*  (food size, a.u.)",
        ylabel="observation y  (light intensity, a.u.)",
    )

    # Visualization 2: time series + histogram of the stream
    fig2, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].plot(samples, color="#1f77b4", lw=1)
    axes[0].axhline(process.mean(args.x_true), color="red", ls="--",
                    label=f"true mean {process.mean(args.x_true):.2f}")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("y")
    axes[0].set_title("Sensor stream")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(samples, bins=30, color="#1f77b4", alpha=0.7,
                 edgecolor="white")
    axes[1].axvline(process.mean(args.x_true), color="red", ls="--",
                    label="true mean")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("count")
    axes[1].set_title("Sensor histogram")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if args.save:
        out_dir = ensure_dir(default_figure_dir() / "chapter_01")
        save_or_show(fig1, out_dir / "01_box_scenario_generator.png")
        save_or_show(fig2, out_dir / "01_box_scenario_stream.png")
        LOG.info("Saved figures to %s", out_dir)
    else:
        plt.show()


if __name__ == "__main__":
    main()
