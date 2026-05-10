"""Bonus animation — sweep the prior↔data precision tradeoff.

Run::

    python chapters/chapter_03/animation_precision_sweep.py [--save]

The single most subtle effect in Bayesian inference is the smooth
interpolation between prior-dominated and data-dominated posteriors as
the precision ratio is swept. This animation makes that interpolation
visible in real time.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_precision_sweep,
    save_animation,
)

LOG = get_logger("ch3.precision_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=7.0)
    p.add_argument("--n-frames", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = make_grid(0.0, 5.0, 500)
    log_ratios = np.linspace(-2.0, 2.0, args.n_frames)

    priors, likelihoods, posteriors = [], [], []
    for log_ratio in log_ratios:
        ratio = 10.0 ** float(log_ratio)
        sigma2_y = 1.0 / ratio
        s2_x = ratio
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
            m_x=4.0, s2_x=s2_x, prior_kind="gaussian",
        )
        result = GridBayesianInference(model, x_grid).infer(args.y_obs)
        priors.append(result.prior)
        likelihoods.append(result.likelihood)
        posteriors.append(result.posterior)
    LOG.info("Built %d frames sweeping log10(s2_x/sigma2_y) ∈ [%.1f, %.1f]",
             args.n_frames, log_ratios[0], log_ratios[-1])

    truth = (args.y_obs - 3.0) / 2.0
    anim = animate_precision_sweep(
        x_grid, priors, likelihoods, posteriors, list(log_ratios),
        truth=truth,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_precision_sweep.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
