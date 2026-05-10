"""Bonus animation — bimodality emerging from a non-injective generator.

Run::

    python chapters/chapter_03/animation_bimodal_emergence.py [--save]

A quadratic generator has two pre-images for every observation. The
posterior is unimodal when the prior is concentrated near one pre-image
and *splits* into two modes when the prior straddles zero. The
animation sweeps the prior mean across the symmetry point so the
transition is visible in real time.
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
    animate_bimodal_emergence,
    save_animation,
)

LOG = get_logger("ch3.bimodal_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y-obs", type=float, default=11.0)
    p.add_argument("--n-frames", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = make_grid(-3.0, 3.0, 500)
    prior_means = np.linspace(-2.5, 2.5, args.n_frames)

    posteriors = []
    truths = []
    for m_x in prior_means:
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.05,
            m_x=float(m_x), s2_x=2.0, prior_kind="gaussian",
            psi=lambda x: x ** 2,
        )
        result = GridBayesianInference(model, x_grid).infer(args.y_obs)
        posteriors.append(result.posterior)
        # The true |x| under y = 3 + 2 x² with y = 11 is x = ±2.
        truths.append(np.sqrt((args.y_obs - 3.0) / 2.0))
    LOG.info("Built %d frames; prior mean swept from %.2f to %.2f",
             args.n_frames, prior_means[0], prior_means[-1])

    anim = animate_bimodal_emergence(
        x_grid, posteriors, list(map(float, prior_means)),
        truths=truths,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_bimodal_emergence.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
