"""Bonus animation — LGS sensor fusion one observation at a time.

Run::

    python chapters/chapter_03/animation_lgs_online.py [--save] [--n 30]

The 2-D credible ellipse shrinks (and rotates, when ``cov_y`` is
anisotropic) as each new noisy observation is folded into the
posterior. Information accumulation in vector-valued inference made
visible.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianMVProcess,
    LinearGaussianSystem,
    diagonal_cov,
    get_logger,
    isotropic_cov,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_lgs_online, save_animation

LOG = get_logger("ch3.lgs_online_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    truth = np.array([0.4, 0.6])
    # Anisotropic noise so the ellipse rotates as observations arrive.
    cov_y = diagonal_cov(np.array([0.10, 0.04]))
    proc = LinearGaussianMVProcess(Theta=np.eye(2), cov_y=cov_y, rng=rng)
    Y = proc.sample(truth, n=args.n).reshape(args.n, 2)

    lgs = LinearGaussianSystem(
        Theta=np.eye(2), cov_y=cov_y,
        mx=np.array([0.5, 0.5]), cov_x=isotropic_cov(2, 1.0),
    )
    means = np.empty((args.n, 2))
    covs = np.empty((args.n, 2, 2))
    for t in range(args.n):
        post = lgs.posterior_batch(Y[: t + 1])
        means[t] = post.mean
        covs[t] = post.cov

    LOG.info("Final posterior mean = %s, std = %s",
             np.round(means[-1], 3), np.round(np.sqrt(np.diag(covs[-1])), 3))

    anim = animate_lgs_online(
        means, covs, observations=Y, truth=truth,
        xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
        title="Chapter 3 · LGS sensor fusion · online ellipse contraction",
        interval_ms=120,
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        path = save_animation(anim, out / "animation_lgs_online.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
