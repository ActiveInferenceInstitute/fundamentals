"""Bonus — animated gradient descent for the MLE / MAP loss.

Run::

    python chapters/chapter_02/animation_gradient_descent.py [--save] [--n 200]

Visualizes the iterate rolling down the negative log-likelihood surface, with
a synchronized loss-vs-iteration trace. Saves to GIF when ``--save`` is set.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianProcess,
    get_logger,
    gradient_descent,
)
from active_inference.estimators.mle import mle_grad_x, mle_loss
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    animate_gradient_descent,
    save_animation,
)

LOG = get_logger("ch2.gd_anim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--x-true", type=float, default=2.5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng)
    ys = process.sample(args.x_true, n=args.n).flatten()

    result = gradient_descent(
        loss_fn=lambda x: float(mle_loss(x, ys, 3.0, 2.0, 0.5)),
        grad_fn=lambda x: mle_grad_x(x, ys, 3.0, 2.0, 0.5),
        x0=5.0,
        learning_rate=args.lr,
        max_iter=args.max_iter,
    )

    x_grid = np.linspace(0.0, 5.0, 500)
    loss_grid = np.array([float(mle_loss(x, ys, 3.0, 2.0, 0.5)) for x in x_grid])

    LOG.info("Converged after %d iterations to x = %.4f (true %.4f)",
             result.n_iterations, result.x, args.x_true)

    anim = animate_gradient_descent(
        loss_grid, x_grid,
        history=result.history,
        losses=result.losses,
        truth=args.x_true,
        title="Chapter 2 · gradient descent on the NLL",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        path = save_animation(anim, out / "animation_gradient_descent.gif")
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
