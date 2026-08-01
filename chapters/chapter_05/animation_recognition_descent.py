"""Animation — predictive-coding recognition dynamics (Chapter 5, bonus).

Run::

    python chapters/chapter_05/animation_recognition_descent.py [--save] [--nonlinear]

Renders a GIF of Algorithm 5.2.1 in motion: the belief mean ``μ_x`` descending the
MAP free energy toward its fixed point, the precision-weighted prediction errors
``ε_x``/``ε_y`` decaying, and the free energy falling. The default **linear** model
lands exactly on the Chapter 4 grid posterior mean (drawn as the ``oracle`` line) —
the dynamic version of the cross-chapter euphoric surprise (predictive coding,
variational inference and grid Bayes converging on the same belief). Pass
``--nonlinear`` for the ``g(x)=x²+1`` example (Fig. 5.2.3).

Built on the composable :func:`animate_recognition_dynamics`, which animates any
result exposing ``mus`` / ``free_energies`` traces — the same function used for the
Chapter 4 fixed-form result.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearFunction,
    LinearGaussianModel,
    PredictiveCodingModel,
    QuadraticFunction,
    get_logger,
    oracle_agreement,
    predictive_coding_free_energy,
    predictive_coding_inference,
    surprisal,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_recognition_dynamics, save_animation

LOG = get_logger("ch5.anim_recognition")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--nonlinear", action="store_true",
                   help="run g(x)=x²+1 (default is the linear oracle model)")
    p.add_argument("--stride", type=int, default=1,
                   help="animate every Nth iteration (keeps the GIF small)")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    grid = np.linspace(-6.0, 12.0, 2001)

    if args.nonlinear:
        model = PredictiveCodingModel(g=QuadraticFunction(1.0, 1.0), sigma2_y=0.25,
                                      m_x=3.0, s2_x=0.25)
        y = 5.84
        res = predictive_coding_inference(model, y, kappa=0.008, n_iter=200)
        fvals = [predictive_coding_free_energy(model, y, float(m)).free_energy
                 for m in grid]
        oracle = float(grid[int(np.argmin(fvals))])  # grid argmin of F (nonlinear)
        LOG.info("NONLINEAR: PC mu*=%.5f | grid argmin F=%.5f", res.mu_star, oracle)
        title = "Example 5.4 · nonlinear recognition dynamics"
        truth = 2.2385
    else:
        model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                      m_x=4.0, s2_x=0.25)
        y = 7.0
        # kappa=0.02 gives a gradual, watchable ~25-step descent (kappa=0.05
        # would snap to the fixed point in ~3 frames).
        res = predictive_coding_inference(model, y, kappa=0.02, n_iter=120)
        lgm = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                  m_x=4.0, s2_x=0.25)
        oracle = float(GridBayesianInference(lgm, grid).infer(y).posterior_mean)
        agree = oracle_agreement(res.mu_star, oracle, tol=1e-2)
        LOG.info("LINEAR: PC mu*=%.5f | oracle=%.5f | agree=%s (err=%.2e)",
                 res.mu_star, oracle, agree.passed, agree.abs_error)
        title = "Example 5.4 · linear recognition dynamics → Ch.4 posterior mean"
        truth = 2.0

    sup = surprisal(
        LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25),
        y, grid) if not args.nonlinear else None
    anim = animate_recognition_dynamics(
        res, truth=truth, oracle=oracle, surprisal=sup, title=title,
        stride=args.stride)
    LOG.info("built recognition animation: %d iterations, stride=%d",
             len(res.mus), args.stride)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        name = ("animation_recognition_nonlinear.gif" if args.nonlinear
                else "animation_recognition_linear.gif")
        path = save_animation(anim, out / name, fps=12)
        LOG.info("Saved to %s", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
