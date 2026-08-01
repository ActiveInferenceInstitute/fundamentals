"""§5.2 — Predictive-coding recognition dynamics (Algorithm 5.2.1, Fig. 5.2.3).

Run::

    python chapters/chapter_05/example_5_4_recognition_dynamics.py [--save]
    python chapters/chapter_05/example_5_4_recognition_dynamics.py --linear [--save]

Implements Algorithm 5.2.1: gradient descent on the MAP free energy via
precision-weighted prediction errors, ``μ_x ← μ_x − κ ∂F/∂μ_x`` with
``∂F/∂μ_x = λ_x ε_x − λ_y ε_y g'(μ_x)``.

* **Nonlinear (default)** — Example 5.3/5.4: ``g(x)=x²+1``, ``m_x=3``, ``y=5.84``.
  ``μ_x`` falls from 3 toward the free-energy minimum (≈2.24) and the sensory
  prediction error ``ε_y`` decays toward zero (Fig. 5.2.3).
* **Linear** (``--linear``) — ``g(x)=2x+3``, ``m_x=4``, ``y=7``. The fixed point is
  ``μ_x = 2.4``, **identical to Chapter 4's exact grid posterior mean** — printed as
  an explicit oracle cross-check (three algorithms, one answer).
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
from active_inference.visualizations import plot_recognition_dynamics, save_or_show

LOG = get_logger("ch5.ex4")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--linear", action="store_true",
                   help="run the linear model (oracle = Ch.4 posterior mean 2.4)")
    return p.parse_args()


def run_linear():
    """Compute a chapter-local helper quantity for the orchestrated example."""
    model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                  m_x=4.0, s2_x=0.25)
    y = 7.0
    res = predictive_coding_inference(model, y, kappa=0.02, n_iter=400)
    # Oracle: Chapter 4 grid posterior mean.
    lgm = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
    grid = np.linspace(-6.0, 12.0, 2001)
    post = GridBayesianInference(lgm, grid).infer(y)
    agree = oracle_agreement(res.mu_star, post.posterior_mean, tol=1e-2)
    LOG.info("LINEAR: PC mu*=%.5f | grid posterior mean=%.5f | agree=%s (err=%.2e)",
             res.mu_star, post.posterior_mean, agree.passed, agree.abs_error)
    fig = plot_recognition_dynamics(
        res, truth=2.0, oracle=float(post.posterior_mean),
        surprisal=surprisal(lgm, y, grid),
        title="Example 5.4 · linear PC — fixed point = Ch.4 posterior mean")
    return fig, "example_5_4_recognition_linear.png"


def run_nonlinear():
    """Compute a chapter-local helper quantity for the orchestrated example."""
    model = PredictiveCodingModel(g=QuadraticFunction(1.0, 1.0), sigma2_y=0.25,
                                  m_x=3.0, s2_x=0.25)
    y = 5.84
    res = predictive_coding_inference(model, y, kappa=0.01, n_iter=2000)
    grid = np.linspace(0.0, 4.0, 8001)
    F = [predictive_coding_free_energy(model, y, float(u)).free_energy for u in grid]
    argmin = float(grid[int(np.argmin(F))])
    LOG.info("NONLINEAR: PC mu*=%.5f | grid argmin F=%.5f | eps_y: %.3f -> %.3e",
             res.mu_star, argmin, res.eps_y[0], res.eps_y[-1])
    fig = plot_recognition_dynamics(
        res, truth=argmin, oracle=argmin,
        title="Example 5.4 · nonlinear predictive coding (Fig. 5.2.3)")
    return fig, "example_5_4_recognition_nonlinear.png"


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    fig, name = run_linear() if args.linear else run_nonlinear()
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / name)
        LOG.info("Saved to %s", out / name)
    else:
        plt.show()


if __name__ == "__main__":
    main()
