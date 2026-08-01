"""§5.3 / §5.5 — Multivariate predictive coding.

Run:: ``python chapters/chapter_05/example_5_3_multivariate.py [--save] [--regime linear|nonlinear]``

Generalizes the scalar recognition dynamics to a vector hidden state with precision
*matrices* and a Jacobian matrix (book §5.3): ``μ ← μ − κ(Π_x ε_x − Jᵀ Π_y ε_y)``.
Each regime carries an independent oracle so the fixed point is *verified*, not merely
plotted.

* **linear** (default) — ``g(x) = A x + b``, constant Jacobian ``A``; under a near-flat
  prior the fixed point is the least-squares inverse ``A⁻¹(y−b)``, computed independently
  by :func:`~active_inference.pc_multivariate_linear_fixed_point`. On a 1-D restriction
  this reproduces the scalar predictive-coding result exactly.
* **nonlinear** (``--regime nonlinear``) — the book's §5.5 model ``g(x) = x ⊙ x + 1``
  (element-wise square), ``x* = [0.5, 2.5]``, ``m_x = [1, 1]``, ``Σ_y = 0.5 I`` with a
  near-flat state prior (isolating the likelihood so the fixed point is the ML inverse).
  Since ``g`` depends on ``x`` only via ``u = x⊙x``, a noiseless observation is inverted
  exactly by ``x* = sign ⊙ √(y − 1)``, computed independently by
  :func:`~active_inference.pc_parameterized_lstsq_oracle` — the faithful nonlinear
  counterpart of the linear demo, and a square special case of ``example_5_6_parameterized.py``
  (whose ``--regime informative`` keeps the prior in play).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    get_logger,
    multivariate_predictive_coding,
    pc_multivariate_linear_fixed_point,
    pc_parameterized_lstsq_oracle,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_multivariate_pc, save_or_show

LOG = get_logger("ch5.ex3")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--regime", choices=("linear", "nonlinear"), default="linear",
                   help="linear g(x)=Ax+b (default) or the book's §5.5 g(x)=x⊙x+1")
    return p.parse_args()


def _run_linear():
    """Linear g(x)=Ax+b with the closed-form least-squares oracle (near-flat prior)."""
    A = np.array([[2.0, 0.5], [-0.3, 1.5]])
    b = np.array([1.0, -1.0])
    x_true = np.array([2.0, -1.0])
    y = A @ x_true + b
    precision_y = np.array([1.0, 1.0])
    precision_x = np.array([1e-4, 1e-4])  # near-flat prior → least-squares

    res = multivariate_predictive_coding(
        g=lambda x: A @ x + b, jacobian=lambda x: A, y=y, m_x=np.zeros(2),
        precision_y=precision_y, precision_x=precision_x, kappa=0.05, n_iter=5000,
    )
    oracle = pc_multivariate_linear_fixed_point(
        A, b, y, np.zeros(2), precision_y=precision_y, precision_x=precision_x)
    title = "§5.3 · linear multivariate PC (closed-form least-squares oracle)"
    return res, x_true, oracle, title


def _run_nonlinear():
    """Book §5.5: nonlinear g(x)=x⊙x+1, recovered against the √-inverse oracle."""
    b = np.ones(2)
    x_true = np.array([0.5, 2.5])
    y = x_true * x_true + b

    def g(x: np.ndarray) -> np.ndarray:
        """Book §5.5 element-wise-square generating function g(x)=x⊙x+1."""
        return x * x + b

    def jacobian(x: np.ndarray) -> np.ndarray:
        """Diagonal Jacobian ∂g/∂x = diag(2x)."""
        return np.diag(2.0 * x)

    # Near-flat prior isolates the likelihood so the fixed point is the ML inverse;
    # kappa=0.02 keeps the quartic descent stable at x≈2.5 (larger steps overshoot).
    res = multivariate_predictive_coding(
        g=g, jacobian=jacobian, y=y, m_x=np.ones(2),
        precision_y=np.full(2, 2.0), precision_x=np.full(2, 1e-6),
        mu0=np.sign(x_true) * np.ones(2), kappa=0.02, n_iter=200000, tol=1e-14,
    )
    oracle = pc_parameterized_lstsq_oracle(np.eye(2), b, y, sign=np.sign(x_true))
    title = "§5.5 · nonlinear multivariate PC  g(x)=x⊙x+1"
    return res, x_true, oracle, title


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    res, x_true, oracle, title = (_run_nonlinear() if args.regime == "nonlinear"
                                  else _run_linear())

    LOG.info("regime=%s | mu*=%s | oracle=%s | ‖μ*−o‖=%.2e | converged=%s (%d it)",
             args.regime, np.round(res.mu_star, 4), np.round(oracle, 4),
             float(np.linalg.norm(res.mu_star - oracle)), res.converged, res.n_iter_run)

    fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle, title=title)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        name = (f"example_5_3_multivariate_{args.regime}.png"
                if args.regime == "nonlinear" else "example_5_3_multivariate.png")
        save_or_show(fig, out / name)
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
