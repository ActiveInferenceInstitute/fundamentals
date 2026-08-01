"""§5.6 — Parameterized multivariate predictive coding (Fig. 5.3.5).

Run:: ``python chapters/chapter_05/example_5_6_parameterized.py [--save] [--regime recover|informative]``

The book's parameterized generative model drives a **rectangular** mixing matrix
``Θ ∈ R^{4×2}`` with the *nonlinear* element-wise-square generating function

    g(x) = Θ (x ⊙ x) + b,   x ∈ R²,  y ∈ R⁴

so a 2-D hidden state is observed through 4 over-determined channels (book §5.6,
``Θ = [[-0.1, 0.3], [0.3, 0.4], [0.2, -0.5], [-0.1, 0.1]]``, ``x* = [0.5, 2.5]``).
This is the faithful multivariate case: unlike the linear/​square companion
``example_5_3_multivariate.py``, ``g`` here is nonlinear *and* the observation space is
larger than the state space, so the recognition dynamics must invert a genuinely
over-determined nonlinear map.

Two regimes (``--regime``):

* **recover** (default) — a near-flat state prior isolates the likelihood, so the
  fixed point is the maximum-likelihood inverse. Because ``g`` depends on ``x`` only
  through ``u = x⊙x``, a noiseless observation gives the exact least-squares recovery
  ``x* = sign ⊙ √(Θ⁺(y−b))``. This closed form is computed independently by
  :func:`~active_inference.pc_parameterized_lstsq_oracle` and cross-checked against
  the iterate (‖μ*−oracle‖ ≈ 0), the nonlinear/over-determined analogue of the
  Chapter-4 grid oracle.
* **informative** — the book's precision (``Σ_x = Σ_y = 0.5 I`` → precision 2) keeps the prior in
  play, so the MAP belief settles between the data-consistent state and ``m_x = [1, 1]``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    get_logger,
    multivariate_predictive_coding,
    pc_parameterized_lstsq_oracle,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_multivariate_pc, save_or_show

LOG = get_logger("ch5.ex6")

# Book §5.6 parameterized model (4×2 mixing matrix, element-wise-square drive).
THETA = np.array(
    [[-0.1, 0.3], [0.3, 0.4], [0.2, -0.5], [-0.1, 0.1]]
)
OFFSET = np.ones(4)
X_TRUE = np.array([0.5, 2.5])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--regime", choices=("recover", "informative"), default="recover",
                   help="near-flat likelihood recovery (default) or the book's MAP prior")
    p.add_argument("--kappa", type=float, default=0.05, help="gradient step size")
    p.add_argument("--n-iter", type=int, default=20000, help="max recognition iterations")
    p.add_argument("--m-x", type=float, nargs=2, default=(1.0, 1.0),
                   help="prior mean m_x (book uses [1, 1])")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()

    def g(x: np.ndarray) -> np.ndarray:
        """Parameterized nonlinear generating function g(x)=Θ(x⊙x)+b."""
        return THETA @ (x * x) + OFFSET

    def jacobian(x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian ∂g/∂x = Θ · diag(2x) ∈ R^{4×2}."""
        return THETA @ np.diag(2.0 * x)

    y = g(X_TRUE)  # noiseless, self-consistent observation
    m_x = np.asarray(args.m_x, dtype=float)
    precision_x = np.array([1e-6, 1e-6]) if args.regime == "recover" else np.array([2.0, 2.0])
    precision_y = np.array([2.0, 2.0, 2.0, 2.0])

    res = multivariate_predictive_coding(
        g=g, jacobian=jacobian, y=y, m_x=m_x,
        precision_y=precision_y, precision_x=precision_x,
        mu0=np.sign(X_TRUE) * np.abs(m_x),  # correct orthant: g is even in each x_c
        kappa=args.kappa, n_iter=args.n_iter, tol=1e-14,
    )

    # Independent recovery oracle (exact under the near-flat prior); with the
    # informative prior it is the pure-likelihood reference the MAP belief departs from.
    oracle = pc_parameterized_lstsq_oracle(THETA, OFFSET, y, sign=np.sign(X_TRUE))
    residual = float(np.linalg.norm(g(res.mu_star) - y))
    LOG.info(
        "regime=%s | mu*=%s | lstsq oracle=%s | ‖μ*−o‖=%.2e | ‖g(μ*)−y‖=%.2e | conv=%s (%d it)",
        args.regime, np.round(res.mu_star, 4), np.round(oracle, 4),
        float(np.linalg.norm(res.mu_star - oracle)), residual,
        res.converged, res.n_iter_run,
    )
    if args.regime == "recover":
        LOG.info("recovery check: ‖μ*−x*‖=%.2e (should be ~0)",
                 float(np.linalg.norm(res.mu_star - X_TRUE)))

    title = ("§5.6 · parameterized PC (Θ 4×2, near-flat prior → ML recovery)"
             if args.regime == "recover"
             else "§5.6 · parameterized PC (Θ 4×2, informative prior → MAP)")
    fig = plot_multivariate_pc(res, truth=X_TRUE, oracle=oracle, title=title)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / f"example_5_6_parameterized_{args.regime}.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
