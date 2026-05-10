"""A small, reusable gradient-descent loop with optional analytic gradients.

The chapter orchestrators use this for both MLE and MAP, with either an
analytically derived gradient (fast and exact for the linear-Gaussian case) or
a numerical finite-difference fallback (works for arbitrary loss callables).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class GradientDescentResult:
    """Output of :func:`gradient_descent`.

    Attributes
    ----------
    x : float
        Final iterate.
    history : np.ndarray, shape ``(K + 1,)``
        Iterate at each step (including the initial value).
    losses : np.ndarray, shape ``(K + 1,)``
        Loss value at each iterate.
    n_iterations : int
        Number of completed iterations (``K``).
    converged : bool
        Whether the loop terminated due to the convergence threshold instead
        of exhausting ``max_iter``.
    """

    x: float
    history: np.ndarray
    losses: np.ndarray
    n_iterations: int
    converged: bool


def _numerical_gradient(
    loss_fn: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    return (loss_fn(x + h) - loss_fn(x - h)) / (2.0 * h)


def gradient_descent(
    loss_fn: Callable[[float], float],
    x0: float,
    learning_rate: float,
    max_iter: int = 100,
    grad_fn: Optional[Callable[[float], float]] = None,
    tol: float = 1e-8,
    record_history: bool = True,
) -> GradientDescentResult:
    """Minimize a scalar function of one variable.

    Parameters
    ----------
    loss_fn : callable
        Returns the loss at a scalar ``x``.
    x0 : float
        Initial iterate.
    learning_rate : float
        Step size ``kappa``.
    max_iter : int
        Maximum number of gradient steps.
    grad_fn : callable, optional
        Analytic gradient of ``loss_fn``. If ``None``, finite differences are
        used instead.
    tol : float
        Convergence threshold on |x_{k+1} - x_k|.
    record_history : bool
        If ``False``, only the final iterate and loss are stored — useful for
        tight loops.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    g = grad_fn if grad_fn is not None else (lambda x: _numerical_gradient(loss_fn, x))

    x = float(x0)
    history = [x]
    losses = [float(loss_fn(x))]
    converged = False

    for k in range(max_iter):
        grad = float(g(x))
        x_next = x - learning_rate * grad
        if record_history:
            history.append(x_next)
            losses.append(float(loss_fn(x_next)))
        if abs(x_next - x) < tol:
            x = x_next
            converged = True
            break
        x = x_next

    return GradientDescentResult(
        x=x,
        history=np.asarray(history, dtype=float),
        losses=np.asarray(losses, dtype=float),
        n_iterations=k + 1,
        converged=converged,
    )
