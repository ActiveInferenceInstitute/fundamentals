"""Convenience constructors for the discrete grids used in approximate inference."""

from __future__ import annotations

import numpy as np


def make_grid(low: float, high: float, n_points: int = 500) -> np.ndarray:
    """Evenly-spaced 1-D grid on ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Inclusive endpoints. Must satisfy ``low < high``.
    n_points : int
        Number of grid nodes.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("low and high must be finite")
    if high <= low:
        raise ValueError("high must be greater than low")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    return np.linspace(float(low), float(high), int(n_points))


def make_2d_grid(
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    n_x: int = 200,
    n_y: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Two coordinate arrays defining a rectangular grid for joint densities."""
    x = make_grid(x_low, x_high, n_x)
    y = make_grid(y_low, y_high, n_y)
    return x, y
