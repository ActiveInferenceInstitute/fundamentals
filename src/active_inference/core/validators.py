"""Defensive runtime validators used at module / class boundaries.

These helpers consolidate the repetitive ``if x <= 0: raise ValueError``
patterns that would otherwise scatter through every estimator / model /
process. Each validator returns the (coerced) input on success so it can
be chained inline::

    sigma2_y = require_positive_scalar(sigma2_y, name="sigma2_y")

The aim is *consistent error messages* and a single place to change
validation policy. Type-level invariants (covariance shape, probability
mass) live in :mod:`active_inference.core.types`; this module covers
the simpler scalar / array / shape checks the type aliases don't reach.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def require_positive_scalar(x: float, *, name: str = "value") -> float:
    """Assert ``x > 0`` and finite; return ``x`` cast to float."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x!r}")
    return x


def require_non_negative_scalar(x: float, *, name: str = "value") -> float:
    """Assert ``x >= 0`` and finite; return ``x`` cast to float."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if x < 0:
        raise ValueError(f"{name} must be ≥ 0, got {x!r}")
    return x


def require_in_unit_interval(
    x: float,
    *,
    name: str = "value",
    inclusive: bool = False,
) -> float:
    """Assert ``x ∈ (0, 1)`` (or ``[0, 1]`` when ``inclusive=True``)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if inclusive:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1], got {x!r}")
    else:
        if not (0.0 < x < 1.0):
            raise ValueError(f"{name} must lie in (0, 1), got {x!r}")
    return x


def require_int_at_least(x: int, *, minimum: int = 1,
                          name: str = "value") -> int:
    """Assert ``x >= minimum`` and integer-typed."""
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(x).__name__}")
    x = int(x)
    if x < minimum:
        raise ValueError(f"{name} must be ≥ {minimum}, got {x}")
    return x


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def require_finite_array(arr, *, name: str = "array") -> np.ndarray:
    """Coerce to ``np.ndarray`` (float) and assert all entries are finite."""
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries")
    return arr


def require_1d(arr, *, length: Optional[int] = None,
                name: str = "array") -> np.ndarray:
    """Assert ``arr`` is 1-D, optionally with a specific length."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if length is not None and arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    return arr


def require_2d(arr, *, shape: Optional[tuple[int, int]] = None,
                name: str = "array") -> np.ndarray:
    """Assert ``arr`` is 2-D, optionally with a specific shape."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {arr.ndim}-D")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def require_same_length(*arrays, names: Optional[Sequence[str]] = None) -> None:
    """Assert all positional arrays share their leading dimension."""
    arrays = tuple(np.asarray(a) for a in arrays)
    if not arrays:
        return
    if names is None:
        names = [f"arg{i}" for i in range(len(arrays))]
    n = arrays[0].shape[0]
    for arr, nm in zip(arrays[1:], names[1:]):
        if arr.shape[0] != n:
            raise ValueError(
                f"length mismatch: {names[0]} has {n} rows, "
                f"{nm} has {arr.shape[0]}"
            )


def require_monotone(arr, *, increasing: bool = True,
                     strict: bool = False,
                     name: str = "array") -> np.ndarray:
    """Assert a 1-D array is monotone (increasing or decreasing)."""
    arr = require_1d(arr, name=name)
    diffs = np.diff(arr)
    if increasing:
        ok = np.all(diffs > 0) if strict else np.all(diffs >= 0)
        direction = "strictly increasing" if strict else "non-decreasing"
    else:
        ok = np.all(diffs < 0) if strict else np.all(diffs <= 0)
        direction = "strictly decreasing" if strict else "non-increasing"
    if not ok:
        raise ValueError(f"{name} must be {direction}")
    return arr


# ---------------------------------------------------------------------------
# Composite (used by estimators / models)
# ---------------------------------------------------------------------------


def require_design_matrix(
    X,
    *,
    n_features: Optional[int] = None,
    n_samples: Optional[int] = None,
    name: str = "X",
) -> np.ndarray:
    """Validate a regression / FA design matrix.

    Coerces to 2-D and optionally checks number of features / samples.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {X.ndim}-D")
    if n_samples is not None and X.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have {n_samples} rows, got {X.shape[0]}"
        )
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"{name} must have {n_features} columns, got {X.shape[1]}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite entries")
    return X
