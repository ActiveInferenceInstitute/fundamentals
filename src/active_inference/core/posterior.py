"""A common protocol for the package's three posterior types.

Three posterior dataclasses ship in the package:

- :class:`InferenceResult` (1-D grid posterior over a hidden state)
- :class:`LGSPosterior` (multivariate Gaussian posterior over a hidden state)
- :class:`BLRPosterior` (Gaussian posterior over a parameter vector)

Their internal representations differ — a grid versus a `(mean, cov)`
pair — but their *consumer interface* should not. This module defines a
:class:`Posterior` protocol that captures the methods every posterior
class implements, and a :func:`summarize_posterior` helper that
consumes any of them through that protocol.

Why a Protocol and not a base class?
------------------------------------

Each posterior is a frozen ``@dataclass`` so a base class would create
inheritance friction (MRO, ``__post_init__`` ordering). The Protocol
only describes the *shape* of the seam, not its inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Posterior(Protocol):
    """Structural interface implemented by every posterior in the package.

    A *Posterior* exposes:

    - ``summary(ndigits: int = 4) -> str`` — human-readable one-liner.

    Subtypes additionally provide either ``credible_interval(mass)`` (for
    1-D posteriors) or ``mean`` + ``cov`` attributes (for Gaussian
    posteriors). The consumer helpers below dispatch on whichever
    interface is present, so chapter scripts can write code that works
    uniformly against any posterior.
    """

    def summary(self, ndigits: int = 4) -> str:
        """Human-readable one-line summary; required by the Posterior protocol."""
        ...


def has_credible_interval(p: object) -> bool:
    """True if ``p`` exposes a ``credible_interval`` method (i.e. 1-D grid)."""
    return callable(getattr(p, "credible_interval", None))


def has_mean_cov(p: object) -> bool:
    """True if ``p`` carries Gaussian ``mean`` + ``cov`` attributes."""
    return hasattr(p, "mean") and hasattr(p, "cov")


def posterior_mean(p: object) -> np.ndarray | float:
    """Return the posterior mean of any package Posterior.

    Dispatches:
    - 1-D grid posterior → ``p.posterior_mean`` (scalar)
    - Gaussian posterior → ``p.mean`` (1-D array)
    """
    if hasattr(p, "posterior_mean"):
        return float(getattr(p, "posterior_mean"))
    if hasattr(p, "mean"):
        return np.asarray(getattr(p, "mean"), dtype=float)
    raise AttributeError(
        f"object of type {type(p).__name__} does not expose a posterior mean"
    )


def posterior_std(p: object) -> np.ndarray | float:
    """Return the posterior standard deviation in a uniform way.

    For grid posteriors this is ``sqrt(posterior_variance)``; for Gaussian
    posteriors it is the diagonal of `cov` square-rooted.
    """
    if hasattr(p, "posterior_variance"):
        return float(np.sqrt(getattr(p, "posterior_variance")))
    if hasattr(p, "std") and callable(getattr(p, "std")):
        return np.asarray(p.std(), dtype=float)
    if hasattr(p, "cov"):
        return np.sqrt(np.diag(np.asarray(p.cov, dtype=float)))
    raise AttributeError(
        f"object of type {type(p).__name__} does not expose a posterior std"
    )


def summarize_posterior(p: Posterior, ndigits: int = 4) -> str:
    """Single entry-point that gives a one-line summary of any package posterior."""
    return p.summary(ndigits=ndigits)
