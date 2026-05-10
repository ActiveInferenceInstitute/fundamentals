"""Exact Bayesian inference via grid approximation.

Grid approximation discretizes the hidden-state domain into ``G`` evenly-spaced
points and evaluates every density on that grid. Normalization is performed by
trapezoid integration. This is wasteful in high dimensions but ideal for the
pedagogical, low-dimensional examples in Part I of the book.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnostics import grid_entropy, grid_kl_divergence, logsumexp
from .distributions import normalize_density
from .generative_model import GenerativeModel

ArrayLike = np.ndarray


@dataclass
class InferenceResult:
    """Container for the output of :meth:`GridBayesianInference.infer`.

    Attributes
    ----------
    x_grid : np.ndarray
        Discretized hidden-state domain.
    prior : np.ndarray
        Prior density on ``x_grid``.
    likelihood : np.ndarray
        Likelihood (function of ``x``) on ``x_grid`` — *unnormalized*.
    posterior : np.ndarray
        Normalized posterior density on ``x_grid``.
    log_evidence : float
        Estimate of ``log p(y)`` (the model evidence) via trapezoid integration.
    """

    x_grid: np.ndarray
    prior: np.ndarray
    likelihood: np.ndarray
    posterior: np.ndarray
    log_evidence: float

    @property
    def posterior_mode(self) -> float:
        """Argmax of the posterior on ``x_grid`` (a single scalar)."""
        return float(self.x_grid[int(np.argmax(self.posterior))])

    @property
    def posterior_mean(self) -> float:
        """Trapezoid-rule expectation ``E[x] = ∫ x p(x|y) dx``."""
        return float(np.trapezoid(self.x_grid * self.posterior, self.x_grid))

    @property
    def posterior_variance(self) -> float:
        """Trapezoid-rule variance ``E[(x − E[x])²]`` of the posterior."""
        mean = self.posterior_mean
        return float(
            np.trapezoid((self.x_grid - mean) ** 2 * self.posterior, self.x_grid)
        )

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval containing ``mass`` of the posterior."""
        if not 0 < mass < 1:
            raise ValueError("mass must be in (0, 1)")
        cdf = np.concatenate(
            ([0.0], np.cumsum(0.5 * (self.posterior[1:] + self.posterior[:-1])
                               * np.diff(self.x_grid)))
        )
        cdf /= cdf[-1]
        lo = float(np.interp((1 - mass) / 2, cdf, self.x_grid))
        hi = float(np.interp(1 - (1 - mass) / 2, cdf, self.x_grid))
        return lo, hi

    def cdf(self) -> np.ndarray:
        """Empirical CDF of the posterior on ``x_grid`` (trapezoid rule)."""
        cdf = np.concatenate(
            ([0.0], np.cumsum(0.5 * (self.posterior[1:] + self.posterior[:-1])
                               * np.diff(self.x_grid)))
        )
        return cdf / cdf[-1]

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        """Posterior quantile(s) at probability ``q ∈ (0, 1)``."""
        q = np.asarray(q, dtype=float)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("all quantiles must lie in (0, 1)")
        return np.interp(q, self.cdf(), self.x_grid)

    def entropy(self) -> float:
        """Differential entropy of the posterior (nats)."""
        return grid_entropy(self.posterior, self.x_grid)

    def kl_from_prior(self) -> float:
        """``KL(posterior || prior)`` — how much the data moved the belief."""
        return grid_kl_divergence(self.posterior, self.prior, self.x_grid)

    def summary(self, ndigits: int = 4) -> str:
        """Human-readable one-line summary of the posterior."""
        return (
            f"InferenceResult(mode={round(self.posterior_mode, ndigits)}, "
            f"mean={round(self.posterior_mean, ndigits)}, "
            f"std={round(self.posterior_variance ** 0.5, ndigits)}, "
            f"H[post]={round(self.entropy(), ndigits)}, "
            f"KL[post||prior]={round(self.kl_from_prior(), ndigits)}, "
            f"log_evidence={round(self.log_evidence, ndigits)})"
        )


@dataclass
class GridBayesianInference:
    """Exact posterior inference for a 1-D continuous hidden state via a grid.

    Parameters
    ----------
    model : GenerativeModel
        Any object exposing ``log_likelihood``, ``log_prior``, and (for batch
        inference) ``log_likelihood_batch``.
    x_grid : np.ndarray
        Discretized state domain.
    """

    model: GenerativeModel
    x_grid: np.ndarray

    def __post_init__(self) -> None:
        self.x_grid = np.asarray(self.x_grid, dtype=float)
        if self.x_grid.ndim != 1 or self.x_grid.size < 2:
            raise ValueError("x_grid must be a 1-D array with at least 2 points")

    def infer(self, y: float | ArrayLike) -> InferenceResult:
        """Compute the posterior given a scalar or batch of observations.

        Notes
        -----
        Internally we work in log-space and subtract the maximum prior to
        exponentiation to avoid under- or over-flow.
        """
        y = np.asarray(y, dtype=float)
        log_prior = self.model.log_prior(self.x_grid)

        if y.ndim == 0:
            log_lik = self.model.log_likelihood(float(y), self.x_grid)
            lik = np.exp(log_lik - np.max(log_lik))  # for plotting / display
        else:
            if not hasattr(self.model, "log_likelihood_batch"):
                raise AttributeError(
                    "Model does not implement log_likelihood_batch; cannot infer "
                    "from a batch of observations."
                )
            log_lik = self.model.log_likelihood_batch(y, self.x_grid)
            lik = np.exp(log_lik - np.max(log_lik))

        log_unnorm = log_lik + log_prior
        # Numerically stable normalization via logsumexp + trapezoid rule:
        # log p(y) = log ∫ exp(log p(y, x)) dx ≈ logsumexp(log p(y, x_i)) + log Δx
        # but we use trapezoid weights for higher accuracy than a Riemann sum.
        max_log = float(np.max(log_unnorm))
        unnorm = np.exp(log_unnorm - max_log)
        evidence = np.trapezoid(unnorm, self.x_grid)
        if evidence <= 0 or not np.isfinite(evidence):
            raise ValueError(
                "posterior has non-positive or non-finite mass; "
                "check that the grid covers regions of appreciable likelihood"
            )
        posterior = unnorm / evidence
        log_evidence = float(max_log + np.log(evidence))

        # Prior in linear space (always returned normalized for plotting).
        prior_density = np.exp(log_prior - np.max(log_prior))
        prior_density = normalize_density(prior_density, self.x_grid)

        return InferenceResult(
            x_grid=self.x_grid,
            prior=prior_density,
            likelihood=lik,
            posterior=posterior,
            log_evidence=log_evidence,
        )

    def joint_density(
        self,
        y_grid: Optional[np.ndarray] = None,
        sigma2_y: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``p(x, y)`` on a 2-D state-observation grid.

        Used by the visualizations in §2.4 (joint heatmaps).
        """
        if not hasattr(self.model, "predict_mean"):
            raise AttributeError("model must expose predict_mean()")
        if y_grid is None:
            mu = self.model.predict_mean(self.x_grid)
            sigma = float(np.sqrt(getattr(self.model, "sigma2_y", sigma2_y or 1.0)))
            y_grid = np.linspace(
                float(np.min(mu) - 4 * sigma),
                float(np.max(mu) + 4 * sigma),
                self.x_grid.size,
            )
        y_grid = np.asarray(y_grid, dtype=float)

        log_prior = self.model.log_prior(self.x_grid)            # (G,)
        # Likelihood for every (x, y) pair.
        # Build by evaluating model.log_likelihood for each y.
        joint_log = np.empty((y_grid.size, self.x_grid.size), dtype=float)
        for i, y in enumerate(y_grid):
            joint_log[i] = self.model.log_likelihood(float(y), self.x_grid) + log_prior

        joint = np.exp(joint_log - np.max(joint_log))
        # Normalize over the 2-D grid using trapezoidal integration.
        marg_x = np.trapezoid(joint, y_grid, axis=0)
        total = np.trapezoid(marg_x, self.x_grid)
        joint /= total
        return self.x_grid, y_grid, joint
