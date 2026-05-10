"""Composition helpers that chain process → model → inference → diagnostics.

The package keeps process / model / inference / diagnostics in separate
modules with clean interfaces. This module provides a single fluent
entry point that wires them together for the most common case: a 1-D
linear-Gaussian agent simulating a 1-D environment, performing exact
grid Bayesian inference, and computing standard diagnostics.

Two pieces ship here:

- :class:`Pipeline` — declarative configuration of a full simulation.
- :func:`running_stats` — incremental posterior statistics for a stream
  of observations, useful for animations and convergence plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnostics import grid_kl_divergence
from .generative_model import GenerativeModel, LinearGaussianModel
from .generative_process import GenerativeProcess, LinearGaussianProcess
from .inference import GridBayesianInference, InferenceResult
from ..utils.grids import make_grid


@dataclass
class Pipeline:
    """A simulation + inference pipeline tied to a single grid.

    Parameters
    ----------
    process : GenerativeProcess
        The environment that produces observations.
    model : GenerativeModel
        The agent's beliefs about that environment.
    x_grid : np.ndarray
        Discretized hidden-state domain.

    Methods
    -------
    sample(x_star, n)
        Forward — draw ``n`` observations from the process.
    infer(y)
        Backward — return the grid Bayesian posterior.
    run(x_star, n)
        ``sample`` then ``infer`` in one call. Returns the
        :class:`InferenceResult` with the posterior, mode, mean, std,
        credible interval, KL, and entropy.
    """

    process: GenerativeProcess
    model: GenerativeModel
    x_grid: np.ndarray

    def __post_init__(self) -> None:
        self.x_grid = np.asarray(self.x_grid, dtype=float)
        if self.x_grid.ndim != 1 or self.x_grid.size < 2:
            raise ValueError("x_grid must be a 1-D array with ≥ 2 points")
        self._inferer = GridBayesianInference(self.model, self.x_grid)

    def sample(self, x_star: float, n: int = 1) -> np.ndarray:
        """Forward pass — draw ``n`` observations at ``x_star``."""
        return self.process.sample(x_star, n=n)

    def infer(self, y) -> InferenceResult:
        """Backward pass — exact posterior on ``self.x_grid``."""
        return self._inferer.infer(y)

    def run(self, x_star: float, n: int = 1) -> InferenceResult:
        """Sample ``n`` observations at ``x_star`` and return the posterior."""
        ys = self.sample(x_star, n=n)
        return self.infer(np.asarray(ys).flatten())

    @classmethod
    def linear_gaussian(
        cls,
        *,
        beta0: float, beta1: float,
        sigma2_y: float,
        m_x: float, s2_x: float,
        x_low: float = 0.0, x_high: float = 5.0, n_grid: int = 500,
        prior_kind: str = "gaussian",
        rng: Optional[np.random.Generator] = None,
    ) -> "Pipeline":
        """Pre-configured Pipeline for the canonical linear-Gaussian setup.

        Mirrors the matched process / model used throughout Chapters 1–3.
        """
        process = LinearGaussianProcess(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y, rng=rng,
        )
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            m_x=m_x, s2_x=s2_x, prior_kind=prior_kind,
        )
        return cls(
            process=process, model=model,
            x_grid=make_grid(x_low, x_high, n_grid),
        )


@dataclass
class RunningPosteriorStats:
    """Per-step traces produced by :func:`running_stats`."""

    n_axis: np.ndarray            # shape (N,) — 1, 2, ..., N
    means: np.ndarray             # shape (N,)
    stds: np.ndarray              # shape (N,)
    kl_from_prior: np.ndarray     # shape (N,)
    log_evidences: np.ndarray     # shape (N,) — cumulative log p(y_{1:i})
    posteriors: np.ndarray        # shape (N, G) — full density per step

    def summary(self, ndigits: int = 4) -> str:
        """One-line readout of the final running posterior moments + KL."""
        return (
            f"RunningPosteriorStats(N={self.n_axis.size}, "
            f"final_mean={round(float(self.means[-1]), ndigits)}, "
            f"final_std={round(float(self.stds[-1]), ndigits)}, "
            f"final_KL={round(float(self.kl_from_prior[-1]), ndigits)})"
        )


def running_stats(
    model: GenerativeModel,
    x_grid: np.ndarray,
    samples: np.ndarray,
) -> RunningPosteriorStats:
    """Compute the posterior moments and KL after each new observation.

    Walks the i.i.d. observations one at a time and accumulates the
    log-posterior in place. Returns a :class:`RunningPosteriorStats`
    record suitable for both static convergence plots and the
    ``animate_sufficient_statistics`` helper.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size < 2:
        raise ValueError("x_grid must be a 1-D array with ≥ 2 points")
    log_prior = model.log_prior(x_grid).copy()
    prior_density = np.exp(log_prior - np.max(log_prior))
    prior_density /= np.trapezoid(prior_density, x_grid)

    log_state = log_prior.copy()
    n = samples.size
    means = np.empty(n)
    stds = np.empty(n)
    kls = np.empty(n)
    log_evidences = np.empty(n)
    posteriors = np.empty((n, x_grid.size))
    cumulative_log_evidence = 0.0

    for i, y in enumerate(samples):
        # Incremental log-posterior; track log p(y_i | y_{1:i-1}) for evidence.
        delta_log_lik = model.log_likelihood(float(y), x_grid)
        log_state = log_state + delta_log_lik
        max_log = float(np.max(log_state))
        unnorm = np.exp(log_state - max_log)
        evidence_chunk = float(np.trapezoid(unnorm, x_grid))
        if evidence_chunk <= 0 or not np.isfinite(evidence_chunk):
            raise ValueError(
                "running posterior collapsed to zero mass; widen x_grid"
            )
        density = unnorm / evidence_chunk
        cumulative_log_evidence += max_log + float(np.log(evidence_chunk))
        # Renormalize log_state so the running sum stays bounded.
        log_state = np.log(np.maximum(density, np.finfo(float).tiny))

        m = float(np.trapezoid(x_grid * density, x_grid))
        v = float(np.trapezoid((x_grid - m) ** 2 * density, x_grid))
        means[i] = m
        stds[i] = float(np.sqrt(max(v, 0.0)))
        kls[i] = grid_kl_divergence(density, prior_density, x_grid)
        log_evidences[i] = cumulative_log_evidence
        posteriors[i] = density

    return RunningPosteriorStats(
        n_axis=np.arange(1, n + 1),
        means=means, stds=stds, kl_from_prior=kls,
        log_evidences=log_evidences, posteriors=posteriors,
    )
