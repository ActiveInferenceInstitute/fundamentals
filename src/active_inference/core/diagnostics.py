"""Statistical diagnostics — entropies, divergences, scoring rules, calibration.

These helpers are model-agnostic: they take densities (on a grid) or arrays
of samples and produce scalar / curve diagnostics that sharpen what an
inference run is actually doing.

Conventions
-----------
- All "log" helpers work in natural log (nats), not bits, matching the rest
  of the codebase. Multiply by ``1 / np.log(2)`` if you want bits.
- Grid-based KL/entropy use the trapezoid rule for consistency with
  :func:`active_inference.core.distributions.normalize_density`.
- Multivariate Gaussian routines never invert the covariance directly —
  they Cholesky-solve, matching ``mvn_log_pdf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .distributions import gaussian_log_pdf

ArrayLike = np.ndarray
_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------


def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Numerically stable ``log(sum(exp(a)))``.

    Subtracts the max along ``axis`` before exponentiating to avoid overflow.
    All-(-inf) slices return ``-inf`` without raising or warning.
    """
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    if axis is None:
        return float(np.squeeze(out))
    return np.squeeze(out, axis=axis)


def effective_sample_size(log_weights: np.ndarray) -> float:
    """Kish's effective sample size from log-importance-weights.

    ``ESS = (Σ w_i)² / Σ w_i²``, computed in log-space for stability.
    Returns a scalar in ``[1, len(log_weights)]``.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1:
        raise ValueError("log_weights must be 1-D")
    log_num = 2.0 * logsumexp(log_weights)
    log_den = logsumexp(2.0 * log_weights)
    return float(np.exp(log_num - log_den))


# ---------------------------------------------------------------------------
# Entropy and divergence on a 1-D grid
# ---------------------------------------------------------------------------


def grid_entropy(p: np.ndarray, x_grid: np.ndarray) -> float:
    """Differential entropy ``-∫ p log p dx`` via the trapezoid rule.

    ``p`` should already be a normalized density on ``x_grid``. Bins where
    ``p ≤ 0`` contribute zero (consistent with the standard convention
    ``0 · log 0 = 0``).
    """
    p = np.asarray(p, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if p.shape != x_grid.shape:
        raise ValueError("p and x_grid must share shape")
    integrand = np.zeros_like(p)
    mask = p > 0
    integrand[mask] = -p[mask] * np.log(p[mask])
    return float(np.trapezoid(integrand, x_grid))


def grid_kl_divergence(p: np.ndarray, q: np.ndarray, x_grid: np.ndarray) -> float:
    """Numerically stable ``KL(p || q)`` for densities on a shared 1-D grid.

    Bins where ``p == 0`` contribute zero. Bins where ``q == 0`` while
    ``p > 0`` produce ``+inf`` (KL is undefined there).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if p.shape != q.shape != x_grid.shape:
        raise ValueError("p, q, x_grid must share shape")
    integrand = np.zeros_like(p)
    pos = p > 0
    if np.any(pos & (q <= 0)):
        return float("inf")
    integrand[pos] = p[pos] * (np.log(p[pos]) - np.log(q[pos]))
    return float(np.trapezoid(integrand, x_grid))


# ---------------------------------------------------------------------------
# Closed-form Gaussian diagnostics
# ---------------------------------------------------------------------------


def gaussian_entropy_univariate(sigma2: float) -> float:
    """Differential entropy of a univariate Gaussian: ``½ log(2π e σ²)``."""
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    return 0.5 * (_LOG_2PI + 1.0 + float(np.log(sigma2)))


def gaussian_kl_univariate(
    mu_p: float, sigma2_p: float, mu_q: float, sigma2_q: float,
) -> float:
    """Closed-form ``KL(N(mu_p, σ²_p) || N(mu_q, σ²_q))``.

    ``½ [ log(σ²_q / σ²_p) + (σ²_p + (μ_p − μ_q)²) / σ²_q − 1 ]``.
    """
    if sigma2_p <= 0 or sigma2_q <= 0:
        raise ValueError("variances must be positive")
    return 0.5 * (
        np.log(sigma2_q / sigma2_p)
        + (sigma2_p + (mu_p - mu_q) ** 2) / sigma2_q
        - 1.0
    )


def gaussian_entropy_mvn(cov: np.ndarray) -> float:
    """Differential entropy of an MVN: ``½ log((2π e)^d |Σ|)``."""
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[0]
    L = np.linalg.cholesky(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (d * (_LOG_2PI + 1.0) + log_det)


def gaussian_kl_mvn(
    mu_p: np.ndarray, cov_p: np.ndarray,
    mu_q: np.ndarray, cov_q: np.ndarray,
) -> float:
    """Closed-form KL for two MVNs of the same dimension."""
    mu_p = np.asarray(mu_p, dtype=float).reshape(-1)
    mu_q = np.asarray(mu_q, dtype=float).reshape(-1)
    cov_p = np.asarray(cov_p, dtype=float)
    cov_q = np.asarray(cov_q, dtype=float)
    d = mu_p.size
    if cov_p.shape != (d, d) or cov_q.shape != (d, d) or mu_q.size != d:
        raise ValueError("dimensions of mu / cov must be consistent")
    # Solve, never invert: Σq⁻¹ cov_p (for the trace) and Σq⁻¹ diff (for the
    # quadratic form) come from triangular solves against the Cholesky factor.
    L_p = np.linalg.cholesky(cov_p)
    L_q = np.linalg.cholesky(cov_q)
    log_det_p = 2.0 * np.sum(np.log(np.diag(L_p)))
    log_det_q = 2.0 * np.sum(np.log(np.diag(L_q)))
    from scipy.linalg import solve_triangular

    diff = mu_q - mu_p
    z_diff = solve_triangular(L_q, diff, lower=True)
    quad = float(z_diff @ z_diff)
    # trace(Σq⁻¹ Σp): solve Σq Z = Σp column-wise, then sum the diagonal.
    Z = solve_triangular(L_q, solve_triangular(L_q, cov_p, lower=True), lower=False)
    trace_term = float(np.trace(Z))
    return 0.5 * (
        trace_term
        + quad
        - d
        + (log_det_q - log_det_p)
    )


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


def log_score_gaussian(
    y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray,
) -> np.ndarray:
    """Log score (negative NLL) for a sequence of Gaussian forecasts.

    Higher is better. Equal to ``log p(y | mu, σ²)`` evaluated pointwise.
    """
    return gaussian_log_pdf(y, mu, sigma2)


def crps_gaussian(
    y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray,
) -> np.ndarray:
    """Continuous-ranked probability score for a Gaussian forecast.

    Closed form (Gneiting & Raftery, 2007)::

        CRPS = σ [ z (2 Φ(z) − 1) + 2 φ(z) − 1/√π ]
        z    = (y − μ) / σ

    Lower is better; CRPS == log_score_gaussian when σ² is well calibrated.
    """
    from math import erf, sqrt
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    sigma = np.sqrt(sigma2)
    z = (y - mu) / sigma
    # Φ via erf:  Φ(z) = ½ (1 + erf(z / √2))
    erfvec = np.vectorize(erf)
    Phi = 0.5 * (1.0 + erfvec(z / sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Calibration / coverage from a credible-interval estimator
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCurve:
    """Result of a calibration sweep.

    Attributes
    ----------
    nominal : np.ndarray
        Target credible-interval masses (e.g. 0.05, 0.10, ..., 0.95).
    empirical : np.ndarray
        Fraction of trials whose credible interval at that nominal mass
        actually contained the truth.
    n_trials : int
        Number of independent trials used.
    """

    nominal: np.ndarray
    empirical: np.ndarray
    n_trials: int

    def calibration_error(self) -> float:
        """L1 expected calibration error: ``mean |empirical − nominal|``."""
        return float(np.mean(np.abs(self.empirical - self.nominal)))


def coverage_from_intervals(
    truths: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> float:
    """Return empirical credible-interval coverage after matching shape validation."""
    truths = np.asarray(truths)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    if not (truths.shape == lows.shape == highs.shape):
        raise ValueError("truths, lows, highs must share shape")
    return float(np.mean((truths >= lows) & (truths <= highs)))


def calibration_curve(
    truths: np.ndarray,
    lower_fn,
    upper_fn,
    nominal_levels: np.ndarray,
) -> CalibrationCurve:
    """Compute empirical coverage at a sweep of nominal levels.

    Parameters
    ----------
    truths : (T,)
        True values of the latent quantity per trial.
    lower_fn, upper_fn : callable(level: float) -> (T,) array
        Functions returning the lower / upper edges of the credible
        interval at the given nominal mass for each trial.
    nominal_levels : array-like
        Nominal masses (in (0, 1)) at which to probe coverage.
    """
    nominal = np.asarray(nominal_levels, dtype=float)
    if np.any((nominal <= 0) | (nominal >= 1)):
        raise ValueError("all nominal levels must lie in (0, 1)")
    empirical = np.empty_like(nominal)
    for i, level in enumerate(nominal):
        empirical[i] = coverage_from_intervals(
            truths, lower_fn(float(level)), upper_fn(float(level)),
        )
    return CalibrationCurve(
        nominal=nominal,
        empirical=empirical,
        n_trials=int(np.asarray(truths).size),
    )


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------


@dataclass
class PosteriorPredictiveCheck:
    """Two-sided posterior-predictive p-value for a chosen test statistic."""

    observed: float
    replicated: np.ndarray
    p_value: float

    def summary(self, ndigits: int = 4) -> str:
        """One-line readout of the observed statistic and two-sided p-value."""
        return (
            f"PosteriorPredictiveCheck(observed={round(self.observed, ndigits)}, "
            f"p_value={round(self.p_value, ndigits)})"
        )


def posterior_predictive_check(
    observed: np.ndarray,
    replicated_samples: np.ndarray,
    statistic,
) -> PosteriorPredictiveCheck:
    """Compute a Bayesian posterior-predictive p-value.

    Parameters
    ----------
    observed : (N,) array
        The actual data set.
    replicated_samples : (M, N) array
        ``M`` replicated data sets drawn from the posterior predictive.
    statistic : callable(array) -> float
        A summary statistic (e.g. mean, variance, range).

    Returns
    -------
    PosteriorPredictiveCheck
        Two-sided p-value computed as
        ``2 · min(P(T_rep ≥ T_obs), P(T_rep ≤ T_obs))``.
    """
    observed = np.asarray(observed)
    replicated_samples = np.asarray(replicated_samples)
    if replicated_samples.ndim != 2:
        raise ValueError("replicated_samples must be (M, N)")
    t_obs = float(statistic(observed))
    t_rep = np.array([float(statistic(rep)) for rep in replicated_samples])
    p_upper = float(np.mean(t_rep >= t_obs))
    p_lower = float(np.mean(t_rep <= t_obs))
    p_value = float(2.0 * min(p_upper, p_lower))
    return PosteriorPredictiveCheck(t_obs, t_rep, p_value)


# ---------------------------------------------------------------------------
# Helpers for common workflows
# ---------------------------------------------------------------------------


def normal_ci(mean: float, sigma2: float, level: float = 0.95) -> Tuple[float, float]:
    """Equal-tailed credible interval for a univariate Gaussian.

    Uses ``scipy.special.erfinv`` for the inverse CDF.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    from scipy.special import erfinv
    half = float(np.sqrt(2.0 * sigma2) * erfinv(level))
    return mean - half, mean + half


def standardize(samples: np.ndarray) -> np.ndarray:
    """Z-score samples columnwise: ``(x − mean) / std``."""
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean(axis=0, keepdims=True)
    std = samples.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (samples - mean) / std


# ---------------------------------------------------------------------------
# Optimization / inference validation (used by the variational + predictive
# coding layers, Chapters 4–5)
# ---------------------------------------------------------------------------


def gradient_check(
    f: Callable[[float], float],
    grad: Callable[[float], float],
    x0: float,
    *,
    eps: float = 1e-5,
) -> float:
    r"""Max abs error between an analytic gradient and a central finite difference.

    Verifies that ``grad(x0)`` matches ``(f(x0+eps) − f(x0−eps)) / (2 eps)``. This is
    the tool that lets us *derive* gradients (e.g. predictive coding's recognition
    dynamics) and prove the sign/scale are right rather than transcribing them.
    Returns ``|analytic − numeric|`` (a single scalar; ``< 1e-5`` is a pass for the
    smooth losses in this package).
    """
    numeric = (float(f(x0 + eps)) - float(f(x0 - eps))) / (2.0 * eps)
    return abs(float(grad(x0)) - numeric)


def gradient_check_vector(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    r"""Max abs error between an analytic vector gradient and central finite differences.

    The vector analogue of :func:`gradient_check`: for a scalar field
    ``f: R^n → R`` it perturbs each coordinate independently and compares the
    analytic ``grad(x0) ∈ R^n`` against the central difference
    ``(f(x0 + eps e_i) − f(x0 − eps e_i)) / (2 eps)``.

    This is what lets us *prove* the sign and scale of the multivariate recognition
    gradient ``∂F/∂μ = Π_x ε_x − Jᵀ Π_y ε_y`` (book §5.3 Eq. 21) numerically rather
    than transcribing it — including for over-determined, nonlinear generating
    functions where no closed-form fixed point exists. Returns
    ``max_i |analytic_i − numeric_i|`` (``< 1e-5`` is a pass for the smooth losses in
    this package).
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    analytic = np.atleast_1d(np.asarray(grad(x0), dtype=float))
    numeric = np.empty_like(x0)
    for i in range(x0.size):
        step = np.zeros_like(x0)
        step[i] = eps
        numeric[i] = (float(f(x0 + step)) - float(f(x0 - step))) / (2.0 * eps)
    return float(np.max(np.abs(analytic - numeric)))


@dataclass(frozen=True)
class ConvergenceReport:
    """Summary of a monotone-descent trace (free energy / loss over iterations).

    Attributes
    ----------
    monotone : bool
        Whether the trace is non-increasing to within ``tol`` at every step.
    final : float
        Last value of the trace.
    total_decrease : float
        ``trace[0] − trace[-1]`` (≥ 0 for a descent).
    n_steps : int
        Number of steps (``len(trace) − 1``).
    rate : float
        Empirical linear-convergence factor — the geometric mean of successive
        residual ratios ``(t[k+1] − t∞)/(t[k] − t∞)`` (``nan`` if undefined). A value
        in ``(0, 1)`` indicates linear convergence; smaller is faster.
    max_increase : float
        Largest single-step increase (0.0 for a perfectly monotone trace).
    """

    monotone: bool
    final: float
    total_decrease: float
    n_steps: int
    rate: float
    max_increase: float


def convergence_report(trace, *, tol: float = 1e-9) -> ConvergenceReport:
    """Diagnose a descent ``trace`` (e.g. ``result.free_energies``).

    Checks monotonicity, total decrease, and estimates the empirical convergence
    rate against the final value. Works for any 1-D sequence of losses.
    """
    t = np.asarray(trace, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("trace must have at least two entries")
    diffs = np.diff(t)
    max_increase = float(max(0.0, diffs.max()))
    monotone = bool(np.all(diffs <= tol))
    t_inf = float(t[-1])
    resid = t[:-1] - t_inf
    # ratios of successive residuals, where the previous residual is non-trivial
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (t[1:] - t_inf) / resid
    mask = np.isfinite(ratios) & (np.abs(resid) > 1e-12) & (ratios > 0)
    rate = float(np.exp(np.mean(np.log(ratios[mask])))) if np.any(mask) else float("nan")
    return ConvergenceReport(
        monotone=monotone,
        final=t_inf,
        total_decrease=float(t[0] - t[-1]),
        n_steps=int(t.size - 1),
        rate=rate,
        max_increase=max_increase,
    )


@dataclass(frozen=True)
class OracleAgreement:
    """Agreement between an estimate and a known oracle value."""

    abs_error: float
    passed: bool
    estimate: float
    oracle: float
    tol: float


def oracle_agreement(estimate: float, oracle: float, *, tol: float = 1e-2) -> OracleAgreement:
    """Compare an estimate against an oracle (e.g. PC fixed point vs grid posterior).

    Returns the absolute error and whether it is within ``tol``. This is the
    anti-theater check used across Chapters 4–5: a variational / predictive-coding
    estimate is only "correct" when an independent oracle confirms it.
    """
    err = abs(float(estimate) - float(oracle))
    return OracleAgreement(
        abs_error=err,
        passed=bool(err <= tol),
        estimate=float(estimate),
        oracle=float(oracle),
        tol=float(tol),
    )
