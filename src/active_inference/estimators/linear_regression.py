"""Linear regression estimators — analytic, gradient-descent, and Bayesian.

Three flavors live here:

* :func:`mle_linear_regression` — closed-form normal equation
  ``θ = (XᵀX)⁻¹ Xᵀ y`` (Moore–Penrose for stability).
* :func:`gd_linear_regression` — vectorized gradient descent on the squared
  loss with optional L2 regularization.
* :class:`BayesianLinearRegression` — conjugate Gaussian prior over ``θ``
  with closed-form posterior mean and covariance.

The data-matrix convention follows the book: ``X ∈ ℝ^{N × (C+1)}`` with a
column of ones prepended for the intercept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to a ``(N, C)`` design matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mle_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    intercept: bool = True,
) -> np.ndarray:
    """Closed-form maximum-likelihood ``θ`` for ``y = Xθ + ε``.

    Uses ``np.linalg.lstsq`` (Moore–Penrose) so it is stable when
    ``XᵀX`` is ill-conditioned.

    Parameters
    ----------
    X : np.ndarray, shape ``(N, C)`` or ``(N,)``
        Design matrix without an intercept column unless ``intercept=False``.
    y : np.ndarray, shape ``(N,)``
        Targets.
    intercept : bool
        If ``True`` (default) a column of ones is prepended to ``X``.

    Returns
    -------
    np.ndarray, shape ``(C+1,)`` or ``(C,)``
    """
    X = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def squared_loss(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    intercept: bool = True,
    l2: float = 0.0,
) -> float:
    """Penalized squared error ``½‖y − Xθ‖² + ½ λ ‖θ‖²`` evaluated at ``theta``."""
    X = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    residuals = y - X @ theta
    return float(0.5 * residuals @ residuals + 0.5 * l2 * theta @ theta)


def squared_loss_grad(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    intercept: bool = True,
    l2: float = 0.0,
) -> np.ndarray:
    """Analytic gradient of :func:`squared_loss` w.r.t. ``theta``."""
    X = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    residuals = X @ theta - y
    return X.T @ residuals + l2 * theta


@dataclass
class GDRegressionResult:
    """Result of :func:`gd_linear_regression` — final iterate plus full history.

    Attributes
    ----------
    theta : np.ndarray
        Final parameter vector.
    history : np.ndarray
        Iterate trajectory of shape ``(K + 1, P)``.
    losses : np.ndarray
        Loss value at each iterate, shape ``(K + 1,)``.
    n_iterations : int
        Completed iterations ``K``.
    converged : bool
        Whether the loop terminated by hitting the tolerance.
    """

    theta: np.ndarray
    history: np.ndarray
    losses: np.ndarray
    n_iterations: int
    converged: bool


def gd_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
    theta0: Optional[np.ndarray] = None,
    intercept: bool = True,
    l2: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    record_history: bool = True,
) -> GDRegressionResult:
    """Vectorized gradient descent for ``θ`` with optional L2 regularization.

    Parameters
    ----------
    learning_rate : float
        Step size. Must be smaller than ``2 / λ_max(XᵀX + l2 I)``
        for convergence; the helper does not clip for you.
    theta0 : np.ndarray or None
        Initial iterate. If ``None``, drawn from ``N(0, 1)`` using ``rng``.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    X_aug = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    p = X_aug.shape[1]

    if theta0 is None:
        if rng is None:
            rng = np.random.default_rng()
        theta = rng.normal(size=p)
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
        if theta.size != p:
            raise ValueError(
                f"theta0 must have length {p}, got {theta.size}"
            )

    history = [theta.copy()]
    losses = [squared_loss(theta, X, y, intercept=intercept, l2=l2)]
    converged = False

    for k in range(max_iter):
        grad = squared_loss_grad(theta, X, y, intercept=intercept, l2=l2)
        theta_next = theta - learning_rate * grad
        if record_history:
            history.append(theta_next.copy())
            losses.append(
                squared_loss(theta_next, X, y, intercept=intercept, l2=l2)
            )
        if np.linalg.norm(theta_next - theta) < tol:
            theta = theta_next
            converged = True
            break
        theta = theta_next

    return GDRegressionResult(
        theta=theta,
        history=np.asarray(history),
        losses=np.asarray(losses),
        n_iterations=k + 1,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Bayesian linear regression
# ---------------------------------------------------------------------------


@dataclass
class BLRPosterior:
    """Posterior over the parameter vector ``θ``."""

    mean: np.ndarray
    cov: np.ndarray

    def std(self) -> np.ndarray:
        """Per-parameter posterior standard deviation (sqrt of `cov` diagonal)."""
        return np.sqrt(np.diag(self.cov))

    def predict(self, X: np.ndarray, *, intercept: bool = True) -> np.ndarray:
        """Posterior-mean prediction ``X @ θ_mean`` (no uncertainty)."""
        X_aug = add_intercept(X) if intercept else np.asarray(X, dtype=float)
        return X_aug @ self.mean

    def predictive(self, X: np.ndarray, sigma2_y: float, *,
                   intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean / variance at new inputs ``X``.

        ``var_pred = σ²_y + xᵀ Σ_θ x``.
        """
        X_aug = add_intercept(X) if intercept else np.asarray(X, dtype=float)
        mean_pred = X_aug @ self.mean
        var_pred = sigma2_y + np.einsum(
            "nd,de,ne->n", X_aug, self.cov, X_aug
        )
        return mean_pred, var_pred

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw ``n`` parameter samples from this posterior.

        Returns shape ``(n, P)`` where ``P = mean.size``.
        """
        from ..core.distributions import mvn_sample
        return mvn_sample(self.mean, self.cov, n=n, rng=rng)

    def summary(self, ndigits: int = 4) -> str:
        """Human-readable one-line summary of the posterior over θ."""
        m = np.round(self.mean, ndigits).tolist()
        s = np.round(self.std(), ndigits).tolist()
        return f"BLRPosterior(mean={m}, std={s})"


@dataclass
class BayesianLinearRegression:
    """Conjugate Gaussian prior + Gaussian likelihood over ``θ``.

    Likelihood :  ``p(y | X, θ) = N(y ; Xθ, σ²_y I)``
    Prior      :  ``p(θ)        = N(θ ; mθ, Σθ)``

    Closed-form posterior::

        Σ_post = (Σθ⁻¹ + σ²_y⁻¹ Xᵀ X)⁻¹
        m_post = Σ_post (σ²_y⁻¹ Xᵀ y + Σθ⁻¹ mθ)
    """

    prior_mean: np.ndarray
    prior_cov: np.ndarray
    sigma2_y: float
    intercept: bool = True

    def __post_init__(self) -> None:
        self.prior_mean = np.asarray(self.prior_mean, dtype=float).reshape(-1)
        self.prior_cov = np.asarray(self.prior_cov, dtype=float)
        if self.prior_cov.shape != (self.prior_mean.size, self.prior_mean.size):
            raise ValueError("prior_cov shape must match prior_mean length")
        if self.sigma2_y <= 0:
            raise ValueError("sigma2_y must be positive")

    def fit(self, X: np.ndarray, y: np.ndarray) -> BLRPosterior:
        """Compute the closed-form Gaussian posterior over ``θ`` given ``(X, y)``."""
        X_aug = add_intercept(X) if self.intercept else np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X_aug.shape[1] != self.prior_mean.size:
            raise ValueError(
                f"design matrix has {X_aug.shape[1]} columns but prior has "
                f"{self.prior_mean.size}"
            )
        prec_prior = np.linalg.inv(self.prior_cov)
        prec_post = prec_prior + (X_aug.T @ X_aug) / self.sigma2_y
        cov_post = np.linalg.inv(prec_post)
        mean_post = cov_post @ (
            X_aug.T @ y / self.sigma2_y + prec_prior @ self.prior_mean
        )
        return BLRPosterior(mean=mean_post, cov=cov_post)

    def fit_sequential(self, X: np.ndarray, y: np.ndarray):
        """Yield ``(i, BLRPosterior)`` after assimilating each row of ``X``.

        Useful for animations of the posterior tightening as N grows.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[0]
        for i in range(1, n + 1):
            yield i, self.fit(X[:i], y[:i])
