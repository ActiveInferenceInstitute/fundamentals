"""Variational free energy — the core method of Chapter 4.

Variational Bayesian inference (VBI) turns posterior estimation into an
optimization problem: pick a *variational density* ``q(x)`` and adjust it until
it matches the true posterior ``p(x | y)``.  The objective is **variational free
energy** (VFE), denoted :math:`\\mathcal{F}`.  The book introduces VFE as

.. math::

    \\mathcal{F} = \\int q(x)\\,\\log\\frac{q(x)}{p(x, y)}\\,dx
                 = D_{KL}\\big(q(x) \\,\\|\\, p(x, y)\\big)

(the *G-form*, Eq. 10) and shows it can be algebraically rearranged into several
equivalent forms, each giving a different intuition.  Every form below evaluates
to the *same number* on the same ``(q, model, y)``; :func:`variational_free_energy`
computes them all at once and :meth:`VFEComponents.check` asserts their agreement.

All five forms (book §4.2–4.4):

================  ==================================================  ============
Form              Decomposition                                       Equation
================  ==================================================  ============
``G`` (generative) :math:`\\mathcal F = E_q[\\log q] - E_q[\\log p(x,y)]`   Eq. 10
``D`` (divergence) :math:`\\mathcal F = D_{KL}(q\\|p(x|y)) - \\log p(y)`     Eq. 18/27
``C`` (complexity) :math:`\\mathcal F = D_{KL}(q\\|p(x)) - E_q[\\log p(y|x)]` Eq. 28
``E`` (energy)     :math:`\\mathcal F = -H[q] - E_q[\\log p(x,y)]`           Eq. 29
``MAP``/``MLE``    point-mass special cases (Eq. 30/31)
================  ==================================================  ============

Two facts proved in §4.3 and used everywhere as verification oracles:

* **Upper bound on surprisal** (Eq. 26): :math:`\\mathcal F \\ge -\\log p(y)` for
  *any* ``q`` (Gibbs/Jensen).  See :func:`free_energy_bound_gap`.
* **Tight at the posterior**: :math:`\\mathcal F = -\\log p(y)` iff
  ``q(x) = p(x | y)``, i.e. when the D-form divergence is zero.

Everything is computed on a shared 1-D ``x_grid`` by trapezoid integration, the
same scheme :class:`~active_inference.core.inference.GridBayesianInference` uses,
so the variational results are directly comparable to the exact grid posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .distributions import gaussian_log_pdf, gaussian_pdf, normalize_density
from .generative_model import GenerativeModel
from .inference import GridBayesianInference

ArrayLike = np.ndarray

# A variational density may be supplied as a Gaussian belief, as a vector of
# density values already evaluated on the grid, or as a callable ``q(x)``.
QDensity = Union["GaussianBelief", np.ndarray, Callable[[np.ndarray], np.ndarray]]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Variational density
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianBelief:
    """A Gaussian variational density ``q(x) = N(x ; mu, var)``.

    This is the parametric ``q`` used throughout Chapter 4 (Eq. 12, 39): the
    agent's belief about a hidden state, parameterized by a mean and variance
    that variational inference adjusts.

    Parameters
    ----------
    mu : float
        Mean :math:`\\mu` of the belief.
    var : float
        Variance :math:`\\sigma^2` of the belief. Must be strictly positive.
    """

    mu: float
    var: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not (self.var > 0) or not np.isfinite(self.var):
            raise ValueError(f"var must be a finite positive number, got {self.var!r}")

    @property
    def std(self) -> float:
        """Standard deviation :math:`\\sigma = \\sqrt{\\mathrm{var}}`."""
        return float(np.sqrt(self.var))

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Density ``q(x) = N(x ; mu, var)`` evaluated at ``x``."""
        return gaussian_pdf(np.asarray(x, dtype=float), self.mu, self.var)

    def logpdf(self, x: ArrayLike) -> np.ndarray:
        """Log density ``log q(x)`` evaluated at ``x``."""
        return gaussian_log_pdf(np.asarray(x, dtype=float), self.mu, self.var)

    def entropy(self) -> float:
        """Differential entropy ``H[q] = 0.5 log(2 pi e var)`` (nats)."""
        return 0.5 * (_LOG_2PI + 1.0 + float(np.log(self.var)))

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n`` samples from the belief."""
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(self.mu, self.std, size=n)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GaussianBelief(mu={self.mu:g}, var={self.var:g})"


def evaluate_q(q: QDensity, x_grid: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Evaluate a variational density on ``x_grid`` as a (normalized) density vector.

    Accepts a :class:`GaussianBelief`, a pre-computed density array, or a callable
    ``q(x)``. When ``normalize`` is true the result is renormalized to integrate
    to one over the grid (trapezoid rule) so that free-energy forms agree exactly
    and the surprisal bound holds; this is standard practice for grid methods.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if isinstance(q, GaussianBelief):
        vals = q.pdf(x_grid)
    elif callable(q):
        vals = np.asarray(q(x_grid), dtype=float)
    else:
        vals = np.asarray(q, dtype=float)
        if vals.shape != x_grid.shape:
            raise ValueError(
                f"q density array shape {vals.shape} does not match x_grid {x_grid.shape}"
            )
    if np.any(vals < 0):
        raise ValueError("variational density must be non-negative")
    if not np.all(np.isfinite(vals)):
        raise ValueError("variational density must be finite on the whole grid")
    if normalize:
        vals = normalize_density(vals, x_grid)
    return vals


# ---------------------------------------------------------------------------
# Free-energy components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VFEComponents:
    """Every decomposition of variational free energy for one ``(q, model, y)``.

    All four full forms (G, D, C, E) reconstruct the same ``free_energy``; the
    individual terms expose the intuition behind each (book §4.4, Fig. 4.4.1).

    Attributes
    ----------
    free_energy : float
        Variational free energy :math:`\\mathcal F` (nats).
    divergence : float
        D-form: :math:`D_{KL}(q \\| p(x|y))` — distance to the *true posterior*.
        Zero exactly when ``q`` equals the posterior.
    surprisal : float
        D-form: :math:`-\\log p(y)`, the negative log model-evidence. ``free_energy``
        is an upper bound on this (Eq. 26).
    complexity : float
        C-form: :math:`D_{KL}(q \\| p(x))` — divergence of the belief from the prior
        (Bayesian surprise / the Occam term).
    accuracy : float
        C-form: :math:`E_q[\\log p(y|x)]` — expected log-likelihood (negative
        prediction error). ``free_energy = complexity - accuracy``.
    neg_entropy : float
        E-form: :math:`E_q[\\log q] = -H[q]`.
    energy : float
        E-form: :math:`E_q[\\log p(x, y)]`, the "average energy". ``free_energy =
        neg_entropy - energy``.
    log_evidence : float
        :math:`\\log p(y)` computed exactly on the grid (the oracle).
    """

    free_energy: float
    divergence: float
    surprisal: float
    complexity: float
    accuracy: float
    neg_entropy: float
    energy: float
    log_evidence: float

    # -- reconstructions, one per form (all == free_energy to grid precision) --

    @property
    def g_form(self) -> float:
        """G-form reconstruction: ``E_q[log q] - E_q[log p(x,y)]``."""
        return self.neg_entropy - self.energy

    @property
    def d_form(self) -> float:
        """D-form reconstruction: ``divergence - log p(y)`` (Eq. 18).

        Note: this is a *definitional* identity, not an independent check. Because
        ``divergence`` is computed as ``∫q(log q - (log p(x,y) - log p(y)))`` the
        ``log p(y)`` cancels against ``surprisal``, so ``d_form`` equals the
        E/G-form for any ``log_evidence``. The genuinely independent oracle check
        is :attr:`bound_gap` (``≥ 0``, tight at the posterior) plus the non-trivial
        agreement of the C-form (which regroups the prior/likelihood integrals).
        """
        return self.divergence + self.surprisal

    @property
    def c_form(self) -> float:
        """C-form reconstruction: ``complexity - accuracy`` (Eq. 28)."""
        return self.complexity - self.accuracy

    @property
    def e_form(self) -> float:
        """E-form reconstruction: ``neg_entropy - energy`` (Eq. 29)."""
        return self.neg_entropy - self.energy

    @property
    def bound_gap(self) -> float:
        """``free_energy - (-log p(y)) = divergence`` — the slack in Eq. 26 (≥ 0)."""
        return self.free_energy - (-self.log_evidence)

    def check(self, atol: float = 1e-6) -> None:
        """Assert all four forms agree with ``free_energy`` to ``atol``.

        The C-form regroups the prior/likelihood integrals independently of the
        G/E forms, so its agreement catches term-misassignment bugs. The D-form is
        a definitional reconstruction (see :attr:`d_form`); it is checked for
        completeness but cannot, alone, detect an error. As an additional genuinely
        independent guard this also asserts the KL non-negativity ``divergence >= 0``
        (Gibbs inequality) — a sign error in the divergence integral would trip it.

        Raises
        ------
        AssertionError
            If any form disagrees beyond ``atol``, or the divergence is negative
            beyond ``atol`` — a sign of a numerical bug.
        """
        for name, val in (
            ("G", self.g_form),
            ("D", self.d_form),
            ("C", self.c_form),
            ("E", self.e_form),
        ):
            if not np.isclose(val, self.free_energy, atol=atol, rtol=0.0):
                raise AssertionError(
                    f"{name}-form {val:.10g} disagrees with F {self.free_energy:.10g} "
                    f"(atol={atol})"
                )
        if self.divergence < -atol:
            raise AssertionError(
                f"divergence {self.divergence:.10g} is negative (violates Gibbs "
                f"inequality KL(q‖p(x|y)) >= 0; atol={atol})"
            )

    def summary(self, ndigits: int = 4) -> str:
        """One-line human-readable summary across forms."""
        return (
            f"VFE(F={round(self.free_energy, ndigits)}, "
            f"div={round(self.divergence, ndigits)}, "
            f"surprisal={round(self.surprisal, ndigits)}, "
            f"complexity={round(self.complexity, ndigits)}, "
            f"accuracy={round(self.accuracy, ndigits)}, "
            f"H[q]={round(-self.neg_entropy, ndigits)}, "
            f"energy={round(self.energy, ndigits)})"
        )


def _trapz(values: np.ndarray, x_grid: np.ndarray) -> float:
    """Integrate a one-dimensional grid function using the trapezoid rule."""
    return float(np.trapezoid(values, x_grid))


def _weighted_log(q: np.ndarray, log_values: np.ndarray) -> np.ndarray:
    """Multiply only on q's support, retaining infinities at positive density."""
    out = np.zeros_like(q)
    np.multiply(q, log_values, out=out, where=q > 0)
    return out


def variational_free_energy(
    q: QDensity,
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    log_evidence: Optional[float] = None,
    posterior: Optional[np.ndarray] = None,
    normalize_q: bool = True,
) -> VFEComponents:
    """Compute variational free energy and *every* decomposition (book §4.2–4.4).

    Parameters
    ----------
    q : GaussianBelief | ndarray | callable
        The variational density. A :class:`GaussianBelief`, a density array on
        ``x_grid``, or a callable ``q(x)``.
    model : GenerativeModel
        Provides ``log_likelihood(y, x_grid)`` and ``log_prior(x_grid)`` so the
        log joint ``log p(x, y) = log p(y|x) + log p(x)`` can be formed.
    y : float
        The (clamped) observation :math:`\\hat y`.
    x_grid : ndarray
        Discretized hidden-state domain shared with the exact grid inference.
    log_evidence, posterior : optional
        Pre-computed ``log p(y)`` and exact posterior density on ``x_grid``. When
        omitted they are computed once via :class:`GridBayesianInference` — supply
        them in inner loops to avoid recomputation.
    normalize_q : bool
        Renormalize ``q`` on the grid before integrating (default ``True``).

    Returns
    -------
    VFEComponents
        All forms and decomposition terms; ``.free_energy`` is the VFE.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    q_vals = evaluate_q(q, x_grid, normalize=normalize_q)
    log_q = np.log(np.where(q_vals > 0, q_vals, 1.0))  # log used only where q>0

    log_lik = np.asarray(model.log_likelihood(y, x_grid), dtype=float)   # log p(y|x)
    log_prior = np.asarray(model.log_prior(x_grid), dtype=float)         # log p(x)
    log_joint = log_lik + log_prior                                      # log p(x,y)

    # Log evidence (the oracle) — compute once if not supplied. The D-form builds
    # ``log p(x|y) = log_joint − log_evidence`` directly, so only the scalar
    # ``log_evidence`` is needed here; ``posterior`` stays in the signature for
    # backward compatibility and callers that pass it in inner loops.
    del posterior  # not needed: log p(x|y) is reconstructed from log_joint below
    if log_evidence is None:
        result = GridBayesianInference(model=model, x_grid=x_grid).infer(y)
        log_evidence = float(result.log_evidence)

    # E / G form pieces.
    q_log_q = _weighted_log(q_vals, log_q)
    neg_entropy = _trapz(q_log_q, x_grid)                     # E_q[log q] = -H[q]
    energy = _trapz(_weighted_log(q_vals, log_joint), x_grid)  # E_q[log p(x,y)]
    free_energy = neg_entropy - energy

    # C form pieces.
    complexity = _trapz(q_log_q - _weighted_log(q_vals, log_prior), x_grid)
    accuracy = _trapz(_weighted_log(q_vals, log_lik), x_grid)  # E_q[log p(y|x)]

    # D form pieces — divergence from the *true posterior*.
    #
    # KL(q ‖ p(x|y)) = ∫ q (log q − log p(x|y)) dx. We form log p(x|y) as
    # ``log p(x, y) − log p(y) = log_joint − log_evidence`` rather than taking the
    # log of the normalized posterior *array*: the array underflows to exactly 0
    # in the deep tail (where ``log 0 = −∞`` would be silently dropped), whereas
    # ``log_joint`` retains the model's true support. Zero q contributes zero;
    # positive q outside model support retains infinite divergence.
    # (``posterior`` is still accepted/returned as the grid oracle.)
    log_post = log_joint - float(log_evidence)
    integrand = q_log_q - _weighted_log(q_vals, log_post)
    divergence = _trapz(integrand, x_grid)
    surprisal = -float(log_evidence)

    return VFEComponents(
        free_energy=free_energy,
        divergence=divergence,
        surprisal=surprisal,
        complexity=complexity,
        accuracy=accuracy,
        neg_entropy=neg_entropy,
        energy=energy,
        log_evidence=float(log_evidence),
    )


# ---------------------------------------------------------------------------
# Thin form wrappers — each returns (F, *named decomposition terms)
# ---------------------------------------------------------------------------


def vfe_g_form(q: QDensity, model: GenerativeModel, y: float, x_grid: np.ndarray) -> float:
    """Return Eq. 10 G-form free energy by integrating q log(q/p(x,y))."""
    return variational_free_energy(q, model, y, x_grid).free_energy


def vfe_d_form(
    q: QDensity, model: GenerativeModel, y: float, x_grid: np.ndarray
) -> tuple[float, float, float]:
    """D-form (Eq. 18): returns ``(F, divergence=KL(q‖p(x|y)), surprisal=-log p(y))``."""
    c = variational_free_energy(q, model, y, x_grid)
    return c.free_energy, c.divergence, c.surprisal


def vfe_c_form(
    q: QDensity, model: GenerativeModel, y: float, x_grid: np.ndarray
) -> tuple[float, float, float]:
    """C-form (Eq. 28): returns ``(F, complexity=KL(q‖p(x)), accuracy=E_q[log p(y|x)])``."""
    c = variational_free_energy(q, model, y, x_grid)
    return c.free_energy, c.complexity, c.accuracy


def vfe_e_form(
    q: QDensity, model: GenerativeModel, y: float, x_grid: np.ndarray
) -> tuple[float, float, float]:
    """E-form (Eq. 29): returns ``(F, neg_entropy=-H[q], energy=E_q[log p(x,y)])``."""
    c = variational_free_energy(q, model, y, x_grid)
    return c.free_energy, c.neg_entropy, c.energy


def vfe_map_form(model: GenerativeModel, y: float, mu: float) -> float:
    """MAP-form point objective (Eq. 30): ``log p(y | x=mu) + log p(x=mu)``.

    This is the joint log-density at the belief mean. *Maximizing* it over ``mu``
    recovers the MAP estimate — the point-mass special case of VFE minimization
    when only the mean of ``q`` is of interest.
    """
    grid = np.array([float(mu)])
    return float(model.log_likelihood(y, grid)[0] + model.log_prior(grid)[0])


def vfe_mle_form(model: GenerativeModel, y: float, mu: float) -> float:
    """MLE-form point objective (Eq. 31): ``log p(y | x=mu)`` (uniform-prior case)."""
    grid = np.array([float(mu)])
    return float(model.log_likelihood(y, grid)[0])


# ---------------------------------------------------------------------------
# Surprisal / model evidence helpers (book §4.3, Examples 4.2–4.3)
# ---------------------------------------------------------------------------


def log_model_evidence(model: GenerativeModel, y: float, x_grid: np.ndarray) -> float:
    """``log p(y) = log ∫ p(x, y) dx`` on the grid (Eq. 19)."""
    return float(GridBayesianInference(model=model, x_grid=x_grid).infer(y).log_evidence)


def surprisal(model: GenerativeModel, y: float, x_grid: np.ndarray) -> float:
    """Return book Section 4.3 surprisal, the negative log model evidence."""
    return -log_model_evidence(model, y, x_grid)


def free_energy_bound_gap(
    q: QDensity, model: GenerativeModel, y: float, x_grid: np.ndarray
) -> float:
    """Slack in the bound :math:`\\mathcal F \\ge -\\log p(y)` (Eq. 26); always ``>= 0``.

    Equals the D-form divergence :math:`D_{KL}(q \\| p(x|y))`. Zero iff ``q`` is the
    true posterior.
    """
    return variational_free_energy(q, model, y, x_grid).bound_gap
