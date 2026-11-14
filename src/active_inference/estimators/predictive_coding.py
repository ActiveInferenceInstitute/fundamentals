r"""Predictive-coding inference algorithms — Chapter 5 §5.2, §5.3, §5.4.

Three algorithms, all gradient descent on the MAP/Laplace free energy
(:mod:`active_inference.core.predictive_coding`), organized around precision-
weighted prediction errors:

* :func:`predictive_coding_inference` — **Algorithm 5.2.1**, univariate perception.
  Iterates ``μ_x ← μ_x − κ ∂F/∂μ_x`` (Eq. 15/16) until both the free-energy
  change and gradient are small. Its fixed point is the MAP estimate; for a
  linear generating function that equals Chapter 4's exact grid posterior mean.
* :func:`multivariate_predictive_coding` — **§5.3**, vector states/observations with
  precision *matrices* and a Jacobian matrix.  Reduces exactly to the univariate
  routine on a 1-D problem.
* :func:`hierarchical_predictive_coding` — **§5.4**, a stack of ``L`` layers
  (Eq. 30–34).  Each layer predicts the layer below through its generating function
  and is corrected by the bottom-up prediction error: "top-down predictions,
  bottom-up prediction errors."  The summed free energy is non-increasing.

Every gradient used here is *derived* (chain rule) and the routines record a full
trace so the tests can assert monotonicity and oracle agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.predictive_coding import (
    GenerativeFunction,
    PCFreeEnergy,
    PredictiveCodingModel,
    pc_free_energy_grad,
    predictive_coding_free_energy,
)


# ===========================================================================
# §5.2 — Univariate predictive coding (Algorithm 5.2.1)
# ===========================================================================


@dataclass
class PredictiveCodingResult:
    """Trace of :func:`predictive_coding_inference`.

    Attributes
    ----------
    mus : ndarray
        Belief mean ``μ_x`` per iteration (index 0 = initialization).
    free_energies : ndarray
        MAP free energy ``F`` per iteration.
    eps_x, eps_y : ndarray
        State and sensory prediction errors per iteration.
    mu_y : ndarray
        Expected observation ``g(μ_x)`` per iteration (the prediction).
    mu_star : float
        Final belief mean (the MAP estimate).
    converged : bool
        Whether both ``|ΔF| < tol`` and the final gradient norm are below ``tol``.
    n_iter_run : int
        Number of update steps taken.
    """

    mus: np.ndarray
    free_energies: np.ndarray
    eps_x: np.ndarray
    eps_y: np.ndarray
    mu_y: np.ndarray
    mu_star: float
    converged: bool
    n_iter_run: int

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])

    def summary(self, ndigits: int = 4) -> str:
        """Return key diagnostic quantities as a compact dictionary."""
        return (
            f"PredictiveCodingResult(mu*={round(self.mu_star, ndigits)}, "
            f"F={round(self.final_free_energy, ndigits)}, "
            f"converged={self.converged}, n_iter={self.n_iter_run})"
        )


def predictive_coding_inference(
    model: PredictiveCodingModel,
    y: float,
    *,
    mu0: Optional[float] = None,
    kappa: float = 0.1,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> PredictiveCodingResult:
    r"""Univariate predictive coding for perception (Algorithm 5.2.1).

    Gradient descent on the MAP free energy via precision-weighted prediction
    errors::

        μ_x ← μ_x − κ ∂F/∂μ_x,   ∂F/∂μ_x = λ_x ε_x − λ_y ε_y g'(μ_x)   (Eq. 15/16)

    Initialised at the prior mean ``mu0 = m_x`` (book line 1). ``kappa`` is the
    learning rate :math:`\kappa`; the book notes convergence is sensitive to it.
    Stops when both ``|ΔF| < tol`` and ``|∂F/∂μ_x| < tol``, or after ``n_iter`` steps.
    """
    mu = float(model.m_x if mu0 is None else mu0)

    def record(mu_val: float) -> PCFreeEnergy:
        """Append one step of inference diagnostics to the trace."""
        return predictive_coding_free_energy(model, y, mu_val)

    c = record(mu)
    mus = [mu]
    fes = [c.free_energy]
    eps_x = [c.eps_x]
    eps_y = [c.eps_y]
    mu_y = [c.mu_y]
    converged = False
    n_run = 0

    for _ in range(int(n_iter)):
        grad = pc_free_energy_grad(model, y, mu)
        mu = mu - kappa * grad
        c = record(mu)
        mus.append(mu)
        fes.append(c.free_energy)
        eps_x.append(c.eps_x)
        eps_y.append(c.eps_y)
        mu_y.append(c.mu_y)
        n_run += 1
        if abs(fes[-2] - fes[-1]) < tol and abs(pc_free_energy_grad(model, y, mu)) < tol:
            converged = True
            break

    return PredictiveCodingResult(
        mus=np.asarray(mus),
        free_energies=np.asarray(fes),
        eps_x=np.asarray(eps_x),
        eps_y=np.asarray(eps_y),
        mu_y=np.asarray(mu_y),
        mu_star=float(mu),
        converged=converged,
        n_iter_run=n_run,
    )


# ===========================================================================
# §5.3 — Multivariate predictive coding (precision matrices + Jacobian)
# ===========================================================================


@dataclass
class MultivariatePCResult:
    """Store vector predictive-coding trajectory, VFE trace, fixed point, and convergence.

    Mirrors the univariate :class:`PredictiveCodingResult`: alongside the belief and
    free-energy traces it records the **raw prediction errors** per iteration
    (``eps_y = y − g(μ)`` sensory, ``eps_x = μ − m_x`` state — unweighted; the
    precision weighting lives in the free energy), so the multivariate figure can
    show how the prediction errors evolve, like the scalar case vectorized.
    """

    mus: np.ndarray            # (n_iter+1, C)
    free_energies: np.ndarray  # (n_iter+1,)
    mu_star: np.ndarray        # (C,)
    converged: bool
    n_iter_run: int
    eps_y: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (n_iter+1, D)
    eps_x: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (n_iter+1, C)

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])

    @property
    def eps_y_norm(self) -> np.ndarray:
        """Per-iteration sensory prediction-error magnitude ``‖ε_y‖₂``."""
        return np.linalg.norm(self.eps_y, axis=1)

    @property
    def eps_x_norm(self) -> np.ndarray:
        """Per-iteration state prediction-error magnitude ``‖ε_x‖₂``."""
        return np.linalg.norm(self.eps_x, axis=1)


def _as_precision(prec_or_var: np.ndarray, dim: int) -> np.ndarray:
    """Coerce a scalar / vector / matrix into a (dim, dim) precision matrix."""
    arr = np.asarray(prec_or_var, dtype=float)
    if arr.ndim == 0:
        return np.eye(dim) * float(arr)
    if arr.ndim == 1:
        return np.diag(arr)
    return arr


def multivariate_predictive_coding(
    g: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    m_x: np.ndarray,
    *,
    precision_y: np.ndarray,
    precision_x: np.ndarray,
    mu0: Optional[np.ndarray] = None,
    kappa: float = 0.1,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> MultivariatePCResult:
    r"""Multivariate predictive coding (book §5.3).

    Vector hidden state ``x ∈ R^C`` and observation ``y ∈ R^D`` with a (possibly
    nonlinear) generating function ``g: R^C → R^D`` and its Jacobian
    ``J(μ) = ∂g/∂μ ∈ R^{D×C}``.  The MAP free energy is

    .. math::
        \mathcal F = \tfrac12\,\varepsilon_y^\top \Pi_y \varepsilon_y
                   + \tfrac12\,\varepsilon_x^\top \Pi_x \varepsilon_x + \text{const}

    with ``ε_y = y − g(μ)`` and ``ε_x = μ − m_x``, precision matrices ``Π_y, Π_x``.
    Recognition dynamics (gradient descent)::

        μ ← μ − κ (Π_x ε_x − J(μ)ᵀ Π_y ε_y)

    ``precision_y`` / ``precision_x`` accept a scalar, a **precision** vector (used
    as the diagonal of the precision matrix — *not* inverted), or a full precision
    matrix. Reduces exactly to :func:`predictive_coding_inference` on a 1-D problem.
    Convergence requires both the free-energy change and the final gradient's
    infinity norm to be below ``tol``; an equal-loss cycle is not stationary.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    m_x = np.atleast_1d(np.asarray(m_x, dtype=float))
    C = m_x.shape[0]
    D = y.shape[0]
    Pi_y = _as_precision(precision_y, D)
    Pi_x = _as_precision(precision_x, C)
    mu = m_x.copy() if mu0 is None else np.atleast_1d(np.asarray(mu0, dtype=float)).copy()

    def free_energy(m: np.ndarray) -> float:
        """Return the variational free-energy objective for current beliefs."""
        eps_y = y - np.atleast_1d(np.asarray(g(m), dtype=float))
        eps_x = m - m_x
        return float(0.5 * eps_y @ Pi_y @ eps_y + 0.5 * eps_x @ Pi_x @ eps_x)

    def errors(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ε_y, ε_x)`` at belief ``m`` (sensory then state)."""
        eps_y = y - np.atleast_1d(np.asarray(g(m), dtype=float))
        eps_x = m - m_x
        return eps_y, eps_x

    eps_y0, eps_x0 = errors(mu)
    mus = [mu.copy()]
    fes = [free_energy(mu)]
    eps_ys = [eps_y0]
    eps_xs = [eps_x0]
    converged = False
    n_run = 0

    for _ in range(int(n_iter)):
        eps_y, eps_x = errors(mu)
        J = np.atleast_2d(np.asarray(jacobian(mu), dtype=float)).reshape(D, C)
        grad = Pi_x @ eps_x - J.T @ Pi_y @ eps_y
        mu = mu - kappa * grad
        eps_y_next, eps_x_next = errors(mu)
        mus.append(mu.copy())
        fes.append(free_energy(mu))
        eps_ys.append(eps_y_next)
        eps_xs.append(eps_x_next)
        n_run += 1
        if abs(fes[-2] - fes[-1]) < tol:
            final_jacobian = np.asarray(jacobian(mu), dtype=float).reshape(D, C)
            final_grad = Pi_x @ eps_x_next - final_jacobian.T @ Pi_y @ eps_y_next
            if np.linalg.norm(final_grad, ord=np.inf) < tol:
                converged = True
                break

    return MultivariatePCResult(
        mus=np.asarray(mus),
        free_energies=np.asarray(fes),
        mu_star=mu.copy(),
        converged=converged,
        n_iter_run=n_run,
        eps_y=np.asarray(eps_ys),
        eps_x=np.asarray(eps_xs),
    )


def pc_multivariate_linear_fixed_point(
    A: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    m_x: np.ndarray,
    *,
    precision_y: np.ndarray,
    precision_x: np.ndarray,
) -> np.ndarray:
    r"""Closed-form recognition fixed point ``μ*`` for a **linear** ``g(x)=Ax+b``.

    The matrix analogue of :func:`~active_inference.core.predictive_coding.pc_linear_fixed_point`.
    Setting the gradient of the multivariate MAP free energy to zero,

    .. math::
        \frac{\partial\mathcal F}{\partial\mu}
            = \Pi_x(\mu-m_x) - A^\top\Pi_y\,(y - A\mu - b) = 0,

    gives the (precision-weighted) generalized least-squares / posterior mean

    .. math::
        \mu^* = (\Pi_x + A^\top\Pi_y A)^{-1}
                \big(\Pi_x m_x + A^\top\Pi_y (y-b)\big).

    For a near-flat prior (``Π_x → 0``) this reduces to the ordinary least-squares
    inverse ``A⁻¹(y−b)`` (square invertible ``A``). It is the independent oracle for
    :func:`multivariate_predictive_coding` on a linear model — the vector counterpart
    of the cross-chapter scalar oracle.

    ``precision_y`` / ``precision_x`` accept a scalar, a **precision** vector (used
    as the diagonal of the precision matrix — *not* inverted), or a full precision
    matrix, exactly like :func:`multivariate_predictive_coding` (both route through
    the same ``_as_precision`` coercion, so the oracle cannot silently disagree with
    the iterate).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    m_x = np.atleast_1d(np.asarray(m_x, dtype=float))
    D, C = A.shape
    Pi_y = _as_precision(precision_y, D)
    Pi_x = _as_precision(precision_x, C)
    lhs = Pi_x + A.T @ Pi_y @ A
    rhs = Pi_x @ m_x + A.T @ Pi_y @ (y - b)
    return np.linalg.solve(lhs, rhs)


def pc_parameterized_lstsq_oracle(
    Theta: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    *,
    sign: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Independent recovery oracle for the parameterized model ``g(x)=Θ(x⊙x)+b`` (§5.6).

    The book's parameterized multivariate generating function maps a hidden state
    ``x ∈ R^C`` to an observation ``y ∈ R^D`` through a **rectangular** mixing matrix
    ``Θ ∈ R^{D×C}`` acting on the *element-wise square* ``x⊙x``:

    .. math::
        g(x) = \Theta\,(x \odot x) + b .

    Because ``g`` depends on ``x`` only through ``u = x \odot x``, a noiseless,
    self-consistent observation ``y = g(x^*)`` satisfies the linear system
    ``\Theta u^* = y - b`` exactly.  The (over-determined, ``D>C``) squared state is
    therefore recovered by the least-squares solve ``u^* = \Theta^{+}(y-b)`` and the
    state itself by ``x^* = \operatorname{sign} \odot \sqrt{u^*}`` — the sign of each
    component is unidentifiable from ``g`` alone (``g`` is even in every ``x_c``), so it
    is supplied by the caller (defaulting to ``+1``, i.e. the positive orthant that the
    recognition dynamics settle into when initialised there).

    For a noiseless, self-consistent ``y`` this coincides exactly with the flat-prior
    fixed point of :func:`multivariate_predictive_coding` on this model (the residual
    is zero, so no precision weighting can change the solution) — its independent
    cross-check, the nonlinear, over-determined counterpart of
    :func:`pc_multivariate_linear_fixed_point`.  For a *noisy* (inconsistent) ``y`` the
    two coincide only when ``precision_y`` is **isotropic** (a scalar or scalar
    multiple of the identity); a general diagonal/full ``precision_y`` makes the
    recognition-dynamics fixed point solve the weighted normal equations
    ``Θᵀ Π_y Θ u* = Θᵀ Π_y (y−b)``, which differs from this function's unweighted
    least-squares solve. ``u^*`` is clipped at zero before the square root so a mildly
    inconsistent (noisy) ``y`` still returns a real state.

    Parameters
    ----------
    Theta : array (D, C)
        Rectangular mixing matrix.
    b : array (D,)
        Additive observation offset.
    y : array (D,)
        Observation to invert.
    sign : array (C,), optional
        Per-component sign of the recovered state (``±1``); defaults to all ``+1``.

    Returns
    -------
    array (C,)
        The recovered hidden state ``x^*``.
    """
    Theta = np.atleast_2d(np.asarray(Theta, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    D, C = Theta.shape
    if b.shape[0] != D or y.shape[0] != D:
        raise ValueError(
            f"b and y must have length D={D} to match Theta; got {b.shape[0]} and {y.shape[0]}"
        )
    u_star, *_ = np.linalg.lstsq(Theta, y - b, rcond=None)
    u_star = np.clip(u_star, 0.0, np.inf)
    if sign is None:
        sign_vec = np.ones(C)
    else:
        sign_vec = np.sign(np.atleast_1d(np.asarray(sign, dtype=float)))
        sign_vec[sign_vec == 0.0] = 1.0
        if sign_vec.shape[0] != C:
            raise ValueError(f"sign must have length C={C}; got {sign_vec.shape[0]}")
    return sign_vec * np.sqrt(u_star)


# ===========================================================================
# §5.4 — Hierarchical predictive coding (Eq. 30–34)
# ===========================================================================


@dataclass
class HierarchicalPCModel:
    r"""A hierarchical predictive-coding generative model (book §5.4, Eq. 30).

    Nodes ``μ^[0], …, μ^[L]`` where ``μ^[0] = y`` is the (fixed) sensory input and
    ``μ^[1..L]`` are hidden states.  ``generators[k]`` is the generating function
    ``g^[k+1]`` that predicts node ``k`` from node ``k+1`` (``k = 0 … L−1``); the
    top node is pulled toward the constant ``m_x``.  ``variances[l]`` is ``σ²[l]``
    for node ``l = 0 … L`` (length ``L+1``).

    Layer-wise prediction errors (Eq. 32, node convention of Fig. 5.4.2):

    .. math::
        \varepsilon^{[l]} = \mu^{[l]} - g^{[l+1]}(\mu^{[l+1]}) \;\; (l<L), \qquad
        \varepsilon^{[L]} = \mu^{[L]} - m_x .

    **Top-node semantics.** ``m_x`` is the top-level prediction. Two cases the book
    uses: a fixed top prior of mean ``m_x`` (Eq. 30), or an **unconstrained** top
    layer (``g^{[L+1]}=0`` ⇒ ``ε^{[L]}=μ^{[L]}``, p.306 and Example 5.7) — obtained
    by setting ``m_x=0``.  Example 5.7 uses the unconstrained top with the hidden
    states *initialised* at 3 (``mu0=[3,…]``), converging to ``μ^{[1]}=1, μ^{[2]}=0``.
    """

    generators: Sequence[GenerativeFunction]
    variances: Sequence[float]
    m_x: float

    def __post_init__(self) -> None:
        self.L = len(self.generators)
        if len(self.variances) != self.L + 1:
            raise ValueError(
                f"variances must have length L+1={self.L + 1}, got {len(self.variances)}"
            )
        if any(v <= 0 for v in self.variances):
            raise ValueError("all variances must be strictly positive")

    @property
    def precisions(self) -> np.ndarray:
        """Return the precision parameter implied by the validated variance."""
        return 1.0 / np.asarray(self.variances, dtype=float)

    def prediction(self, lvl: int, mus: np.ndarray) -> float:
        """Top-down prediction for node ``lvl`` from the layer above."""
        if lvl == self.L:
            return self.m_x
        return float(np.asarray(self.generators[lvl](mus[lvl + 1])))

    def errors(self, mus: np.ndarray) -> np.ndarray:
        """Layer-wise prediction errors ``ε^[l]`` for ``l = 0 … L`` (Eq. 32)."""
        return np.array([mus[lvl] - self.prediction(lvl, mus) for lvl in range(self.L + 1)])

    def free_energy(self, mus: np.ndarray) -> float:
        r"""Summed VFE ``Σ_l ½(λ^[l] ε^[l]² + log σ²[l])`` (Eq. 34)."""
        eps = self.errors(mus)
        lam = self.precisions
        return float(np.sum(0.5 * (lam * eps**2 + np.log(self.variances))))

    def layer_free_energies(self, mus: np.ndarray) -> np.ndarray:
        """Per-layer VFE contributions (the summands of Eq. 34)."""
        eps = self.errors(mus)
        lam = self.precisions
        return 0.5 * (lam * eps**2 + np.log(self.variances))


@dataclass
class HierarchicalPCResult:
    """Store hierarchical predictive-coding beliefs, errors, layer energies, and convergence."""

    mus: np.ndarray            # (n_iter+1, L+1) — column 0 is the fixed observation
    errors: np.ndarray         # (n_iter+1, L+1)
    free_energies: np.ndarray  # (n_iter+1,) summed VFE
    layer_free_energies: np.ndarray  # (n_iter+1, L+1)
    mu_star: np.ndarray        # (L+1,)
    converged: bool
    n_iter_run: int

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])


def hierarchical_predictive_coding(
    model: HierarchicalPCModel,
    y: float,
    *,
    mu0: Optional[Sequence[float]] = None,
    kappa: float = 0.05,
    n_iter: int = 500,
    tol: float = 1e-12,
) -> HierarchicalPCResult:
    r"""Hierarchical predictive coding (book §5.4, Algorithm via Eq. 33).

    Updates every hidden node ``μ^[l]`` (``l = 1 … L``; ``μ^[0]=y`` is clamped) by
    gradient descent on the summed free energy (Eq. 34).  The analytic gradient for
    node ``l`` (derived; FD-checked in tests) is

    .. math::
        \frac{\partial\mathcal F}{\partial\mu^{[l]}}
            = \lambda^{[l]}\varepsilon^{[l]}
            - \lambda^{[l-1]}\varepsilon^{[l-1]}\,g^{[l]\prime}(\mu^{[l]}),

    i.e. this layer's own (top-down) prediction error minus the Jacobian-weighted
    prediction error of the layer below — "bottom-up errors meet top-down
    predictions."  ``μ^[0]`` is held at the observation ``y``.
    """
    L = model.L
    if mu0 is None:
        # Initialise hidden nodes at the prior mean (book line 1, Eq. 45 analogue).
        mus = np.array([float(y)] + [float(model.m_x)] * L)
    else:
        mus = np.array([float(y)] + [float(v) for v in mu0])
        if mus.shape[0] != L + 1:
            raise ValueError(f"mu0 must have length L={L}")

    lam = model.precisions

    def grad_layer(lvl: int, m: np.ndarray) -> float:
        """Support iterative estimation for the corresponding book algorithm."""
        eps = model.errors(m)
        # own top-down error term
        g_own = lam[lvl] * eps[lvl]
        # bottom-up error from the layer below (node lvl predicts node lvl-1 via g^[lvl])
        gprime = float(np.asarray(model.generators[lvl - 1].derivative(m[lvl])))
        g_below = lam[lvl - 1] * eps[lvl - 1] * gprime
        return g_own - g_below

    fes = [model.free_energy(mus)]
    errs = [model.errors(mus)]
    lfes = [model.layer_free_energies(mus)]
    hist = [mus.copy()]
    converged = False
    n_run = 0

    for _ in range(int(n_iter)):
        # Simultaneous update of all hidden layers (Jacobi-style sweep).
        grads = np.array([grad_layer(lvl, mus) for lvl in range(1, L + 1)])
        mus = mus.copy()
        mus[1:] = mus[1:] - kappa * grads
        fes.append(model.free_energy(mus))
        errs.append(model.errors(mus))
        lfes.append(model.layer_free_energies(mus))
        hist.append(mus.copy())
        n_run += 1
        if abs(fes[-2] - fes[-1]) < tol:
            converged = True
            break

    return HierarchicalPCResult(
        mus=np.asarray(hist),
        errors=np.asarray(errs),
        free_energies=np.asarray(fes),
        layer_free_energies=np.asarray(lfes),
        mu_star=mus.copy(),
        converged=converged,
        n_iter_run=n_run,
    )
