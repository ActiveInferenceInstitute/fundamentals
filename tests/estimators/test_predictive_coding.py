"""Tests for ``estimators.predictive_coding`` — the Chapter 5 PC algorithms.

Oracle discipline (anti-theater): predictive coding's fixed point is verified
against an *independent* source of truth — Chapter 4's :class:`GridBayesianInference`
posterior (linear model) and an independent grid argmin of the MAP free energy
(nonlinear model). Monotonicity, hierarchical convergence, and the multivariate→
scalar reduction are all asserted, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.diagnostics import (
    convergence_report,
    gradient_check_vector,
    oracle_agreement,
)
from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.inference import GridBayesianInference
from active_inference.core.predictive_coding import (
    LinearFunction,
    PredictiveCodingModel,
    QuadraticFunction,
    predictive_coding_free_energy,
)
from active_inference.core.predictive_coding import pc_linear_fixed_point
from active_inference.estimators.predictive_coding import (
    HierarchicalPCModel,
    HierarchicalPCResult,
    MultivariatePCResult,
    PredictiveCodingResult,
    hierarchical_predictive_coding,
    multivariate_predictive_coding,
    pc_multivariate_linear_fixed_point,
    pc_parameterized_lstsq_oracle,
    predictive_coding_inference,
)

Y_OBS = 7.0


def linear_pc() -> PredictiveCodingModel:
    return PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25, m_x=4.0, s2_x=0.25)


def nonlinear_pc() -> PredictiveCodingModel:
    return PredictiveCodingModel(g=QuadraticFunction(1.0, 1.0), sigma2_y=0.25, m_x=3.0, s2_x=0.25)


# ---------------------------------------------------------------------------
# §5.2 — univariate recognition dynamics (ISC-47..50)
# ---------------------------------------------------------------------------


class TestUnivariatePC:
    def test_returns_result_with_traces(self) -> None:
        r = predictive_coding_inference(linear_pc(), Y_OBS, kappa=0.02, n_iter=400)
        assert isinstance(r, PredictiveCodingResult)
        n = len(r.mus)
        assert len(r.free_energies) == n
        assert len(r.eps_x) == n == len(r.eps_y) == len(r.mu_y)

    def test_linear_converges_to_grid_oracle(self) -> None:
        # ISC-49: PC fixed point == Chapter 4 grid posterior mean (2.4).
        r = predictive_coding_inference(linear_pc(), Y_OBS, kappa=0.02, n_iter=2000)
        lgm = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
        oracle = GridBayesianInference(lgm, np.linspace(-6, 12, 2001)).infer(Y_OBS)
        agree = oracle_agreement(r.mu_star, oracle.posterior_mean, tol=1e-2)
        assert agree.passed, agree
        assert r.mu_star == pytest.approx(2.4, abs=1e-2)

    def test_linear_monotone_free_energy(self) -> None:
        # ISC-48: small-κ gradient descent on F is non-increasing.
        r = predictive_coding_inference(linear_pc(), Y_OBS, kappa=0.02, n_iter=2000)
        assert convergence_report(r.free_energies).monotone

    def test_nonlinear_converges_to_grid_argmin(self) -> None:
        # ISC-50: nonlinear PC reaches the independent grid argmin of F_MAP.
        m = nonlinear_pc()
        r = predictive_coding_inference(m, 5.84, kappa=0.01, n_iter=5000)
        grid = np.linspace(0.0, 4.0, 8001)
        F = np.array([predictive_coding_free_energy(m, 5.84, float(u)).free_energy for u in grid])
        argmin = float(grid[int(np.argmin(F))])
        assert r.mu_star == pytest.approx(argmin, abs=1e-2)
        # Example 5.4 sensory prediction error decays toward zero.
        assert abs(r.eps_y[-1]) < abs(r.eps_y[0])

    def test_nonlinear_monotone_at_small_kappa(self) -> None:
        r = predictive_coding_inference(nonlinear_pc(), 5.84, kappa=0.01, n_iter=5000)
        assert convergence_report(r.free_energies).monotone

    def test_init_defaults_to_prior_mean(self) -> None:
        r = predictive_coding_inference(linear_pc(), Y_OBS, kappa=0.02, n_iter=1)
        assert r.mus[0] == pytest.approx(4.0)   # μ⁽⁰⁾ = m_x (Algorithm 5.2.1 line 1)


# ---------------------------------------------------------------------------
# §5.3 — multivariate PC reduces to the scalar routine (ISC-53)
# ---------------------------------------------------------------------------


class TestMultivariatePC:
    def test_returns_result(self) -> None:
        def g(x):
            return np.array([2.0 * x[0] + 3.0])

        def J(x):
            return np.array([[2.0]])

        r = multivariate_predictive_coding(
            g, J, np.array([7.0]), np.array([4.0]),
            precision_y=1 / 0.25, precision_x=1 / 0.25, kappa=0.02, n_iter=400,
        )
        assert isinstance(r, MultivariatePCResult)

    def test_reduces_to_scalar(self) -> None:
        # ISC-53: a 1-D multivariate run equals the scalar predictive_coding_inference.
        def g(x):
            return np.array([2.0 * x[0] + 3.0])

        def J(x):
            return np.array([[2.0]])

        rm = multivariate_predictive_coding(
            g, J, np.array([7.0]), np.array([4.0]),
            precision_y=1 / 0.25, precision_x=1 / 0.25, kappa=0.02, n_iter=2000,
        )
        rs = predictive_coding_inference(linear_pc(), Y_OBS, kappa=0.02, n_iter=2000)
        assert float(rm.mu_star[0]) == pytest.approx(rs.mu_star, abs=1e-6)

    def test_2d_decoupled_problem(self) -> None:
        # Two independent linear dimensions → each converges to its own oracle.
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, -1.0])

        def g(x):
            return A @ x + b

        def J(x):
            return A

        y = np.array([5.0, 2.0])      # targets
        m_x = np.array([0.0, 0.0])
        r = multivariate_predictive_coding(
            g, J, y, m_x, precision_y=np.array([1.0, 1.0]),
            precision_x=np.array([1e-6, 1e-6]), kappa=0.05, n_iter=5000,
        )
        # near-flat prior → MLE: g(x)=y ⇒ x = A⁻¹(y−b)
        expected = np.linalg.solve(A, y - b)
        assert np.allclose(r.mu_star, expected, atol=1e-2)

    def test_records_prediction_error_traces(self) -> None:
        # ISC-4/ISC-5: the result carries (T, D)/(T, C) error traces that decay.
        A = np.array([[2.0, 0.5], [-0.3, 1.5]])
        b = np.array([1.0, -1.0])
        x_true = np.array([2.0, -1.0])
        y = A @ x_true + b

        def g(x):
            return A @ x + b

        r = multivariate_predictive_coding(
            g, lambda x: A, y, np.zeros(2),
            precision_y=np.array([1.0, 1.0]), precision_x=np.array([1e-4, 1e-4]),
            kappa=0.05, n_iter=5000,
        )
        T = r.mus.shape[0]
        assert r.eps_y.shape == (T, 2)
        assert r.eps_x.shape == (T, 2)
        # Sensory error magnitude shrinks from initialization to fixed point.
        assert r.eps_y_norm[-1] < r.eps_y_norm[0]
        # Under this near-flat prior the STATE error grows: the belief leaves m_x=0
        # to fit the data (prior↔likelihood trade-off) — so "both decay to zero" is
        # the wrong narrative, and the plotter docstring says so.
        assert r.eps_x_norm[-1] > r.eps_x_norm[0]
        # The final sensory error matches y − g(μ*) recomputed independently.
        assert np.allclose(r.eps_y[-1], y - g(r.mu_star), atol=1e-9)


class TestMultivariateLinearOracle:
    def test_reduces_to_scalar_fixed_point(self) -> None:
        # ISC-2: the matrix oracle equals the scalar pc_linear_fixed_point on 1-D.
        model = linear_pc()  # g(x)=2x+3, σ_y²=0.25, m_x=4, s_x²=0.25
        scalar = pc_linear_fixed_point(model, Y_OBS)
        vec = pc_multivariate_linear_fixed_point(
            np.array([[2.0]]), np.array([3.0]), np.array([Y_OBS]), np.array([4.0]),
            precision_y=1 / 0.25, precision_x=1 / 0.25,
        )
        assert float(vec[0]) == pytest.approx(scalar, abs=1e-9)

    def test_matches_iterative_fixed_point(self) -> None:
        # ISC-3: closed-form oracle == iterative multivariate fixed point (coupled A).
        A = np.array([[2.0, 0.5], [-0.3, 1.5]])
        b = np.array([1.0, -1.0])
        y = A @ np.array([2.0, -1.0]) + b
        m_x = np.array([0.5, -0.5])
        precision_y = np.array([1.5, 0.8])
        precision_x = np.array([0.3, 0.7])
        oracle = pc_multivariate_linear_fixed_point(
            A, b, y, m_x, precision_y=precision_y, precision_x=precision_x)
        # Tight tol + ample iterations so the descent reaches the fixed point
        # (the |ΔF|<tol stop otherwise halts ~5e-5 short on this slow config).
        r = multivariate_predictive_coding(
            lambda x: A @ x + b, lambda x: A, y, m_x,
            precision_y=precision_y, precision_x=precision_x,
            kappa=0.05, n_iter=100000, tol=1e-15,
        )
        assert np.allclose(r.mu_star, oracle, atol=1e-6)

    def test_flat_prior_is_least_squares(self) -> None:
        # Π_x → 0 ⇒ closed form reduces to the OLS inverse A⁻¹(y−b).
        A = np.array([[2.0, 0.5], [-0.3, 1.5]])
        b = np.array([1.0, -1.0])
        y = A @ np.array([2.0, -1.0]) + b
        oracle = pc_multivariate_linear_fixed_point(
            A, b, y, np.zeros(2), precision_y=np.array([1.0, 1.0]),
            precision_x=np.array([1e-9, 1e-9]))
        assert np.allclose(oracle, np.linalg.solve(A, y - b), atol=1e-4)


# ---------------------------------------------------------------------------
# §5.6 — parameterized nonlinear over-determined PC (Example 5.6)
# ---------------------------------------------------------------------------

# Book §5.6 parameterized model: rectangular 4×2 mixing matrix, g(x)=Θ(x⊙x)+b.
THETA_56 = np.array([[-0.1, 0.3], [0.3, 0.4], [0.2, -0.5], [-0.1, 0.1]])
OFFSET_56 = np.ones(4)
X_TRUE_56 = np.array([0.5, 2.5])


def _param_g(x: np.ndarray) -> np.ndarray:
    return THETA_56 @ (x * x) + OFFSET_56


def _param_jac(x: np.ndarray) -> np.ndarray:
    return THETA_56 @ np.diag(2.0 * x)


class TestParameterizedOracle:
    def test_oracle_inverts_consistent_observation(self) -> None:
        # Noiseless y = g(x*) ⇒ least-squares squared-state recovery is exact.
        y = _param_g(X_TRUE_56)
        est = pc_parameterized_lstsq_oracle(THETA_56, OFFSET_56, y, sign=np.sign(X_TRUE_56))
        assert np.allclose(est, X_TRUE_56, atol=1e-9)

    def test_sign_is_configurable(self) -> None:
        # g is even in each component ⇒ oracle honours the caller-supplied sign.
        y = _param_g(X_TRUE_56)
        est = pc_parameterized_lstsq_oracle(
            THETA_56, OFFSET_56, y, sign=np.array([-1.0, 1.0]))
        assert np.allclose(est, np.array([-0.5, 2.5]), atol=1e-9)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            pc_parameterized_lstsq_oracle(THETA_56, OFFSET_56, np.ones(3))
        with pytest.raises(ValueError):
            pc_parameterized_lstsq_oracle(
                THETA_56, OFFSET_56, _param_g(X_TRUE_56), sign=np.ones(3))

    def test_flat_prior_recognition_matches_oracle(self) -> None:
        # The iterative over-determined nonlinear descent lands on the closed form.
        y = _param_g(X_TRUE_56)
        oracle = pc_parameterized_lstsq_oracle(
            THETA_56, OFFSET_56, y, sign=np.sign(X_TRUE_56))
        r = multivariate_predictive_coding(
            g=_param_g, jacobian=_param_jac, y=y, m_x=np.ones(2),
            precision_y=np.full(4, 2.0), precision_x=np.full(2, 1e-6),
            mu0=np.ones(2), kappa=0.05, n_iter=200000, tol=1e-15,
        )
        assert isinstance(r, MultivariatePCResult)
        assert np.allclose(r.mu_star, oracle, atol=1e-4)
        assert np.allclose(r.mu_star, X_TRUE_56, atol=1e-4)
        # Sensory prediction error is driven to zero (data fully explained).
        assert float(np.linalg.norm(_param_g(r.mu_star) - y)) < 1e-4

    def test_recognition_gradient_matches_finite_differences(self) -> None:
        # Prove the multivariate/over-determined recognition gradient
        # ∂F/∂μ = Π_x ε_x − Jᵀ Π_y ε_y (§5.3 Eq. 21) numerically — the same
        # expression the descent loop applies — at several off-fixed-point beliefs.
        y = _param_g(X_TRUE_56)
        Pi_y = np.diag(np.full(4, 2.0))
        Pi_x = np.diag(np.full(2, 1.5))
        m_x = np.array([1.0, 1.0])

        def free_energy(mu: np.ndarray) -> float:
            eps_y = y - _param_g(mu)
            eps_x = mu - m_x
            return float(0.5 * eps_y @ Pi_y @ eps_y + 0.5 * eps_x @ Pi_x @ eps_x)

        def analytic_grad(mu: np.ndarray) -> np.ndarray:
            eps_y = y - _param_g(mu)
            eps_x = mu - m_x
            return Pi_x @ eps_x - _param_jac(mu).T @ Pi_y @ eps_y

        for mu in (np.array([0.3, 2.0]), np.array([1.2, 3.1]), np.array([0.8, 1.4])):
            err = gradient_check_vector(free_energy, analytic_grad, mu, eps=1e-6)
            assert err < 1e-5

    def test_informative_prior_pulls_toward_prior_mean(self) -> None:
        # With a real prior the MAP belief sits between x* and m_x (precision balance).
        y = _param_g(X_TRUE_56)
        m_x = np.array([1.0, 1.0])
        r = multivariate_predictive_coding(
            g=_param_g, jacobian=_param_jac, y=y, m_x=m_x,
            precision_y=np.full(4, 2.0), precision_x=np.full(2, 2.0),
            mu0=m_x, kappa=0.05, n_iter=20000, tol=1e-14,
        )
        # Belief moved off the prior mean toward the data-consistent state, but the
        # prior keeps it strictly short of exact recovery.
        assert np.linalg.norm(r.mu_star - X_TRUE_56) > 1e-2
        assert np.linalg.norm(r.mu_star - m_x) > 1e-2


# ---------------------------------------------------------------------------
# §5.2 — precision balances the two prediction errors (Example 5.2)
# ---------------------------------------------------------------------------


class TestPrecisionBalance:
    # Book §5.2: g(x)=2x+3, ŷ=7 (data-consistent x*=2), prior m_x=4.
    SETTINGS = [(0.5, 2.0), (0.1, 1.0), (1.0, 0.1)]

    def _model(self, s2_x, sigma2_y):
        return PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=sigma2_y,
                                     m_x=4.0, s2_x=s2_x)

    def test_grid_minimum_matches_closed_form(self) -> None:
        mu = np.linspace(0.0, 6.0, 2001)
        for s2_x, sigma2_y in self.SETTINGS:
            model = self._model(s2_x, sigma2_y)
            fe = np.array([predictive_coding_free_energy(model, 7.0, float(m)).free_energy
                           for m in mu])
            grid_min = float(mu[int(np.argmin(fe))])
            oracle = pc_linear_fixed_point(model, 7.0)
            assert oracle_agreement(grid_min, oracle, tol=5e-3).passed

    def test_minimum_lies_between_data_and_prior(self) -> None:
        # The MAP belief is a precision-weighted compromise: 2 (data) ≤ μ* ≤ 4 (prior).
        for s2_x, sigma2_y in self.SETTINGS:
            mu_star = pc_linear_fixed_point(self._model(s2_x, sigma2_y), 7.0)
            assert 2.0 - 1e-9 <= mu_star <= 4.0 + 1e-9

    def test_higher_precision_ratio_pulls_toward_prior(self) -> None:
        # μ* increases monotonically with λ_x/λ_y = σ_y²/s_x² (toward m_x=4).
        ordered = sorted(self.SETTINGS, key=lambda s: s[1] / s[0])  # by ratio
        minima = [pc_linear_fixed_point(self._model(s2, sy), 7.0) for s2, sy in ordered]
        assert all(a <= b + 1e-9 for a, b in zip(minima, minima[1:]))


# ---------------------------------------------------------------------------
# §5.4 — hierarchical PC (ISC-51, ISC-52)
# ---------------------------------------------------------------------------


def example_5_7_model() -> HierarchicalPCModel:
    # Unconstrained top (m_x=0): g^[L+1]=0 ⇒ ε^[L]=μ^[L] (book p.306 / Example 5.7).
    return HierarchicalPCModel(
        generators=[QuadraticFunction(1.0, 1.0), QuadraticFunction(1.0, 1.0)],
        variances=[1.0, 1.0, 1.0],
        m_x=0.0,
    )


class TestHierarchicalPC:
    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            HierarchicalPCModel(generators=[QuadraticFunction()], variances=[1.0], m_x=0.0)
        with pytest.raises(ValueError):
            HierarchicalPCModel(generators=[QuadraticFunction()], variances=[1.0, -1.0], m_x=0.0)

    def test_returns_result(self) -> None:
        r = hierarchical_predictive_coding(example_5_7_model(), 2.0, mu0=[3.0, 3.0],
                                           kappa=0.03, n_iter=600)
        assert isinstance(r, HierarchicalPCResult)
        assert r.mus.shape[1] == 3      # nodes: y, μ¹, μ²

    def test_reproduces_example_5_7(self) -> None:
        # ISC-52: x*=1, g=x²+1 ⇒ y=2, μ¹→1, μ²→0, all ε→0, ΣF→0.
        r = hierarchical_predictive_coding(example_5_7_model(), 2.0, mu0=[3.0, 3.0],
                                           kappa=0.03, n_iter=20000, tol=1e-15)
        assert r.mu_star == pytest.approx([2.0, 1.0, 0.0], abs=1e-2)
        assert r.final_free_energy == pytest.approx(0.0, abs=1e-3)
        assert np.all(np.abs(r.errors[-1]) < 1e-2)

    def test_summed_free_energy_monotone(self) -> None:
        # ISC-51: hierarchical summed VFE is non-increasing.
        r = hierarchical_predictive_coding(example_5_7_model(), 2.0, mu0=[3.0, 3.0],
                                           kappa=0.03, n_iter=5000)
        assert convergence_report(r.free_energies).monotone

    def test_observation_node_is_clamped(self) -> None:
        r = hierarchical_predictive_coding(example_5_7_model(), 2.0, mu0=[3.0, 3.0],
                                           kappa=0.03, n_iter=500)
        # node 0 (the sensory data) never moves off y.
        assert np.allclose(r.mus[:, 0], 2.0)


# ---------------------------------------------------------------------------
# ISC-53 — the cross-chapter oracle is not a single point.
#
# Advisor blind-spot #4: "mu*=2.4 == Ch.4 grid mean" is one configuration, which
# is indistinguishable from a hardcode or a coincidence. This sweep asserts the
# linear-Gaussian PC fixed point equals BOTH (a) the closed-form Gaussian
# posterior mean and (b) Chapter 4's independent GridBayesianInference posterior
# mean, across a grid of priors, precisions, slopes and observations — so the
# match holds *for the right reason*, not at one lucky point.
# ---------------------------------------------------------------------------


def _closed_form_posterior_mean(beta0, beta1, sigma2_y, m_x, s2_x, y):
    """Exact posterior mean of x for y = beta0 + beta1*x + N(0, sigma2_y),
    prior x ~ N(m_x, s2_x). Linear-Gaussian conjugate result."""
    prec = 1.0 / s2_x + beta1**2 / sigma2_y
    num = m_x / s2_x + beta1 * (y - beta0) / sigma2_y
    return num / prec


class TestLinearOracleSweep:
    GRID = [
        (beta0, beta1, sigma2_y, m_x, s2_x, y)
        for beta0 in (0.0, 3.0)
        for beta1 in (0.5, 2.0, -1.5)
        for sigma2_y in (0.1, 0.25, 1.0)
        for m_x in (-1.0, 4.0)
        for s2_x in (0.25, 2.0)
        for y in (1.0, 7.0)
    ]

    @pytest.mark.parametrize("beta0,beta1,sigma2_y,m_x,s2_x,y", GRID)
    def test_pc_fixed_point_equals_both_oracles(
        self, beta0, beta1, sigma2_y, m_x, s2_x, y
    ) -> None:
        analytic = _closed_form_posterior_mean(beta0, beta1, sigma2_y, m_x, s2_x, y)

        # (a) Predictive coding (gradient descent on MAP free energy).
        # The MAP free energy is quadratic in mu with curvature
        # L = lambda_x + beta1^2 * lambda_y; fixed-step descent is only stable for
        # kappa < 2/L (the book's kappa-sensitivity warning, made precise). We pick
        # kappa = 0.4/L — well inside the stability bound — so descent converges for
        # every config. The fixed point itself is kappa-independent.
        pc_model = PredictiveCodingModel(
            g=LinearFunction(beta1, beta0), sigma2_y=sigma2_y, m_x=m_x, s2_x=s2_x
        )
        curvature = pc_model.lambda_x + beta1**2 * pc_model.lambda_y
        kappa = 0.4 / curvature
        pc = predictive_coding_inference(pc_model, y, kappa=kappa, n_iter=5000, tol=1e-12)
        assert pc.mu_star == pytest.approx(analytic, abs=1e-4)

        # (b) Chapter 4's grid Bayesian posterior mean — fully independent path.
        span = abs(analytic) + 10.0
        grid = np.linspace(analytic - span, analytic + span, 4001)
        lg = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y, m_x=m_x, s2_x=s2_x,
            prior_kind="gaussian",
        )
        grid_mean = GridBayesianInference(model=lg, x_grid=grid).infer(y).posterior_mean
        assert pc.mu_star == pytest.approx(grid_mean, abs=1e-3)

    def test_sweep_is_nontrivial(self) -> None:
        # Guard against the sweep silently collapsing to a constant (which would
        # make "agreement" vacuous): the oracle means must actually vary.
        means = [_closed_form_posterior_mean(*cfg) for cfg in self.GRID]
        assert np.ptp(means) > 1.0   # spread of >1 unit across the grid


class TestStationaryConvergence:
    @pytest.mark.parametrize("vector", [False, True])
    @pytest.mark.parametrize("kappa", [0.1, 1e-20, 0.02])
    def test_loss_plateau_requires_a_stationary_belief(self, vector, kappa):
        if vector:
            result = multivariate_predictive_coding(
                lambda x: 2.0 * x + 3.0,
                lambda x: np.array([[2.0]]),
                np.array([7.0]), np.array([4.0]),
                precision_y=4.0, precision_x=4.0,
                kappa=kappa, n_iter=200,
            )
            final = result.mu_star[0]
        else:
            result = predictive_coding_inference(
                linear_pc(), 7.0, kappa=kappa, n_iter=200,
            )
            final = result.mu_star
        assert np.isfinite(result.free_energies).all()
        residual = abs(20.0 * final - 48.0)
        if kappa == 0.02:
            assert result.converged
            assert final == pytest.approx(2.4, abs=1e-9)
            assert residual < 1e-9
        else:
            assert not result.converged
            assert result.n_iter_run == 200
            assert residual > 1.0
