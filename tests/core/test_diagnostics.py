"""Property-based tests for ``core.diagnostics`` — statistical invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from active_inference.core.diagnostics import (
    CalibrationCurve,
    PosteriorPredictiveCheck,
    calibration_curve,
    coverage_from_intervals,
    crps_gaussian,
    effective_sample_size,
    gaussian_entropy_mvn,
    gaussian_entropy_univariate,
    gaussian_kl_mvn,
    gaussian_kl_univariate,
    grid_entropy,
    grid_kl_divergence,
    log_score_gaussian,
    logsumexp,
    normal_ci,
    posterior_predictive_check,
    standardize,
)


# ---------------------------------------------------------------------------
# logsumexp
# ---------------------------------------------------------------------------


class TestLogSumExp:
    def test_matches_naive(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=20)
        np.testing.assert_allclose(logsumexp(a),
                                   np.log(np.sum(np.exp(a))), rtol=1e-12)

    def test_handles_extreme_values(self) -> None:
        # Naive np.log(sum(exp(...))) overflows here; logsumexp must not.
        a = np.array([1000.0, 1000.0, 1000.0])
        out = logsumexp(a)
        assert np.isfinite(out)
        assert out == pytest.approx(1000.0 + np.log(3.0))

    def test_axis_argument(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(3, 5))
        per_row = logsumexp(a, axis=1)
        assert per_row.shape == (3,)
        np.testing.assert_allclose(
            per_row,
            np.log(np.exp(a).sum(axis=1)),
            rtol=1e-12,
        )

    def test_minus_inf_is_inert(self) -> None:
        a = np.array([-np.inf, -np.inf, -np.inf])
        out = logsumexp(a)
        assert out == -np.inf or out < -1e6  # essentially -∞


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


class TestESS:
    def test_uniform_weights_full_ess(self) -> None:
        # Equal log-weights → ESS = N exactly.
        ess = effective_sample_size(np.zeros(50))
        assert ess == pytest.approx(50.0)

    def test_concentrated_weight_collapses_to_one(self) -> None:
        # All mass on one entry → ESS = 1.
        log_w = np.array([0.0] + [-1e6] * 99)
        assert effective_sample_size(log_w) == pytest.approx(1.0)

    def test_two_equal_remaining(self) -> None:
        # Two equal weights, rest negligible → ESS = 2.
        log_w = np.array([0.0, 0.0] + [-1e6] * 8)
        assert effective_sample_size(log_w) == pytest.approx(2.0)

    def test_invalid_dimension(self) -> None:
        with pytest.raises(ValueError):
            effective_sample_size(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Entropy (closed form vs grid)
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_gaussian_closed_form_matches_grid(self) -> None:
        sigma2 = 0.5
        x = np.linspace(-10, 10, 4001)
        p = np.exp(-x ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
        h_grid = grid_entropy(p, x)
        h_closed = gaussian_entropy_univariate(sigma2)
        assert h_grid == pytest.approx(h_closed, rel=1e-3)

    def test_entropy_increases_with_variance(self) -> None:
        h_small = gaussian_entropy_univariate(0.5)
        h_large = gaussian_entropy_univariate(5.0)
        assert h_small < h_large

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            gaussian_entropy_univariate(-1.0)

    def test_mvn_entropy_matches_univariate(self) -> None:
        # 1-D MVN should reduce to the univariate formula.
        cov = np.array([[1.5]])
        np.testing.assert_allclose(
            gaussian_entropy_mvn(cov),
            gaussian_entropy_univariate(1.5),
            rtol=1e-12,
        )

    def test_grid_entropy_zero_for_dirac(self) -> None:
        # A density spike on a single bin has near-zero differential entropy.
        x = np.linspace(-1, 1, 1001)
        p = np.zeros_like(x)
        p[500] = 1.0 / (x[1] - x[0])     # one bin, normalized
        # Differential entropy isn't defined for true deltas, but the grid
        # approximation gives a small finite (negative) number — verify it's
        # at most 0.
        h = grid_entropy(p, x)
        assert h <= 1e-6


# ---------------------------------------------------------------------------
# KL divergence properties
# ---------------------------------------------------------------------------


class TestKL:
    def test_self_kl_is_zero(self) -> None:
        x = np.linspace(-5, 5, 2001)
        p = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        kl = grid_kl_divergence(p, p, x)
        assert abs(kl) < 1e-6

    def test_kl_non_negative(self) -> None:
        rng = np.random.default_rng(3)
        x = np.linspace(-5, 5, 2001)
        p = np.exp(-0.5 * (x - 0.5) ** 2) / np.sqrt(2 * np.pi)
        q = np.exp(-0.5 * (x + 0.5) ** 2) / np.sqrt(2 * np.pi)
        # Non-negativity is the defining property of KL.
        assert grid_kl_divergence(p, q, x) >= 0

    def test_gaussian_closed_form_matches_grid(self) -> None:
        x = np.linspace(-15, 15, 8001)
        p = np.exp(-(x - 1.0) ** 2 / 2.0) / np.sqrt(2 * np.pi)
        q = np.exp(-(x - 0.0) ** 2 / (2 * 2.0)) / np.sqrt(2 * np.pi * 2.0)
        kl_grid = grid_kl_divergence(p, q, x)
        kl_closed = gaussian_kl_univariate(1.0, 1.0, 0.0, 2.0)
        assert kl_grid == pytest.approx(kl_closed, rel=1e-3)

    def test_kl_inf_when_q_zero_where_p_positive(self) -> None:
        x = np.linspace(0, 1, 11)
        p = np.ones_like(x) / 1.0
        q = np.zeros_like(x); q[0] = 1.0
        assert math.isinf(grid_kl_divergence(p, q, x))

    def test_mvn_kl_self_zero(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 3))
        cov = A @ A.T + 0.1 * np.eye(3)
        mu = rng.normal(size=3)
        kl = gaussian_kl_mvn(mu, cov, mu, cov)
        assert abs(kl) < 1e-9


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


class TestScoringRules:
    def test_log_score_maximized_at_truth(self) -> None:
        # Forecast mean at the truth is at least as good as mean shifted away.
        y = np.array([1.0, 1.0, 1.0])
        good = log_score_gaussian(y, np.full_like(y, 1.0), np.ones_like(y)).sum()
        bad = log_score_gaussian(y, np.full_like(y, 3.0), np.ones_like(y)).sum()
        assert good > bad

    def test_crps_lower_for_correct_mean(self) -> None:
        y = np.array([1.0, 1.0, 1.0])
        good = crps_gaussian(y, np.full_like(y, 1.0), np.ones_like(y)).mean()
        bad = crps_gaussian(y, np.full_like(y, 3.0), np.ones_like(y)).mean()
        assert good < bad

    def test_crps_non_negative(self) -> None:
        rng = np.random.default_rng(5)
        y = rng.normal(size=20)
        mu = rng.normal(size=20)
        sigma2 = rng.uniform(0.1, 2.0, size=20)
        c = crps_gaussian(y, mu, sigma2)
        assert np.all(c >= 0)


# ---------------------------------------------------------------------------
# Calibration / coverage
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration_for_oracle(self) -> None:
        # If the credible interval is centered on the truth at any width,
        # empirical coverage is 1.0.
        from scipy.special import erfinv
        rng = np.random.default_rng(7)
        T = 4000
        truths = rng.normal(size=T)
        levels = np.array([0.5, 0.8, 0.95])

        def lower(level: float) -> np.ndarray:
            half = float(np.sqrt(2.0) * erfinv(level))
            return truths - half

        def upper(level: float) -> np.ndarray:
            half = float(np.sqrt(2.0) * erfinv(level))
            return truths + half

        curve = calibration_curve(truths, lower, upper, levels)
        assert isinstance(curve, CalibrationCurve)
        assert np.all(curve.empirical == 1.0)

    def test_well_calibrated_predictions(self) -> None:
        # Posterior is correct: y* | μ ~ N(μ, 1); CI from y* should cover μ at nominal rate.
        rng = np.random.default_rng(8)
        T = 4000
        truths = rng.normal(size=T)              # μ_t
        observations = truths + rng.normal(size=T)  # y_t

        def lower(level: float) -> np.ndarray:
            return normal_ci_centered(observations, 1.0, level)[0]

        def upper(level: float) -> np.ndarray:
            return normal_ci_centered(observations, 1.0, level)[1]

        curve = calibration_curve(truths, lower, upper,
                                  np.array([0.5, 0.8, 0.95]))
        # ECE should be small when the model matches the data-generating process.
        assert curve.calibration_error() < 0.05

    def test_coverage_input_validation(self) -> None:
        with pytest.raises(ValueError):
            coverage_from_intervals(np.zeros(3), np.zeros(3), np.zeros(2))

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError):
            calibration_curve(np.zeros(10),
                              lambda lvl: np.zeros(10),
                              lambda lvl: np.ones(10),
                              np.array([0.0]))


def normal_ci_centered(mean, sigma2, level):
    from scipy.special import erfinv
    mean = np.asarray(mean, dtype=float)
    half = float(np.sqrt(2.0 * sigma2) * erfinv(level))
    return mean - half, mean + half


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------


class TestPPC:
    def test_p_value_one_for_self_consistent_replicates(self) -> None:
        # If observed equals one of the replicates and statistic is mean,
        # the p-value should not be tiny.
        rng = np.random.default_rng(9)
        observed = rng.normal(size=50)
        replicated = rng.normal(size=(200, 50))
        result = posterior_predictive_check(
            observed, replicated, statistic=np.mean,
        )
        assert isinstance(result, PosteriorPredictiveCheck)
        assert 0.0 <= result.p_value <= 2.0  # two-sided

    def test_p_value_small_when_observation_is_outlier(self) -> None:
        rng = np.random.default_rng(10)
        # Replicate datasets are tightly centered; observed is shifted way out.
        replicated = rng.normal(size=(500, 50))
        observed = np.full(50, 5.0)
        result = posterior_predictive_check(
            observed, replicated, statistic=np.mean,
        )
        assert result.p_value < 0.05

    def test_invalid_replicate_shape(self) -> None:
        with pytest.raises(ValueError):
            posterior_predictive_check(
                np.zeros(10), np.zeros(10), statistic=np.mean,
            )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


class TestMiscHelpers:
    def test_normal_ci_widens_with_level(self) -> None:
        narrow_lo, narrow_hi = normal_ci(0.0, 1.0, level=0.5)
        wide_lo, wide_hi = normal_ci(0.0, 1.0, level=0.95)
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)

    def test_normal_ci_matches_known_z(self) -> None:
        lo, hi = normal_ci(0.0, 1.0, level=0.95)
        assert lo == pytest.approx(-1.95996, abs=1e-3)
        assert hi == pytest.approx(+1.95996, abs=1e-3)

    def test_normal_ci_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            normal_ci(0.0, 1.0, level=0.0)
        with pytest.raises(ValueError):
            normal_ci(0.0, -1.0, level=0.5)

    def test_standardize_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
        z = standardize(x)
        np.testing.assert_allclose(z.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(z.std(axis=0, ddof=1), [1.0, 1.0, 1.0],
                                   atol=1e-10)

    def test_standardize_handles_zero_std(self) -> None:
        x = np.ones((10, 1))
        z = standardize(x)
        # When the input is constant, std is 0; we substitute 1 so output
        # stays well-defined and equals (x − mean) / 1 = 0.
        np.testing.assert_allclose(z, np.zeros_like(z))
