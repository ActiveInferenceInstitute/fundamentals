"""Tests for ``core.inference`` — grid Bayesian inference."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.inference import GridBayesianInference, InferenceResult


@pytest.fixture
def x_grid() -> np.ndarray:
    return np.linspace(0.0, 5.0, 1001)


def make_model(**overrides) -> LinearGaussianModel:
    kwargs = dict(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                  m_x=4.0, s2_x=0.25, prior_kind="gaussian")
    kwargs.update(overrides)
    return LinearGaussianModel(**kwargs)


class TestSinglePoint:
    def test_posterior_normalizes(self, x_grid: np.ndarray) -> None:
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        assert np.trapezoid(res.posterior, x_grid) == pytest.approx(1.0, rel=1e-4)

    def test_mode_between_prior_and_inverse(self, x_grid: np.ndarray) -> None:
        # Tight prior at 4, y = 7 inverts to 2 → mode in (2, 4).
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        assert 2.0 < res.posterior_mode < 4.0

    def test_uniform_prior_recovers_inverse(self, x_grid: np.ndarray) -> None:
        m = make_model(prior_kind="uniform",
                       uniform_low=0.0, uniform_high=5.0,
                       sigma2_y=1e-3)
        res = GridBayesianInference(m, x_grid).infer(7.0)
        assert res.posterior_mode == pytest.approx(2.0, abs=1e-2)

    def test_credible_interval_brackets_mode(self, x_grid: np.ndarray) -> None:
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        lo, hi = res.credible_interval(0.95)
        assert lo < res.posterior_mode < hi
        assert lo < res.posterior_mean < hi

    def test_credible_interval_invalid_mass(self, x_grid: np.ndarray) -> None:
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        with pytest.raises(ValueError):
            res.credible_interval(0.0)
        with pytest.raises(ValueError):
            res.credible_interval(1.0)


class TestBatchInference:
    def test_batch_matches_sequential(self, x_grid: np.ndarray) -> None:
        rng = np.random.default_rng(123)
        ys = rng.normal(loc=7.0, scale=0.5, size=20)
        m = make_model()
        batch_res = GridBayesianInference(m, x_grid).infer(ys)

        log_state = m.log_prior(x_grid).copy()
        for y in ys:
            log_state += m.log_likelihood(float(y), x_grid)
        log_state -= np.max(log_state)
        seq = np.exp(log_state)
        seq /= np.trapezoid(seq, x_grid)
        np.testing.assert_allclose(batch_res.posterior, seq, atol=1e-6)

    def test_batch_tightens_with_n(self, x_grid: np.ndarray) -> None:
        rng = np.random.default_rng(7)
        m = make_model(sigma2_y=1.0, s2_x=4.0)
        var_history = []
        for n in (1, 5, 20, 100):
            ys = rng.normal(loc=7.0, scale=1.0, size=n)
            res = GridBayesianInference(m, x_grid).infer(ys)
            var_history.append(res.posterior_variance)
        diffs = np.diff(var_history)
        assert np.all(diffs <= 1e-10)


class TestJointDensity:
    def test_joint_normalizes(self, x_grid: np.ndarray) -> None:
        m = make_model(prior_kind="uniform",
                       uniform_low=0.0, uniform_high=5.0,
                       sigma2_y=0.5)
        x, y, joint = GridBayesianInference(m, x_grid).joint_density()
        marg_x = np.trapezoid(joint, y, axis=0)
        total = np.trapezoid(marg_x, x)
        assert total == pytest.approx(1.0, rel=1e-4)


class TestGridValidation:
    def test_grid_too_small(self) -> None:
        with pytest.raises(ValueError):
            GridBayesianInference(make_model(), np.array([0.0]))

    def test_grid_must_be_1d(self) -> None:
        with pytest.raises(ValueError):
            GridBayesianInference(make_model(), np.array([[0.0, 1.0]]))


class TestInferenceResultDataclass:
    def test_attributes_match_grid(self, x_grid: np.ndarray) -> None:
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        assert isinstance(res, InferenceResult)
        assert res.x_grid.shape == res.posterior.shape == res.likelihood.shape
        assert np.isfinite(res.log_evidence)

    def test_summary_contains_mode_and_evidence(self,
                                                 x_grid: np.ndarray) -> None:
        res = GridBayesianInference(make_model(), x_grid).infer(7.0)
        s = res.summary()
        assert "mode=" in s
        assert "log_evidence=" in s
