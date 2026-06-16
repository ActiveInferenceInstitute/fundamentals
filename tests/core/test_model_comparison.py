"""Tests for Bayesian model-comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.model_comparison import (
    bayes_factor,
    bayesian_model_average,
    bayesian_model_expansion,
    bayesian_model_reduction,
    log_bayes_factor,
    model_posterior,
)


class TestModelPosterior:
    def test_normalizes_log_evidence_stably(self) -> None:
        posterior = model_posterior([1000.0, 1000.0 + np.log(3.0)])
        np.testing.assert_allclose(posterior, [0.25, 0.75], rtol=1e-12)
        np.testing.assert_allclose(posterior.sum(), 1.0)

    def test_rejects_non_vector_inputs(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            model_posterior([[0.0, 1.0]])
        with pytest.raises(ValueError, match="finite"):
            model_posterior([0.0, np.nan])


class TestModelScores:
    def test_bayes_factor_and_averaging_match_hand_calculation(self) -> None:
        assert log_bayes_factor(2.5, 1.0) == pytest.approx(1.5)
        assert bayes_factor(np.log(6.0), np.log(2.0)) == pytest.approx(3.0)
        average = bayesian_model_average([10.0, 20.0], [np.log(1.0), np.log(3.0)])
        assert average == pytest.approx(17.5)

    def test_reduction_and_expansion_apply_accuracy_complexity_terms(self) -> None:
        assert bayesian_model_reduction(4.0, accuracy_delta=0.5, complexity_delta=1.25) == pytest.approx(5.75)
        assert bayesian_model_expansion(4.0, accuracy_gain=1.5, complexity_cost=0.25) == pytest.approx(5.25)
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_model_expansion(0.0, accuracy_gain=1.0, complexity_cost=-0.1)
