"""Tests for ``core.compose`` — pipeline + running statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.compose import (
    Pipeline,
    RunningPosteriorStats,
    running_stats,
)
from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.generative_process import LinearGaussianProcess
from active_inference.utils.grids import make_grid


class TestPipelineConstruction:
    def test_linear_gaussian_factory(self) -> None:
        p = Pipeline.linear_gaussian(
            beta0=3.0, beta1=2.0, sigma2_y=0.5,
            m_x=4.0, s2_x=1.0,
            rng=np.random.default_rng(0),
        )
        assert isinstance(p.process, LinearGaussianProcess)
        assert isinstance(p.model, LinearGaussianModel)
        assert p.x_grid.size == 500

    def test_invalid_grid_raises(self) -> None:
        process = LinearGaussianProcess(beta0=0, beta1=1, sigma2_y=1.0)
        model = LinearGaussianModel(beta0=0, beta1=1, sigma2_y=1.0,
                                    m_x=0, s2_x=1)
        with pytest.raises(ValueError):
            Pipeline(process=process, model=model, x_grid=np.array([0.0]))


class TestPipelineRun:
    def test_round_trip_recovers_truth(self) -> None:
        p = Pipeline.linear_gaussian(
            beta0=3.0, beta1=2.0, sigma2_y=0.25,
            m_x=4.0, s2_x=4.0,    # wide prior, data dominates
            rng=np.random.default_rng(0),
        )
        result = p.run(x_star=2.0, n=200)
        assert abs(result.posterior_mode - 2.0) < 0.1

    def test_sample_then_infer_matches_run(self) -> None:
        p = Pipeline.linear_gaussian(
            beta0=3.0, beta1=2.0, sigma2_y=0.25,
            m_x=4.0, s2_x=1.0,
            rng=np.random.default_rng(1),
        )
        ys = p.sample(x_star=2.5, n=10).flatten()
        manual = p.infer(ys)
        # Same observations → same posterior.
        assert manual.posterior_mode == pytest.approx(
            p._inferer.infer(ys).posterior_mode
        )


class TestRunningStats:
    def test_shape_contract(self) -> None:
        rng = np.random.default_rng(0)
        process = LinearGaussianProcess(
            beta0=3.0, beta1=2.0, sigma2_y=0.4, rng=rng,
        )
        ys = process.sample(2.0, n=30).flatten()
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.4,
            m_x=4.0, s2_x=1.0,
        )
        x_grid = make_grid(0, 5, 200)
        stats = running_stats(model, x_grid, ys)
        assert isinstance(stats, RunningPosteriorStats)
        assert stats.n_axis.shape == (30,)
        assert stats.means.shape == (30,)
        assert stats.stds.shape == (30,)
        assert stats.kl_from_prior.shape == (30,)
        assert stats.log_evidences.shape == (30,)
        assert stats.posteriors.shape == (30, 200)

    def test_std_decreases_with_n(self) -> None:
        rng = np.random.default_rng(2)
        process = LinearGaussianProcess(
            beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng,
        )
        ys = process.sample(2.0, n=50).flatten()
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.5,
            m_x=4.0, s2_x=4.0,
        )
        stats = running_stats(model, make_grid(0, 5, 300), ys)
        # Std should monotonically shrink as N grows (allowing tiny noise).
        diffs = np.diff(stats.stds)
        assert np.all(diffs <= 1e-3)

    def test_kl_increases_with_n(self) -> None:
        rng = np.random.default_rng(3)
        process = LinearGaussianProcess(
            beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng,
        )
        ys = process.sample(2.0, n=30).flatten()
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.5,
            m_x=4.0, s2_x=4.0,
        )
        stats = running_stats(model, make_grid(0, 5, 300), ys)
        # KL from prior must grow as more data shifts the posterior away.
        assert stats.kl_from_prior[-1] > stats.kl_from_prior[0]

    def test_log_evidence_monotone(self) -> None:
        # Each new observation can only decrease (never increase) the
        # cumulative log evidence in absolute terms.
        rng = np.random.default_rng(4)
        process = LinearGaussianProcess(
            beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng,
        )
        ys = process.sample(2.0, n=20).flatten()
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.5,
            m_x=4.0, s2_x=4.0,
        )
        stats = running_stats(model, make_grid(0, 5, 300), ys)
        # Cumulative log-evidence is monotonically decreasing.
        diffs = np.diff(stats.log_evidences)
        assert np.all(diffs <= 0)

    def test_invalid_grid_raises(self) -> None:
        model = LinearGaussianModel(beta0=0, beta1=1, sigma2_y=1.0,
                                    m_x=0, s2_x=1)
        with pytest.raises(ValueError):
            running_stats(model, np.array([0.0]), np.array([0.5]))


class TestRunningStatsSummary:
    def test_summary_string(self) -> None:
        rng = np.random.default_rng(5)
        ys = LinearGaussianProcess(
            beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng,
        ).sample(2.0, n=10).flatten()
        model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.5,
                                    m_x=4.0, s2_x=1.0)
        s = running_stats(model, make_grid(0, 5, 200), ys).summary()
        assert "RunningPosteriorStats" in s
        assert "final_mean" in s
        assert "final_KL" in s
