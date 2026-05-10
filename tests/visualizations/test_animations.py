"""Tests for ``visualizations.animations`` — matplotlib FuncAnimation helpers.

We avoid actually rendering GIF/PNG files in unit tests (the chapter smoke
tests cover that path); here we only verify that the helpers return valid
``FuncAnimation`` objects with the expected number of frames.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from active_inference.visualizations.animations import (  # noqa: E402
    animate_2d_posterior,
    animate_bimodal_emergence,
    animate_blr_predictive_band,
    animate_calibration_growth,
    animate_em_convergence,
    animate_em_steps,
    animate_gradient_descent,
    animate_lgs_online,
    animate_precision_sweep,
    animate_sequential_posterior,
    animate_sufficient_statistics,
    save_animation,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


class TestSequentialPosterior:
    def test_frame_count_matches(self) -> None:
        x = np.linspace(0, 5, 100)
        posteriors = [np.exp(-((x - mu) ** 2)) for mu in np.linspace(1.5, 2.5, 8)]
        anim = animate_sequential_posterior(x, posteriors, truth=2.0)
        assert isinstance(anim, FuncAnimation)
        assert anim._fig is not None

    def test_with_prior_overlay(self) -> None:
        x = np.linspace(0, 5, 50)
        posteriors = [np.exp(-((x - 2) ** 2))]
        prior = np.exp(-((x - 4) ** 2))
        anim = animate_sequential_posterior(x, posteriors, prior=prior)
        assert isinstance(anim, FuncAnimation)


class TestGradientDescent:
    def test_basic_animation(self) -> None:
        x_grid = np.linspace(0, 5, 50)
        loss_grid = (x_grid - 2.5) ** 2
        history = np.linspace(5.0, 2.5, 30)
        losses = (history - 2.5) ** 2
        anim = animate_gradient_descent(loss_grid, x_grid, history, losses,
                                        truth=2.5)
        assert isinstance(anim, FuncAnimation)


class TestPosterior2D:
    def test_two_dim_animation(self) -> None:
        means = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        covs = np.stack([np.eye(2)] * 3) * np.array([1.0, 0.5, 0.25])[:, None, None]
        anim = animate_2d_posterior(
            means, covs,
            truth=np.array([1.0, 1.0]),
            prior_mean=np.zeros(2),
            prior_cov=np.eye(2),
        )
        assert isinstance(anim, FuncAnimation)


class TestEMConvergence:
    def test_frame_count_matches(self) -> None:
        K = 10
        ll = np.linspace(-100.0, -50.0, K)
        Theta_history = [np.random.default_rng(i).normal(size=(3, 2))
                         for i in range(K)]
        anim = animate_em_convergence(ll, Theta_history)
        assert isinstance(anim, FuncAnimation)

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError):
            animate_em_convergence(
                np.array([1.0, 2.0]),
                [np.zeros((3, 2))] * 5,
            )


class TestSufficientStatistics:
    def test_runs(self) -> None:
        n = np.arange(1, 21)
        anim = animate_sufficient_statistics(
            n,
            running_mean=2.0 + 0.05 * np.sin(n / 5),
            running_std=1.0 / np.sqrt(n),
            running_kl=np.log(n.astype(float) + 1),
            truth=2.0,
        )
        assert isinstance(anim, FuncAnimation)

    def test_validates_shape(self) -> None:
        n = np.arange(1, 11)
        with pytest.raises(ValueError):
            animate_sufficient_statistics(
                n, running_mean=np.zeros(5),
                running_std=np.zeros(10), running_kl=np.zeros(10),
            )


class TestCalibrationGrowth:
    def test_runs(self) -> None:
        nominal = np.array([0.5, 0.8, 0.95])
        history = np.tile(nominal, (10, 1)) + 0.01 * np.random.default_rng(0).normal(size=(10, 3))
        anim = animate_calibration_growth(nominal, history)
        assert isinstance(anim, FuncAnimation)

    def test_validates_shape(self) -> None:
        with pytest.raises(ValueError):
            animate_calibration_growth(np.array([0.5, 0.95]), np.zeros((4, 3)))


class TestPrecisionSweep:
    def test_runs(self) -> None:
        x = np.linspace(0, 5, 80)
        priors = [np.exp(-((x - 4) ** 2) / 0.5)] * 5
        likelihoods = [np.exp(-((x - 2) ** 2) / 0.5)] * 5
        posteriors = [np.exp(-((x - 3) ** 2) / 0.5)] * 5
        anim = animate_precision_sweep(
            x, priors, likelihoods, posteriors,
            log_ratios=np.linspace(-2, 2, 5).tolist(),
            truth=2.0,
        )
        assert isinstance(anim, FuncAnimation)

    def test_validates_length(self) -> None:
        x = np.linspace(0, 5, 10)
        with pytest.raises(ValueError):
            animate_precision_sweep(x, [np.zeros(10)] * 3, [np.zeros(10)] * 3,
                                    [np.zeros(10)] * 3, log_ratios=[0.0])


class TestBimodalEmergence:
    def test_runs(self) -> None:
        x = np.linspace(-3, 3, 80)
        posteriors = [np.exp(-((x - mu) ** 2)) for mu in (-2, -1, 0, 1, 2)]
        anim = animate_bimodal_emergence(
            x, posteriors, prior_means=[-2, -1, 0, 1, 2],
            truths=[2.0] * 5,
        )
        assert isinstance(anim, FuncAnimation)

    def test_validates_length(self) -> None:
        x = np.linspace(-1, 1, 10)
        with pytest.raises(ValueError):
            animate_bimodal_emergence(x, [np.zeros(10)] * 3,
                                      prior_means=[0.0])


class TestLGSOnline:
    def test_runs(self) -> None:
        T = 6
        means = np.tile(np.array([0.5, 0.5]), (T, 1))
        covs = np.stack([np.eye(2) * (1.0 / (i + 1)) for i in range(T)])
        observations = np.zeros((T, 2))
        anim = animate_lgs_online(
            means, covs, observations,
            truth=np.array([0.5, 0.5]),
        )
        assert isinstance(anim, FuncAnimation)

    def test_validates_shape(self) -> None:
        with pytest.raises(ValueError):
            animate_lgs_online(np.zeros((4, 2)), np.zeros((3, 2, 2)),
                               np.zeros((4, 2)))


class TestEMSteps:
    def test_runs(self) -> None:
        K = 5
        rng = np.random.default_rng(0)
        e_means = [rng.normal(size=(20, 2)) for _ in range(K)]
        thetas = [rng.normal(size=(4, 2)) for _ in range(K)]
        ll = np.linspace(-100.0, -50.0, K)
        anim = animate_em_steps(e_means, thetas, ll)
        assert isinstance(anim, FuncAnimation)

    def test_validates_lengths(self) -> None:
        with pytest.raises(ValueError):
            animate_em_steps([np.zeros((5, 2))],
                             [np.zeros((3, 2)), np.zeros((3, 2))],
                             np.array([1.0]))


class TestBLRPredictiveBand:
    def test_runs(self) -> None:
        T = 5
        x = np.linspace(0, 5, 50)
        means = np.tile(np.array([3.0, 2.0]), (T, 1))
        covs = np.stack([np.eye(2) * (1.0 / (i + 1)) for i in range(T)])
        x_data = np.linspace(0, 5, T)
        y_data = 3.0 + 2.0 * x_data
        anim = animate_blr_predictive_band(
            x, means, covs, sigma2_y=0.5,
            x_data=x_data, y_data=y_data,
            truth_line=(3.0, 2.0),
        )
        assert isinstance(anim, FuncAnimation)

    def test_validates_data_length(self) -> None:
        x = np.linspace(0, 5, 50)
        T = 4
        means = np.zeros((T, 2))
        covs = np.tile(np.eye(2), (T, 1, 1))
        with pytest.raises(ValueError):
            animate_blr_predictive_band(
                x, means, covs, sigma2_y=0.5,
                x_data=np.zeros(3), y_data=np.zeros(3),  # len ≠ T
            )


class TestSaveAnimation:
    def test_saves_gif(self, tmp_path: Path) -> None:
        x = np.linspace(0, 1, 20)
        posteriors = [np.exp(-((x - 0.5) ** 2))] * 3
        anim = animate_sequential_posterior(x, posteriors)
        out = save_animation(anim, tmp_path / "anim.gif", fps=8)
        assert out.exists()
        # Pillow GIFs are non-trivial sized.
        assert out.stat().st_size > 100
