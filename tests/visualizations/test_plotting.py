"""Tests for ``visualizations.plotting`` — static figure helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.core.generative_model import LinearGaussianModel  # noqa: E402
from active_inference.core.inference import GridBayesianInference  # noqa: E402
from active_inference.utils.grids import make_grid  # noqa: E402
from active_inference.visualizations.plotting import (  # noqa: E402
    confidence_ellipse,
    plot_2d_gaussian,
    plot_generating_function,
    plot_gradient_descent,
    plot_joint_heatmap,
    plot_likelihood_ridge,
    plot_precision_comparison,
    plot_prior_likelihood_posterior,
    save_or_show,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close any leftover matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def inference_result():
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    grid = make_grid(0.0, 5.0, 100)
    return GridBayesianInference(model, grid).infer(7.0)


class TestSaveOrShow:
    def test_save_to_disk(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = save_or_show(fig, tmp_path / "fig.png")
        assert out == tmp_path / "fig.png"
        assert (tmp_path / "fig.png").exists()

    def test_returns_none_when_no_path(self) -> None:
        fig, _ = plt.subplots()
        out = save_or_show(fig, save_path=None, show=False)
        assert out is None


class TestPriorLikelihoodPosterior:
    def test_returns_three_axes(self, inference_result) -> None:
        fig = plot_prior_likelihood_posterior(inference_result, truth=2.0)
        assert len(fig.axes) == 3

    def test_save_path(self, inference_result, tmp_path: Path) -> None:
        out_path = tmp_path / "plp.png"
        plot_prior_likelihood_posterior(
            inference_result, save_path=out_path, truth=2.0,
        )
        assert out_path.exists()


class TestGeneratingFunction:
    def test_with_samples(self, tmp_path: Path) -> None:
        x = np.linspace(0, 5, 50)
        y = 2 * x + 3
        out = tmp_path / "gen.png"
        plot_generating_function(
            x, y,
            samples_x=np.array([1.0, 2.0]),
            samples_y=np.array([5.0, 7.0]),
            save_path=out,
        )
        assert out.exists()


class TestRidgePlot:
    def test_multiple_likelihoods(self, tmp_path: Path) -> None:
        x = np.linspace(0, 5, 100)
        likelihoods = [np.exp(-((x - mu) ** 2)) for mu in (1.5, 2.0, 2.5)]
        out = tmp_path / "ridge.png"
        plot_likelihood_ridge(x, likelihoods, save_path=out)
        assert out.exists()


class TestJointHeatmap:
    def test_renders(self, tmp_path: Path) -> None:
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        joint = np.outer(np.sin(x * np.pi), np.cos(y * np.pi)) ** 2
        out = tmp_path / "joint.png"
        plot_joint_heatmap(x, y, joint, save_path=out)
        assert out.exists()


class TestGradientDescentPlot:
    def test_renders_with_truth(self, tmp_path: Path) -> None:
        history = np.linspace(5.0, 2.0, 30)
        losses = (history - 2.0) ** 2
        out = tmp_path / "gd.png"
        plot_gradient_descent(history, losses, truth=2.0, save_path=out)
        assert out.exists()


class TestPrecisionComparison:
    def test_renders(self, inference_result, tmp_path: Path) -> None:
        out = tmp_path / "prec.png"
        plot_precision_comparison(
            [("balanced", inference_result), ("again", inference_result)],
            save_path=out,
        )
        assert out.exists()


class TestEllipse:
    def test_confidence_ellipse_is_patch(self) -> None:
        e = confidence_ellipse(np.array([0.0, 0.0]), np.eye(2))
        # Width and height are 2 * n_std * sqrt(eigvals) — for I and n_std=2 → 4.
        assert e.width == pytest.approx(4.0)
        assert e.height == pytest.approx(4.0)

    def test_2d_gaussian_renders(self, tmp_path: Path) -> None:
        out = tmp_path / "g2d.png"
        plot_2d_gaussian(
            mean=np.array([1.0, 1.0]),
            cov=np.array([[0.5, 0.1], [0.1, 0.5]]),
            samples=np.random.default_rng(0).normal(size=(20, 2)) + 1.0,
            truth=np.array([1.0, 1.0]),
            save_path=out,
        )
        assert out.exists()
