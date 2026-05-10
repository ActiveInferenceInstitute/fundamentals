"""Tests for ``visualizations.diagnostics`` — statistical-figure helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.core.diagnostics import (  # noqa: E402
    CalibrationCurve,
    PosteriorPredictiveCheck,
)
from active_inference.visualizations.diagnostics import (  # noqa: E402
    plot_calibration,
    plot_cdf_comparison,
    plot_coverage_curve,
    plot_kl_trace,
    plot_posterior_predictive_check,
    plot_qq,
    plot_running_statistics,
    plot_score_trace,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


class TestCDF:
    def test_renders(self, tmp_path: Path) -> None:
        x = np.linspace(0, 1, 50)
        cdfs = [np.cumsum(np.linspace(0, 1, 50)) / np.sum(np.linspace(0, 1, 50)),
                np.linspace(0, 1, 50)]
        out = tmp_path / "cdf.png"
        plot_cdf_comparison(x, cdfs, ["uneven", "uniform"], save_path=out)
        assert out.exists()

    def test_validates_label_length(self) -> None:
        x = np.linspace(0, 1, 10)
        with pytest.raises(ValueError):
            plot_cdf_comparison(x, [np.ones(10)], labels=["a", "b"])

    def test_validates_shape(self) -> None:
        x = np.linspace(0, 1, 10)
        with pytest.raises(ValueError):
            plot_cdf_comparison(x, [np.ones(11)], labels=["bad"])


class TestQQ:
    def test_renders_for_normal_samples(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        s = rng.normal(size=200)
        out = tmp_path / "qq.png"
        plot_qq(s, save_path=out)
        assert out.exists()

    def test_too_few_samples(self) -> None:
        with pytest.raises(ValueError):
            plot_qq(np.array([0.5]))

    def test_unsupported_distribution(self) -> None:
        with pytest.raises(NotImplementedError):
            plot_qq(np.zeros(10), distribution="laplace")


class TestCalibration:
    def test_renders(self, tmp_path: Path) -> None:
        curve = CalibrationCurve(
            nominal=np.array([0.5, 0.8, 0.95]),
            empirical=np.array([0.51, 0.79, 0.94]),
            n_trials=200,
        )
        out = tmp_path / "calibration.png"
        plot_calibration(curve, save_path=out)
        assert out.exists()


class TestCoverage:
    def test_renders(self, tmp_path: Path) -> None:
        n_axis = np.array([10, 20, 50, 100])
        coverage = np.array([0.92, 0.94, 0.95, 0.95])
        out = tmp_path / "coverage.png"
        plot_coverage_curve(n_axis, coverage, save_path=out)
        assert out.exists()


class TestKLTrace:
    def test_renders(self, tmp_path: Path) -> None:
        out = tmp_path / "kl.png"
        plot_kl_trace(np.arange(1, 21),
                      np.linspace(0.0, 5.0, 20),
                      save_path=out)
        assert out.exists()


class TestRunningStats:
    def test_renders(self, tmp_path: Path) -> None:
        n = np.arange(1, 51)
        means = 2.0 + np.cumsum(np.zeros_like(n))
        stds = 1.0 / np.sqrt(n)
        out = tmp_path / "running.png"
        plot_running_statistics(n, means, stds, truth=2.0, save_path=out)
        assert out.exists()


class TestScoreTrace:
    def test_renders_with_crps(self, tmp_path: Path) -> None:
        n = np.arange(1, 21)
        log_scores = -n.astype(float) * 0.5
        crps = 1.0 / n.astype(float)
        out = tmp_path / "scores.png"
        plot_score_trace(n, log_scores, crps, save_path=out)
        assert out.exists()

    def test_renders_without_crps(self, tmp_path: Path) -> None:
        n = np.arange(1, 21)
        out = tmp_path / "scores_lonly.png"
        plot_score_trace(n, -n.astype(float) * 0.5, save_path=out)
        assert out.exists()


class TestPPC:
    def test_renders(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        check = PosteriorPredictiveCheck(
            observed=0.0,
            replicated=rng.normal(size=400),
            p_value=0.5,
        )
        out = tmp_path / "ppc.png"
        plot_posterior_predictive_check(check, save_path=out)
        assert out.exists()
