"""Tests for shared orchestrator workflow helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.core.generative_model import LinearGaussianModel  # noqa: E402
from active_inference.orchestrator_workflows import (  # noqa: E402
    build_animation_7_5_multivariate_active_inference,
    build_example_2_7_multiple_samples,
    build_example_2_10_gradient_descent,
    build_example_3_5_bayesian_linear_regression,
    build_example_3_7_factor_analysis_em,
    build_example_6_7_multivariate_generalized_coordinates,
    build_example_7_5_multivariate_active_inference,
    build_multivariate_active_agent_env,
    build_variational_free_energy_visualization,
    sequential_inference_posteriors,
)
from active_inference.utils.grids import make_grid  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestSequentialWorkflowHelpers:
    def test_sequential_posteriors_stay_normalized(self) -> None:
        model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
        grid = make_grid(0.0, 5.0, 200)
        posts = list(sequential_inference_posteriors(model, grid, np.array([7.0, 7.2, 6.8])))
        assert [i for i, _post in posts] == [0, 1, 2]
        for _i, posterior in posts:
            np.testing.assert_allclose(np.trapezoid(posterior, grid), 1.0, rtol=1e-10)


class TestChapterWorkflowBuilders:
    def test_chapter_2_workflows_return_expected_figures(self) -> None:
        multiple = build_example_2_7_multiple_samples(seed=6, n_samples=8, x_true=2.0)
        assert set(multiple.figures) == {
            "example_2_7_ridge",
            "example_2_7_posterior",
            "example_2_7_convergence",
        }
        assert multiple.summary["sequential_mode"] == pytest.approx(multiple.summary["batch_mode"], abs=2e-3)
        descent = build_example_2_10_gradient_descent(
            seed=9,
            n_samples=60,
            x_true=2.5,
            lr=1e-4,
            max_iter=2000,
        )
        assert set(descent.figures) == {
            "example_2_10_mle_descent",
            "example_2_10_map_descent",
            "example_2_10_comparison",
        }
        assert descent.summary["mle_iter"] == pytest.approx(descent.summary["mle_closed"], abs=5e-3)

    def test_chapter_3_workflows_return_expected_figures(self) -> None:
        blr = build_example_3_5_bayesian_linear_regression(seed=4, max_n=8)
        assert set(blr.figures) == {"example_3_5_blr_posteriors", "example_3_5_blr_predictive"}
        assert np.asarray(blr.summary["posterior_mean"]).shape == (2,)
        em = build_example_3_7_factor_analysis_em(seed=6, n_samples=80, n_factors=2, max_iter=20)
        assert set(em.figures) == {"example_3_7_factor_analysis_em"}
        assert np.isfinite(em.summary["final_log_likelihood"])
        assert em.summary["rmse"] >= 0.0

    def test_chapter_6_and_7_workflows_return_exportable_arrays(self) -> None:
        generalized = build_example_6_7_multivariate_generalized_coordinates(
            seed=0,
            n_steps=30,
            dt=0.01,
            embedding_dim=2,
            gamma=2.0,
        )
        assert set(generalized.figures) == {"example_6_7_multivariate_generalized_coordinates"}
        assert generalized.arrays["x_true"].shape[0] == 31
        active = build_example_7_5_multivariate_active_inference(
            seed=0,
            n_steps=40,
            dt=0.01,
            embedding_dim=2,
            gamma=2.0,
        )
        assert set(active.figures) == {"example_7_5_multivariate_active_inference"}
        assert active.arrays["active_xs"].shape[0] == 41
        assert np.isfinite(active.summary["active_preference_error"])

    def test_chapter_7_animation_workflow_returns_animation_and_arrays(self) -> None:
        result = build_animation_7_5_multivariate_active_inference(
            seed=0,
            n_steps=30,
            dt=0.01,
            embedding_dim=2,
            gamma=2.0,
        )
        assert result.arrays["xs"].shape[0] == 31
        assert result.metadata["action_start"] == 6
        result.animation._draw_was_started = True


class TestExtrasWorkflowBuilders:
    def test_variational_free_energy_workflow_returns_source_backed_arrays(self) -> None:
        result = build_variational_free_energy_visualization(y=7.0)
        assert set(result.figures) == {"visualize_variational_free_energy"}
        assert result.arrays["free_energy"].shape == result.arrays["surprisal"].shape
        assert "source_apis" in result.metadata


class TestActiveInferenceWorkflowHelpers:
    def test_build_multivariate_agent_env_shapes_are_consistent(self) -> None:
        agent, env = build_multivariate_active_agent_env(0.5, embedding_dim=2, gamma=2.0)
        assert agent.forward_model.shape == (4, 2)
        assert env.action_matrix.shape == (2, 2)
        np.testing.assert_allclose(env.drift(np.array([6.0, -4.0])), np.zeros(2))
