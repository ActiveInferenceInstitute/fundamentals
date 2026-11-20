"""Regression tests for the unified Chapter 4–5 visualization layer.

Advisor blind-spots #5 and #6: "visually confirmed" is not a quoted artifact, and
the shared :func:`plot_recognition_dynamics` could silently break Chapter 4 output.
These tests exercise the shared figure functions on *real* result objects from
*both* chapters and assert structure programmatically (panel counts, the plotted
belief trace, axis labels) — so the shared path is under test, not just eyeballed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for structural assertions

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.core.generative_model import LinearGaussianModel  # noqa: E402
from active_inference.core.continuous_learning import (  # noqa: E402
    ContinuousHierarchyLayer,
    HierarchicalContinuousModel,
    LearningAttentionModel,
    LearningAttentionState,
)
from active_inference.estimators.continuous_learning import simulate_learning_attention  # noqa: E402
from active_inference.core.predictive_coding import (  # noqa: E402
    LinearFunction,
    PredictiveCodingModel,
    QuadraticFunction,
)
from active_inference.estimators.predictive_coding import (  # noqa: E402
    HierarchicalPCModel,
    hierarchical_predictive_coding,
    multivariate_predictive_coding,
    pc_multivariate_linear_fixed_point,
    predictive_coding_inference,
)
from active_inference.estimators.variational import fixed_form_vi  # noqa: E402
from active_inference.visualizations.unified import (  # noqa: E402
    plot_correlated_embedding_precision,
    plot_discrete_belief_sequence,
    plot_hierarchical_pc,
    plot_hierarchical_message_passing,
    plot_generalized_vector_filter,
    plot_learning_attention,
    plot_multivariate_active_inference,
    plot_multivariate_pc,
    plot_policy_efe_decomposition,
    plot_prediction_errors,
    plot_recognition_dynamics,
)

Y_OBS = 7.0


@pytest.fixture(autouse=True)
def _close_figs():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _ch4_result():
    """A real Chapter 4 FixedFormResult (no eps_x/eps_y attributes)."""
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25,
                                prior_kind="gaussian")
    return fixed_form_vi(model, Y_OBS, np.linspace(-6.0, 12.0, 2001))


def _ch5_result():
    """A real Chapter 5 PredictiveCodingResult (carries eps_x/eps_y)."""
    model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                  m_x=4.0, s2_x=0.25)
    return predictive_coding_inference(model, Y_OBS, kappa=0.02, n_iter=2000)


class TestSharedRecognitionDynamics:
    def test_ch4_result_renders_two_panels(self) -> None:
        # FixedFormResult has no prediction-error traces → 2 panels (belief, F).
        fig = plot_recognition_dynamics(_ch4_result(), truth=2.0)
        assert len(fig.axes) == 2

    def test_ch5_result_renders_three_panels(self) -> None:
        # PredictiveCodingResult carries eps_x/eps_y → 3 panels (belief, errors, F).
        fig = plot_recognition_dynamics(_ch5_result(), truth=2.0, oracle=2.4)
        assert len(fig.axes) == 3

    def test_belief_trace_matches_result(self) -> None:
        # The plotted belief line IS the result's mu trace (not a relabelled mock).
        res = _ch5_result()
        fig = plot_recognition_dynamics(res)
        belief_ax = fig.axes[0]
        line = belief_ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), np.asarray(res.mus), rtol=0, atol=0)
        # and it lands on the cross-chapter oracle (2.4) it shares with Chapter 4.
        assert float(res.mus[-1]) == pytest.approx(2.4, abs=1e-3)

    def test_axes_are_labelled(self) -> None:
        fig = plot_recognition_dynamics(_ch5_result())
        assert fig.axes[0].get_xlabel() == "iteration"
        assert all(ax.get_xlabel() == "iteration" for ax in fig.axes)

    def test_every_panel_has_a_legend(self) -> None:
        # "More legends": each panel must carry a legend.
        fig = plot_recognition_dynamics(_ch5_result(), truth=2.0, oracle=2.4)
        assert all(ax.get_legend() is not None for ax in fig.axes)

    def test_stat_boxes_report_real_numbers(self) -> None:
        # "More statistics": the belief panel's stat box must quote μ* and the
        # oracle error — and they must match the result, not be decorative.
        res = _ch5_result()
        fig = plot_recognition_dynamics(res, truth=2.0, oracle=2.4)
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "μ*" in texts and "oracle" in texts
        assert f"{float(res.mus[-1]):.4f}" in texts          # exact μ* quoted
        oracle_error = abs(float(res.mus[-1]) - 2.4)
        assert f"|μ*−o| = {oracle_error:.2e}" in texts

    def test_fixed_point_is_annotated_on_curve(self) -> None:
        # "Analytical annotations": μ* is marked as a point on the belief curve.
        res = _ch5_result()
        fig = plot_recognition_dynamics(res)
        annotated = any(rf"{float(res.mus[-1]):.4f}" in t.get_text()
                        for t in fig.axes[0].texts)
        assert annotated

    def test_rejects_object_without_traces(self) -> None:
        class Empty:
            pass

        with pytest.raises(TypeError):
            plot_recognition_dynamics(Empty())


class TestPredictionErrorsFigure:
    def test_two_panels_and_min_marked(self) -> None:
        model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                      m_x=4.0, s2_x=0.25)
        grid = np.linspace(0.0, 5.0, 401)
        fig = plot_prediction_errors(model, Y_OBS, grid, truth=2.0)
        assert len(fig.axes) == 2
        # The free-energy panel's minimum sits at the analytic posterior mean 2.4.
        from active_inference.core.predictive_coding import predictive_coding_free_energy

        fvals = [predictive_coding_free_energy(model, Y_OBS, float(m)).free_energy
                 for m in grid]
        assert float(grid[int(np.argmin(fvals))]) == pytest.approx(2.4, abs=0.02)

    def test_analytic_fixed_point_in_stat_box(self) -> None:
        # The free-energy panel must quote the closed-form (analytic) minimizer.
        model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                      m_x=4.0, s2_x=0.25)
        fig = plot_prediction_errors(model, Y_OBS, np.linspace(0.0, 5.0, 401),
                                     truth=2.0)
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "analytic" in texts and "2.4000" in texts
        assert "L (curv)" in texts          # curvature reported


class TestHierarchicalFigure:
    def test_three_panels_and_node_count(self) -> None:
        model = HierarchicalPCModel(
            generators=[QuadraticFunction(1.0, 1.0), QuadraticFunction(1.0, 1.0)],
            variances=[1.0, 1.0, 1.0], m_x=0.0,
        )
        res = hierarchical_predictive_coding(model, 2.0, mu0=[3.0, 3.0],
                                             kappa=0.03, n_iter=2000)
        fig = plot_hierarchical_pc(res, truth=[2.0, 1.0, 0.0])
        assert len(fig.axes) == 3
        # belief panel has one line per node (L+1 = 3) plus the truth axhlines.
        belief_ax = fig.axes[0]
        node_lines = [ln for ln in belief_ax.lines if ln.get_label().startswith("$")]
        assert len(node_lines) == 3
        # final beliefs reproduce Example 5.7.
        assert res.mu_star == pytest.approx([2.0, 1.0, 0.0], abs=1e-2)


class TestGeneralizedFilterFigure:
    def test_three_panels_with_tracking(self) -> None:
        from active_inference.core.generalized_filtering import DynamicStateSpaceModel
        from active_inference.estimators.generalized_filtering import (
            DynamicProcess, generalized_filter, simulate_dynamic_process)
        from active_inference.visualizations.unified import plot_generalized_filter
        proc = DynamicProcess(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0))
        tr = simulate_dynamic_process(proc, x0=5.0, n_steps=300, dt=0.01,
                                      rng=np.random.default_rng(0))
        model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0),
                                       g=LinearFunction(1.0, -3.0), s2_x=5.0, sigma2_y=0.02)
        res = generalized_filter(model, tr.ys, dt=0.01, kappa=0.1, mu0=15.0)
        fig = plot_generalized_filter(res, truth=tr.xs, dt=0.01)
        assert len(fig.axes) == 3
        # belief line is the result's mu trace.
        belief_ax = fig.axes[0]
        belief_line = [ln for ln in belief_ax.lines if "mu" in ln.get_label()][0]
        np.testing.assert_allclose(belief_line.get_ydata(), np.asarray(res.mus))

    def test_correlated_precision_heatmap_is_labelled(self) -> None:
        from active_inference.core.generalized_filtering import correlated_embedding_precision

        gammas = np.array([1.0, 2.0])
        precisions = [correlated_embedding_precision(1.0, 4, gamma=float(g)) for g in gammas]
        fig = plot_correlated_embedding_precision(precisions, gammas)
        assert len(fig.axes) == 3  # two heatmaps plus colorbar
        assert all(ax.get_xlabel() for ax in fig.axes[:2])
        assert fig.axes[0].images

    def test_vector_generalized_filter_figure_has_four_panels(self) -> None:
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            MultivariateDynamicModel,
            correlated_embedding_precision,
        )
        from active_inference.estimators.generalized_filtering import (
            MultivariateDynamicProcess,
            generalized_measurements_from_series,
            generalized_vector_filter,
            multivariate_generalized_filter,
            simulate_multivariate_process,
        )

        A = np.array([[0.0, 1.0], [-4.0 / 3.0, 0.0]])
        b = np.array([0.0, 20.0 / 3.0])
        f = LinearVectorFunction(A, b)
        g = LinearVectorFunction(np.eye(2), np.array([-3.0, -3.0]))
        trace = simulate_multivariate_process(
            MultivariateDynamicProcess(f=f, g=g, omega_x=0.0, omega_y=0.0),
            x0=np.array([0.0, 5.0]),
            n_steps=120,
            dt=0.01,
            rng=np.random.default_rng(0),
        )
        y_tilde = generalized_measurements_from_series(trace.ys, embedding_dim=3, dt=0.01)
        ordinary = multivariate_generalized_filter(
            MultivariateDynamicModel(f=f, g=g, precision_x=0.5, precision_y=10.0, dim_x=2, dim_y=2),
            trace.ys,
            dt=0.01,
            kappa=1.0,
            mu0=np.array([8.0, 8.0]),
        )
        result = generalized_vector_filter(
            GeneralizedVectorModel(
                f=f,
                g=g,
                precision_x=correlated_embedding_precision(np.eye(2) * 0.5, 3, gamma=2.0),
                precision_y=correlated_embedding_precision(np.eye(2) * 10.0, 3, gamma=2.0),
                embedding_dim=3,
                dim_x=2,
                dim_y=2,
            ),
            y_tilde,
            dt=0.01,
            kappa=1.0,
        )
        fig = plot_generalized_vector_filter(result, trace.xs, ordinary=ordinary, dt=0.01)
        assert len(fig.axes) == 4
        assert all(ax.get_title() for ax in fig.axes)

    def test_multivariate_active_inference_figure_has_accessible_panels(self) -> None:
        from active_inference.core.active_inference import MultivariateActiveInferenceAgent
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            correlated_embedding_precision,
        )
        from active_inference.estimators.active_inference import (
            MultivariateActiveEnvironment,
            simulate_multivariate_active_inference,
        )

        M = 2
        preference = np.zeros(2)
        exogenous = np.array([6.0, -4.0])
        model = GeneralizedVectorModel(
            f=LinearVectorFunction(-np.eye(2), preference),
            g=LinearVectorFunction(np.eye(2), np.zeros(2)),
            precision_x=correlated_embedding_precision(np.eye(2), M, gamma=2.0),
            precision_y=correlated_embedding_precision(np.eye(2) * 20.0, M, gamma=2.0),
            embedding_dim=M,
            dim_x=2,
            dim_y=2,
        )
        forward = np.zeros((M * 2, 2))
        forward[[0, M], :] = np.eye(2)
        result = simulate_multivariate_active_inference(
            MultivariateActiveInferenceAgent(model, forward_model=forward, kappa_x=0.4, kappa_a=0.5),
            MultivariateActiveEnvironment(
                drift=LinearVectorFunction(-np.eye(2), exogenous),
                g=LinearVectorFunction(np.eye(2), np.zeros(2)),
                action_matrix=np.eye(2),
            ),
            x0=np.array([4.0, -2.0]),
            mu0_tilde=np.zeros((M, 2)),
            n_steps=120,
            dt=0.01,
            action_start=30,
            rng=np.random.default_rng(0),
        )
        fig = plot_multivariate_active_inference(result, preference=preference, exogenous=exogenous, dt=0.01)
        assert len(fig.axes) == 4
        assert all(ax.get_xlabel() for ax in fig.axes)


class TestDiscreteBeliefSequenceFigure:
    def test_two_panels_without_vfe(self) -> None:
        beliefs = np.array([[0.6, 0.4], [0.25, 0.75], [0.1, 0.9]])
        fig = plot_discrete_belief_sequence(
            beliefs,
            observations=["water", "hot", "bright"],
            state_labels=["rain", "sun"],
        )
        # Heatmap + colorbar + line panel.
        assert len(fig.axes) == 3
        line_ax = fig.axes[1]
        assert len(line_ax.lines) == 2
        np.testing.assert_allclose(line_ax.lines[0].get_ydata(), beliefs[:, 0])

    def test_three_content_panels_with_vfe(self) -> None:
        beliefs = np.array([[0.6, 0.4], [0.25, 0.75], [0.1, 0.9]])
        fig = plot_discrete_belief_sequence(
            beliefs,
            state_labels=["rain", "sun"],
            free_energies=np.array([1.2, 0.9, 0.7]),
        )
        # Heatmap + colorbar + trajectory + VFE.
        assert len(fig.axes) == 4
        assert fig.axes[2].get_ylabel() == r"$\mathcal{F}$"

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            plot_discrete_belief_sequence(np.array([0.5, 0.5]))


class TestPolicyEFEDecompositionFigure:
    def _traces(self):
        from active_inference.core.pomdp import POMDPModel, evaluate_policy_components, softmax
        A = np.array([[0.6, 0.15], [0.4, 0.85]])
        D = np.array([0.5, 0.5])
        B = np.array([
            [[0.9, 0.9], [0.1, 0.1]],
            [[0.1, 0.1], [0.9, 0.9]],
        ])
        m = POMDPModel(A=A, D=D, B=B)
        C = softmax(np.array([1.5, 0.0]))
        traces = [
            evaluate_policy_components(m, D, [0], C),
            evaluate_policy_components(m, D, [1], C),
        ]
        q = softmax(-np.array([t.total for t in traces]))
        return traces, q

    def test_renders_decomposition_and_step_trace(self) -> None:
        traces, q = self._traces()
        fig = plot_policy_efe_decomposition(
            traces,
            policy_labels=["exploit", "explore"],
            posterior=q,
        )
        assert len(fig.axes) == 2
        # First panel: risk bars + ambiguity bars + total-G markers.
        assert len(fig.axes[0].patches) >= 4
        assert fig.axes[1].lines[0].get_ydata()[0] == pytest.approx(traces[0].total)

    def test_rejects_empty_trace_list(self) -> None:
        with pytest.raises(ValueError):
            plot_policy_efe_decomposition([])


class TestChapter8Figures:
    def _learning_result(self):
        ys = np.full(120, 2.0)
        model = LearningAttentionModel(
            state_attractor=5.0,
            theta_prior_mean=0.0,
            zeta_prior_mean=0.0,
            sigma2_y=0.02,
            sigma2_theta=50.0,
            sigma2_zeta=50.0,
        )
        return simulate_learning_attention(
            model,
            ys,
            initial=LearningAttentionState(mu_x=2.0, mu_theta=0.0, mu_zeta=0.0),
            dt=0.03,
            kappa_x=0.7,
            kappa_theta=2.0,
            kappa_zeta=0.25,
            damping=1.2,
        )

    def test_learning_attention_figure_is_accessibly_labelled(self) -> None:
        res = self._learning_result()
        fig = plot_learning_attention(res, truth_x=np.full_like(res.mus, 5.0), truth_theta=3.0)
        assert len(fig.axes) == 4
        assert all(ax.get_title() for ax in fig.axes)
        assert all(ax.get_xlabel() for ax in fig.axes[:3])
        assert fig.axes[0].get_legend() is not None
        assert fig.get_size_inches()[0] >= 12.0

    def test_hierarchical_message_passing_figure_has_named_nodes(self) -> None:
        model = HierarchicalContinuousModel(
            lower=ContinuousHierarchyLayer(obs_offset=3.0, sensory_precision=20.0),
            link_precision=2.0,
            context_prior_mean=5.0,
            context_precision=0.5,
        )
        fig = plot_hierarchical_message_passing(model, y=2.0, belief=np.array([4.0, 2.0]))
        assert len(fig.axes) == 1
        labels = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "sensory" in labels
        assert "lower state" in labels
        assert "upper context" in labels


class TestParameterLearningFigure:
    """Validate the Chapter 10 ``plot_parameter_learning`` figure (Figs 10.1.3/10.1.4)."""

    def _result(self):
        from active_inference.estimators.pomdp import simulate_array_learning
        A_true = np.array([[0.7, 0.6], [0.3, 0.4]])
        B_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        return simulate_array_learning(A_true=A_true, B_true=B_true, learn="A",
                                       n_trials=3, steps_per_trial=200), A_true

    def test_two_panels_and_truth_dots(self) -> None:
        from active_inference.visualizations.unified import plot_parameter_learning
        res, A_true = self._result()
        fig = plot_parameter_learning(res.A_history, res.a_confidence, truth=A_true, symbol="A")
        assert len(fig.axes) == 2
        # Left panel: one line per matrix entry (4) plus 4 truth-dot markers.
        left = fig.axes[0]
        line_entries = [ln for ln in left.lines if ln.get_label().startswith("$A")]
        assert len(line_entries) == 4
        # Final plotted probability equals the learned A (last history row), per entry.
        learned = res.A_history[-1].ravel()
        for k, ln in enumerate(line_entries):
            assert ln.get_ydata()[-1] == pytest.approx(learned[k])

    def test_works_without_truth(self) -> None:
        from active_inference.visualizations.unified import plot_parameter_learning
        res, _ = self._result()
        fig = plot_parameter_learning(res.A_history, res.a_confidence, symbol="A")
        assert len(fig.axes) == 2


class TestMultivariatePCFigure:
    """plot_multivariate_pc — §5.3 routed through the unified vocabulary."""

    @staticmethod
    def _result():
        A = np.array([[2.0, 0.5], [-0.3, 1.5]])
        b = np.array([1.0, -1.0])
        x_true = np.array([2.0, -1.0])
        y = A @ x_true + b
        precision_y = np.array([1.0, 1.0])
        precision_x = np.array([1e-4, 1e-4])
        res = multivariate_predictive_coding(
            lambda x: A @ x + b, lambda x: A, y, np.zeros(2),
            precision_y=precision_y, precision_x=precision_x, kappa=0.05, n_iter=5000,
        )
        oracle = pc_multivariate_linear_fixed_point(
            A, b, y, np.zeros(2), precision_y=precision_y, precision_x=precision_x)
        return res, x_true, oracle

    def test_three_panels_with_error_traces(self) -> None:
        res, x_true, oracle = self._result()
        fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle)
        # Error traces present ⇒ 3 panels, each labelled with a legend.
        assert len(fig.axes) == 3
        assert all(ax.get_xlabel() for ax in fig.axes)
        assert all(ax.get_legend() is not None for ax in fig.axes)

    def test_panel1_plots_belief_components(self) -> None:
        res, x_true, oracle = self._result()
        fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle)
        belief_lines = [ln for ln in fig.axes[0].lines if ln.get_label().startswith(r"$\mu")]
        assert len(belief_lines) == res.mus.shape[1]
        # The plotted belief line IS the result's μ trace (not a redraw).
        assert np.allclose(belief_lines[0].get_ydata(), res.mus[:, 0])

    def test_error_panel_plots_both_error_norms(self) -> None:
        res, x_true, oracle = self._result()
        fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle)
        labels = [ln.get_label() for ln in fig.axes[1].lines]
        assert any("varepsilon_x" in lab for lab in labels)
        assert any("varepsilon_y" in lab for lab in labels)

    def test_stat_box_quotes_real_mu_star_and_oracle_error(self) -> None:
        res, x_true, oracle = self._result()
        fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle)
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        # The real fixed point appears in the stat box, component by component.
        for v in res.mu_star:
            assert f"{v:.3g}" in texts
        # The reported oracle gap matches the recomputed norm.
        gap = float(np.linalg.norm(res.mu_star - oracle))
        assert f"{gap:.2e}" in texts

    def test_two_panels_without_error_traces(self) -> None:
        # Back-compat: a result with empty eps traces falls back to 2 panels.
        from active_inference.estimators.predictive_coding import MultivariatePCResult
        bare = MultivariatePCResult(
            mus=np.array([[0.0, 0.0], [1.0, -0.5], [2.0, -1.0]]),
            free_energies=np.array([8.0, 1.0, 0.0]),
            mu_star=np.array([2.0, -1.0]), converged=True, n_iter_run=2,
        )
        fig = plot_multivariate_pc(bare, truth=[2.0, -1.0])
        assert len(fig.axes) == 2
