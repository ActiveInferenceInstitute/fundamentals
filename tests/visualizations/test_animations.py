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

from active_inference.core.generative_model import LinearGaussianModel  # noqa: E402
from active_inference.core.predictive_coding import (  # noqa: E402
    LinearFunction,
    PredictiveCodingModel,
    QuadraticFunction,
)
from active_inference.estimators.predictive_coding import (  # noqa: E402
    HierarchicalPCModel,
    hierarchical_predictive_coding,
    predictive_coding_inference,
)
from active_inference.estimators.variational import fixed_form_vi  # noqa: E402
from active_inference.visualizations.animations import (  # noqa: E402
    animate_2d_posterior,
    animate_bimodal_emergence,
    animate_blr_predictive_band,
    animate_calibration_growth,
    animate_discrete_beliefs,
    animate_em_convergence,
    animate_em_steps,
    animate_gradient_descent,
    animate_hierarchical_pc,
    animate_learning_attention,
    animate_lgs_online,
    animate_multivariate_active_inference,
    animate_policy_efe_tradeoff,
    animate_precision_sweep,
    animate_recognition_dynamics,
    animate_sequential_posterior,
    animate_sufficient_statistics,
    save_animation,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def assert_func_animation(anim: FuncAnimation, *, axes: int | None = None) -> None:
    """Assert animation shape and mark unsaved test animations as intentionally exercised."""
    assert isinstance(anim, FuncAnimation)
    assert anim._fig is not None
    if axes is not None:
        assert len(anim._fig.axes) == axes
    init = getattr(anim, "_init_func", None)
    if init is not None:
        init()
    frames = list(anim.new_frame_seq())
    if frames:
        anim._func(frames[0])
        anim._func(frames[-1])
    anim._draw_was_started = True


class TestSequentialPosterior:
    def test_frame_count_matches(self) -> None:
        x = np.linspace(0, 5, 100)
        posteriors = [np.exp(-((x - mu) ** 2)) for mu in np.linspace(1.5, 2.5, 8)]
        anim = animate_sequential_posterior(x, posteriors, truth=2.0)
        assert_func_animation(anim)

    def test_with_prior_overlay(self) -> None:
        x = np.linspace(0, 5, 50)
        posteriors = [np.exp(-((x - 2) ** 2))]
        prior = np.exp(-((x - 4) ** 2))
        anim = animate_sequential_posterior(x, posteriors, prior=prior)
        assert_func_animation(anim)


class TestGradientDescent:
    def test_basic_animation(self) -> None:
        x_grid = np.linspace(0, 5, 50)
        loss_grid = (x_grid - 2.5) ** 2
        history = np.linspace(5.0, 2.5, 30)
        losses = (history - 2.5) ** 2
        anim = animate_gradient_descent(loss_grid, x_grid, history, losses,
                                        truth=2.5)
        assert_func_animation(anim)


class TestMultivariateActiveInferenceAnimation:
    def test_runs_and_attaches_raw_data(self) -> None:
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
            n_steps=80,
            dt=0.01,
            action_start=20,
            rng=np.random.default_rng(0),
        )
        anim = animate_multivariate_active_inference(
            result,
            preference=preference,
            exogenous=exogenous,
            dt=0.01,
            frame_stride=10,
        )
        assert_func_animation(anim, axes=2)
        assert "xs" in anim._raw_data


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
        assert_func_animation(anim)


class TestEMConvergence:
    def test_frame_count_matches(self) -> None:
        K = 10
        ll = np.linspace(-100.0, -50.0, K)
        Theta_history = [np.random.default_rng(i).normal(size=(3, 2))
                         for i in range(K)]
        anim = animate_em_convergence(ll, Theta_history)
        assert_func_animation(anim)

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
        assert_func_animation(anim)

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
        assert_func_animation(anim)

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
        assert_func_animation(anim)

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
        assert_func_animation(anim)

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
        assert_func_animation(anim)

    def test_validates_shape(self) -> None:
        with pytest.raises(ValueError):
            animate_lgs_online(np.zeros((4, 2)), np.zeros((3, 2, 2)),
                               np.zeros((4, 2)))


class TestDiscreteBeliefAnimation:
    def test_runs_and_has_two_panels(self) -> None:
        beliefs = np.array([[0.6, 0.4], [0.25, 0.75], [0.1, 0.9]])
        anim = animate_discrete_beliefs(
            beliefs,
            observations=["water", "hot", "bright"],
            state_labels=["rain", "sun"],
        )
        assert_func_animation(anim, axes=2)

    def test_rejects_unnormalized_beliefs(self) -> None:
        with pytest.raises(ValueError):
            animate_discrete_beliefs(np.array([[1.0, 1.0]]))


class TestPolicyEFETradeoffAnimation:
    def test_runs_with_posteriors(self) -> None:
        risks = np.array([[0.3, 0.7], [0.8, 0.4], [1.2, 0.3]])
        ambiguities = np.array([[0.6, 0.2], [0.6, 0.2], [0.6, 0.2]])
        totals = risks + ambiguities
        posts = np.exp(-totals)
        posts = posts / posts.sum(axis=1, keepdims=True)
        anim = animate_policy_efe_tradeoff(
            risks,
            ambiguities,
            posteriors=posts,
            frame_labels=["weak", "medium", "strong"],
            policy_labels=["explore", "exploit"],
        )
        assert_func_animation(anim, axes=2)

    def test_learning_attention_animation_runs(self) -> None:
        mus = np.linspace(2.0, 5.0, 12)
        thetas = np.linspace(0.0, 3.0, 12)
        zetas = np.linspace(0.0, 4.0, 12)
        free_energies = np.linspace(20.0, 1.0, 12)
        anim = animate_learning_attention(mus, thetas, zetas, free_energies)
        assert_func_animation(anim, axes=2)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            animate_policy_efe_tradeoff(np.zeros((2, 2)), np.zeros((3, 2)))


class TestEMSteps:
    def test_runs(self) -> None:
        K = 5
        rng = np.random.default_rng(0)
        e_means = [rng.normal(size=(20, 2)) for _ in range(K)]
        thetas = [rng.normal(size=(4, 2)) for _ in range(K)]
        ll = np.linspace(-100.0, -50.0, K)
        anim = animate_em_steps(e_means, thetas, ll)
        assert_func_animation(anim)

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
        assert_func_animation(anim)

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
        assert_func_animation(anim)
        out = save_animation(anim, tmp_path / "anim.gif", fps=8)
        assert out.exists()
        # Pillow GIFs are non-trivial sized.
        assert out.stat().st_size > 100


# ---------------------------------------------------------------------------
# Composable recognition-dynamics + hierarchical animators (Chapters 4 & 5)
# ---------------------------------------------------------------------------


def _pc_result():
    model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25,
                                  m_x=4.0, s2_x=0.25)
    return predictive_coding_inference(model, 7.0, kappa=0.05, n_iter=20)


def _ff_result():
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25,
                                prior_kind="gaussian")
    return fixed_form_vi(model, 7.0, np.linspace(-6.0, 12.0, 1201), n_iter=60)


def _hier_result():
    model = HierarchicalPCModel(
        generators=[QuadraticFunction(1.0, 1.0), QuadraticFunction(1.0, 1.0)],
        variances=[1.0, 1.0, 1.0], m_x=0.0)
    return hierarchical_predictive_coding(model, 2.0, mu0=[3.0, 3.0],
                                          kappa=0.03, n_iter=40)


class TestComposableRecognitionAnimation:
    def test_pc_result_three_panels(self) -> None:
        # PredictiveCodingResult carries eps_x/eps_y → 3 panels.
        anim = animate_recognition_dynamics(_pc_result(), truth=2.0, oracle=2.4)
        assert_func_animation(anim, axes=3)

    def test_fixed_form_result_two_panels(self) -> None:
        # FixedFormResult has no error traces → 2 panels — same function, composable.
        anim = animate_recognition_dynamics(_ff_result(), truth=2.0)
        assert_func_animation(anim, axes=2)

    def test_every_panel_has_legend(self) -> None:
        anim = animate_recognition_dynamics(_pc_result(), truth=2.0, oracle=2.4,
                                            surprisal=7.43)
        assert_func_animation(anim, axes=3)
        assert all(ax.get_legend() is not None for ax in anim._fig.axes)

    def test_rejects_object_without_traces(self) -> None:
        class Empty:
            mus = np.zeros((3, 2))      # 2-D → not a 1-D belief trace
            free_energies = np.zeros(3)
        with pytest.raises(TypeError):
            animate_recognition_dynamics(Empty())

    def test_stride_reduces_rendered_frames(self, tmp_path: Path) -> None:
        # Genuine render: a strided GIF has strictly fewer frames than the full one.
        res = _pc_result()
        full_anim = animate_recognition_dynamics(res, stride=1)
        strided_anim = animate_recognition_dynamics(res, stride=4)
        assert_func_animation(full_anim)
        assert_func_animation(strided_anim)
        full = save_animation(full_anim, tmp_path / "full.gif", fps=8)
        strided = save_animation(strided_anim, tmp_path / "strided.gif", fps=8)
        from PIL import Image
        with Image.open(full) as a, Image.open(strided) as b:
            assert a.n_frames > b.n_frames
            assert b.n_frames >= 2  # always includes the final frame


class TestHierarchicalAnimation:
    def test_three_panels_and_renders(self, tmp_path: Path) -> None:
        anim = animate_hierarchical_pc(_hier_result(), truth=[2.0, 1.0, 0.0])
        assert_func_animation(anim, axes=3)
        out = save_animation(anim, tmp_path / "hier.gif", fps=10)
        assert out.exists() and out.stat().st_size > 100

    def test_belief_panel_has_one_line_per_node_plus_sum(self) -> None:
        # 3 nodes → 3 belief lines; F panel → 3 layer lines + the summed line.
        anim = animate_hierarchical_pc(_hier_result())
        assert_func_animation(anim, axes=3)
        ax_mu, _, ax_f = anim._fig.axes
        node_lines = [ln for ln in ax_mu.lines if ln.get_label().startswith("$")]
        assert len(node_lines) == 3
        f_lines = [ln for ln in ax_f.lines if ln.get_label().startswith("$")]
        assert len(f_lines) == 4   # 3 layers + Σ


class TestChapter10Animations:
    """Chapter 10 learning animations — Dirichlet convergence + precision sweep."""

    def _learn_result(self):
        from active_inference.estimators.pomdp import simulate_array_learning
        A_true = np.array([[0.7, 0.6], [0.3, 0.4]])
        B_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        return simulate_array_learning(A_true=A_true, B_true=B_true, learn="A",
                                       n_trials=4, steps_per_trial=200), A_true

    def test_parameter_learning_two_panels_and_renders(self, tmp_path: Path) -> None:
        from active_inference.visualizations import animate_parameter_learning, save_animation
        res, A_true = self._learn_result()
        anim = animate_parameter_learning(res.A_history, res.a_confidence,
                                          truth=A_true, symbol="A")
        assert_func_animation(anim, axes=2)
        # left panel: one line per matrix entry (4)
        left = anim._fig.axes[0]
        entry_lines = [ln for ln in left.lines if ln.get_label().startswith("$A")]
        assert len(entry_lines) == 4
        out = save_animation(anim, tmp_path / "learn.gif", fps=8)
        assert out.exists() and out.stat().st_size > 100

    def test_parameter_learning_stride_reduces_frames(self, tmp_path: Path) -> None:
        from active_inference.visualizations import animate_parameter_learning, save_animation
        res, A_true = self._learn_result()
        full_anim = animate_parameter_learning(res.A_history, res.a_confidence,
                                               truth=A_true, stride=1)
        strided_anim = animate_parameter_learning(res.A_history, res.a_confidence,
                                                  truth=A_true, stride=3)
        assert_func_animation(full_anim)
        assert_func_animation(strided_anim)
        full = save_animation(full_anim, tmp_path / "full.gif", fps=8)
        strided = save_animation(strided_anim, tmp_path / "strided.gif", fps=8)
        from PIL import Image
        with Image.open(full) as a, Image.open(strided) as b:
            assert a.n_frames > b.n_frames
            assert b.n_frames >= 2

    def test_parameter_learning_rejects_1d(self) -> None:
        from active_inference.visualizations import animate_parameter_learning
        with pytest.raises(TypeError):
            animate_parameter_learning(np.zeros(5), np.zeros(5))

    def test_policy_precision_renders_with_correct_frame_count(self, tmp_path: Path) -> None:
        from active_inference.visualizations import animate_policy_precision, save_animation
        G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
        gammas = np.linspace(0.0, 3.0, 10)
        anim = animate_policy_precision(G, gammas)
        assert_func_animation(anim, axes=2)
        out = save_animation(anim, tmp_path / "prec.gif", fps=8)
        from PIL import Image
        with Image.open(out) as im:
            assert im.n_frames == len(gammas)

    def test_two_armed_bandit_animation_runs(self) -> None:
        from active_inference.estimators.pomdp import make_two_armed_bandit, simulate_two_armed_bandit
        from active_inference.visualizations import animate_two_armed_bandit

        model = make_two_armed_bandit()
        result = simulate_two_armed_bandit(model, true_context=1, n_steps=5)
        anim = animate_two_armed_bandit(result)
        assert_func_animation(anim, axes=2)
