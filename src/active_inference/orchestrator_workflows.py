"""Reusable workflows behind chapter and extras orchestrator scripts.

The executable files under ``chapters/`` and ``extras/`` stay intentionally
thin: parse CLI flags, choose output paths, and persist figures/data. This
module owns the reusable numerical and plotting workflows that would otherwise
make those wrappers exceed the repository's orchestrator contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .core.active_inference import MultivariateActiveInferenceAgent
from .core.distributions import diagonal_cov, mvn_sample
from .core.generalized_filtering import (
    GeneralizedVectorModel,
    LinearVectorFunction,
    MultivariateDynamicModel,
    correlated_embedding_precision,
)
from .core.generative_model import LinearGaussianModel
from .core.generative_process import LinearGaussianProcess
from .core.inference import GridBayesianInference
from .core.thermodynamics import vfe_thermodynamic_state
from .core.variational import GaussianBelief, variational_free_energy
from .estimators.active_inference import (
    MultivariateActiveEnvironment,
    simulate_multivariate_active_inference,
)
from .estimators.em import fit_factor_analysis
from .estimators.generalized_filtering import (
    MultivariateDynamicProcess,
    generalized_measurements_from_series,
    generalized_vector_filter,
    multivariate_generalized_filter,
    simulate_multivariate_process,
)
from .estimators.gradient_descent import gradient_descent
from .estimators.linear_regression import BayesianLinearRegression
from .estimators.map import map_analytic_linear, map_grad_x, map_loss
from .estimators.mle import mle_analytic_linear, mle_grad_x, mle_loss
from .extra_topics import extra_topic_spec
from .utils.grids import make_grid
from .visualizations import (
    COLORS,
    animate_multivariate_active_inference,
    confidence_ellipse,
    plot_generalized_vector_filter,
    plot_gradient_descent,
    plot_likelihood_ridge,
    plot_multivariate_active_inference,
    plot_prior_likelihood_posterior,
)


@dataclass
class WorkflowResult:
    """Figures, exportable arrays, metadata, and scalar diagnostics."""

    figures: dict[str, plt.Figure] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class AnimationWorkflowResult:
    """Animation plus exportable arrays, metadata, and scalar diagnostics."""

    animation: FuncAnimation
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


MULTIVARIATE_AIF_PREFERENCE = np.array([0.0, 0.0])
MULTIVARIATE_AIF_EXOGENOUS = np.array([6.0, -4.0])
_HOOKE_K, _HOOKE_MASS, _HOOKE_V0, _HOOKE_THETA_Y = 4.0, 3.0, 5.0, 3.0


def sequential_inference_posteriors(
    model: LinearGaussianModel,
    x_grid: np.ndarray,
    samples: np.ndarray,
):
    """Yield ``(i, posterior)`` after assimilating samples one by one."""
    log_state = model.log_prior(x_grid).copy()
    for i, y in enumerate(samples):
        log_state = log_state + model.log_likelihood(float(y), x_grid)
        normed = np.exp(log_state - np.max(log_state))
        normed /= np.trapezoid(normed, x_grid)
        yield i, normed


def build_example_2_7_multiple_samples(
    *,
    seed: int,
    n_samples: int,
    x_true: float,
) -> WorkflowResult:
    """Build figures for Example 2.7's batch-vs-sequential inference demo."""
    rng = np.random.default_rng(seed)
    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25, rng=rng)
    samples = process.sample(x_true, n=n_samples).flatten()
    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0,
        beta1=2.0,
        sigma2_y=0.25,
        m_x=4.0,
        s2_x=0.25,
        prior_kind="gaussian",
    )
    res_batch = GridBayesianInference(model, x_grid).infer(samples)

    modes: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for _, post in sequential_inference_posteriors(model, x_grid, samples):
        modes.append(float(x_grid[int(np.argmax(post))]))
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(x_grid))))
        cdf /= cdf[-1]
        los.append(float(np.interp(0.025, cdf, x_grid)))
        his.append(float(np.interp(0.975, cdf, x_grid)))
    if not np.allclose(modes[-1], res_batch.posterior_mode, atol=2e-3):
        raise RuntimeError("sequential and batch posterior modes diverged")

    per_sample_lik = [np.exp(model.log_likelihood(float(y), x_grid)) for y in samples[:9]]
    fig_ridge = plot_likelihood_ridge(
        x_grid,
        per_sample_lik,
        labels=[f"y_{i + 1} = {y:.2f}" for i, y in enumerate(samples[:9])],
        title="Example 2.7 · per-sample likelihoods (first 9 of N)",
    )
    fig_post = plot_prior_likelihood_posterior(
        res_batch,
        title=f"Example 2.7 · posterior after N = {n_samples}",
        truth=x_true,
    )
    fig_conv, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    n_axis = np.arange(1, n_samples + 1)
    ax.plot(n_axis, modes, color="#2ca02c", lw=2, label="posterior mode")
    ax.fill_between(n_axis, los, his, alpha=0.25, color="#2ca02c", label="95% credible interval")
    ax.axhline(x_true, color="red", ls=":", lw=1.5, label="true x*")
    ax.set_xlabel("samples assimilated")
    ax.set_ylabel("estimate")
    ax.set_title("Sequential Bayesian update converges to the truth")
    ax.legend()
    ax.grid(alpha=0.3)

    return WorkflowResult(
        figures={
            "example_2_7_ridge": fig_ridge,
            "example_2_7_posterior": fig_post,
            "example_2_7_convergence": fig_conv,
        },
        summary={
            "sample_mean": float(samples.mean()),
            "sample_std": float(samples.std()),
            "process_mean": float(process.mean(x_true)),
            "batch_mode": float(res_batch.posterior_mode),
            "batch_variance": float(res_batch.posterior_variance),
            "sequential_mode": float(modes[-1]),
        },
    )


def build_example_2_10_gradient_descent(
    *,
    seed: int,
    n_samples: int,
    x_true: float,
    lr: float,
    max_iter: int,
) -> WorkflowResult:
    """Build figures for Example 2.10's MLE/MAP gradient-descent demo."""
    rng = np.random.default_rng(seed)
    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng)
    samples = process.sample(x_true, n=n_samples).flatten()
    mle_iter = gradient_descent(
        loss_fn=lambda x: float(mle_loss(x, samples, 3.0, 2.0, 0.5)),
        grad_fn=lambda x: mle_grad_x(x, samples, 3.0, 2.0, 0.5),
        x0=5.0,
        learning_rate=lr,
        max_iter=max_iter,
    )
    mle_closed = mle_analytic_linear(samples, 3.0, 2.0)
    if abs(mle_iter.x - mle_closed) >= 5e-3:
        raise RuntimeError("MLE gradient descent failed to match the analytic solution")

    map_iter = gradient_descent(
        loss_fn=lambda x: float(map_loss(x, samples, 3.0, 2.0, 0.5, 4.0, 0.25)),
        grad_fn=lambda x: map_grad_x(x, samples, 3.0, 2.0, 0.5, 4.0, 0.25),
        x0=4.0,
        learning_rate=lr,
        max_iter=max_iter,
    )
    map_closed = map_analytic_linear(samples, 3.0, 2.0, 0.5, 4.0, 0.25)
    if abs(map_iter.x - map_closed) >= 5e-3:
        raise RuntimeError("MAP gradient descent failed to match the analytic solution")

    fig_mle = plot_gradient_descent(
        mle_iter.history,
        mle_iter.losses,
        truth=x_true,
        title=f"MLE gradient descent (converged after {mle_iter.n_iterations})",
    )
    fig_map = plot_gradient_descent(
        map_iter.history,
        map_iter.losses,
        truth=x_true,
        title=f"MAP gradient descent (converged after {map_iter.n_iterations})",
    )
    fig_cmp, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(mle_iter.history, color="#1f77b4", lw=2, label="MLE iterate")
    ax.plot(map_iter.history, color="#d62728", lw=2, label="MAP iterate")
    ax.axhline(x_true, color="black", ls=":", label="x*")
    ax.axhline(mle_closed, color="#1f77b4", ls="--", alpha=0.5, label=f"MLE closed = {mle_closed:.3f}")
    ax.axhline(map_closed, color="#d62728", ls="--", alpha=0.5, label=f"MAP closed = {map_closed:.3f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("hidden-state estimate")
    ax.set_title("MLE vs MAP descent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    return WorkflowResult(
        figures={
            "example_2_10_mle_descent": fig_mle,
            "example_2_10_map_descent": fig_map,
            "example_2_10_comparison": fig_cmp,
        },
        summary={
            "sample_mean": float(samples.mean()),
            "sample_std": float(samples.std()),
            "mle_closed": float(mle_closed),
            "mle_iter": float(mle_iter.x),
            "map_closed": float(map_closed),
            "map_iter": float(map_iter.x),
        },
    )


def build_example_3_5_bayesian_linear_regression(*, seed: int, max_n: int) -> WorkflowResult:
    """Build figures for Example 3.5's Bayesian linear-regression demo."""
    rng = np.random.default_rng(seed)
    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)
    x_pool = rng.uniform(0, 5, size=max_n)
    y_pool = np.array([float(process.sample(float(x), n=1)[0]) for x in x_pool])
    blr = BayesianLinearRegression(prior_mean=np.zeros(2), prior_cov=np.eye(2) * 4.0, sigma2_y=1.0)

    snapshots = [0] + [n for n in (1, 5, 15) if n < max_n] + [max_n]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 4), constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    truth = np.array([3.0, 2.0])
    final_post = None
    for ax, n in zip(axes, snapshots):
        if n == 0:
            mean, cov, label = blr.prior_mean, blr.prior_cov, "prior"
        else:
            post = blr.fit(x_pool[:n], y_pool[:n])
            mean, cov, label = post.mean, post.cov, f"posterior (N = {n})"
            if n == max_n:
                final_post = post
        diff = pts - mean
        prec = np.linalg.inv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, prec, diff).reshape(xx.shape)
        ax.contourf(xx, yy, np.exp(-0.5 * quad), levels=10, cmap="Greens")
        for n_std, alpha in zip((1, 2), (0.45, 0.18)):
            ax.add_patch(confidence_ellipse(mean, cov, n_std=n_std, fc="none", ec="black", alpha=alpha + 0.4, lw=1.5))
        ax.scatter(*truth, marker="x", color="red", s=80, lw=2)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", xlabel=r"$\hat{\beta}_0$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat{\beta}_1$")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
    if final_post is None:
        raise RuntimeError("final posterior was not computed")

    fig_pred, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    grid_x = np.linspace(0, 5, 200).reshape(-1, 1)
    mean_pred, var_pred = final_post.predictive(grid_x, sigma2_y=blr.sigma2_y)
    std_pred = np.sqrt(var_pred)
    ax.fill_between(grid_x.flatten(), mean_pred - 2 * std_pred, mean_pred + 2 * std_pred, alpha=0.25, color="#2ca02c", label="95% predictive")
    ax.plot(grid_x, mean_pred, color="#2ca02c", lw=2, label="predictive mean")
    ax.plot(grid_x, 3.0 + 2.0 * grid_x.flatten(), color="red", ls=":", label="true line")
    ax.scatter(x_pool, y_pool, s=14, color="black", alpha=0.6, label="data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior predictive distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    return WorkflowResult(
        figures={
            "example_3_5_blr_posteriors": fig,
            "example_3_5_blr_predictive": fig_pred,
        },
        summary={
            "posterior_mean": final_post.mean,
            "posterior_std": final_post.std(),
        },
    )


def build_example_3_7_factor_analysis_em(
    *,
    seed: int,
    n_samples: int,
    n_factors: int,
    max_iter: int,
) -> WorkflowResult:
    """Build the factor-analysis EM figure for Example 3.7."""
    rng = np.random.default_rng(seed)
    true_theta = np.array([[1.0, 0.5], [0.7, -0.3], [-0.2, 1.0], [0.4, 0.4], [0.0, 0.9]])
    true_diag = np.array([0.10, 0.20, 0.05, 0.30, 0.15])
    n_obs = true_theta.shape[0]
    x_latent = mvn_sample(np.zeros(n_factors), np.eye(n_factors), n=n_samples, rng=rng)
    noise = mvn_sample(np.zeros(n_obs), diagonal_cov(true_diag), n=n_samples, rng=rng)
    y = x_latent @ true_theta.T + noise
    result = fit_factor_analysis(
        y,
        n_factors=n_factors,
        max_iter=max_iter,
        tol=1e-6,
        rng=np.random.default_rng(seed + 1),
    )
    y_hat = result.posterior_means @ result.Theta.T
    rmse = float(np.sqrt(((y - y_hat) ** 2).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(result.log_likelihoods, color="#1f77b4", lw=2)
    axes[0, 0].set_xlabel("EM iteration")
    axes[0, 0].set_ylabel("incomplete log p(Y)")
    axes[0, 0].set_title("Marginal log-likelihood (monotone increase)")
    axes[0, 0].grid(alpha=0.3)
    width = 0.35
    idx = np.arange(true_diag.size)
    axes[0, 1].bar(idx - width / 2, true_diag, width=width, color="#1f77b4", label="true")
    axes[0, 1].bar(idx + width / 2, np.diag(result.cov_y), width=width, color="#2ca02c", label="EM estimate")
    axes[0, 1].set_xticks(idx)
    axes[0, 1].set_xticklabels([f"y_{i}" for i in idx])
    axes[0, 1].set_ylabel("noise variance")
    axes[0, 1].set_title("Per-channel noise (recovered)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis="y")
    vmax = max(np.abs(true_theta).max(), np.abs(result.Theta).max())
    im0 = axes[1, 0].imshow(true_theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 0].set_title(r"True loadings $\Theta_*$")
    fig.colorbar(im0, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_xlabel("factor")
    axes[1, 0].set_ylabel("output dim")
    im1 = axes[1, 1].imshow(result.Theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 1].set_title(rf"EM estimate $\hat{{\Theta}}$ (rmse = {rmse:.3f})")
    fig.colorbar(im1, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_xlabel("factor")
    axes[1, 1].set_ylabel("output dim")

    return WorkflowResult(
        figures={"example_3_7_factor_analysis_em": fig},
        summary={
            "converged": bool(result.converged),
            "n_iterations": int(result.n_iterations),
            "final_log_likelihood": float(result.log_likelihoods[-1]),
            "rmse": rmse,
        },
    )


def hooke_vector_functions() -> tuple[LinearVectorFunction, LinearVectorFunction]:
    """Return Hooke's-law flow and shifted-identity observation functions."""
    a_f = np.array([[0.0, 1.0], [-_HOOKE_K / _HOOKE_MASS, 0.0]])
    b_f = np.array([0.0, (_HOOKE_K / _HOOKE_MASS) * _HOOKE_V0])
    return LinearVectorFunction(a_f, b_f), LinearVectorFunction(np.eye(2), np.array([-_HOOKE_THETA_Y, -_HOOKE_THETA_Y]))


def build_example_6_7_multivariate_generalized_coordinates(
    *,
    seed: int,
    n_steps: int,
    dt: float,
    embedding_dim: int,
    gamma: float,
) -> WorkflowResult:
    """Build Example 6.7's generalized-coordinate filtering figure/data."""
    f, g = hooke_vector_functions()
    process = MultivariateDynamicProcess(f=f, g=g, omega_x=0.0, omega_y=0.0)
    trace = simulate_multivariate_process(process, x0=np.array([0.0, 5.0]), n_steps=n_steps, dt=dt, rng=np.random.default_rng(seed))
    base_model = MultivariateDynamicModel(f=f, g=g, precision_x=0.5, precision_y=10.0, dim_x=2, dim_y=2)
    y_tilde = generalized_measurements_from_series(trace.ys, embedding_dim=embedding_dim, dt=dt)
    model = GeneralizedVectorModel(
        f=f,
        g=g,
        precision_x=correlated_embedding_precision(np.eye(2) * 0.5, embedding_dim, gamma=gamma),
        precision_y=correlated_embedding_precision(np.eye(2) * 10.0, embedding_dim, gamma=gamma),
        embedding_dim=embedding_dim,
        dim_x=2,
        dim_y=2,
    )
    ordinary = multivariate_generalized_filter(base_model, trace.ys, dt=dt, kappa=1.0, mu0=np.array([8.0, 8.0]))
    generalized = generalized_vector_filter(model, y_tilde, dt=dt, kappa=1.0, mu0_tilde=np.zeros((embedding_dim, 2)))
    burn = n_steps // 3
    fig = plot_generalized_vector_filter(generalized, trace.xs, ordinary=ordinary, dt=dt)
    arrays = {
        "time": np.arange(trace.xs.shape[0]) * dt,
        "x_true": trace.xs,
        "observations": trace.ys,
        "generalized_measurements": y_tilde,
        "ordinary_beliefs": ordinary.mus,
        "generalized_beliefs": generalized.mus,
        "generalized_eps_x": generalized.eps_x,
        "generalized_eps_y": generalized.eps_y,
        "ordinary_free_energy": ordinary.free_energies,
        "generalized_free_energy": generalized.free_energies,
        "precision_x": model.Pi_x,
        "precision_y": model.Pi_y,
    }
    ordinary_error = ordinary.tracking_error(trace.xs, burn_in=burn)
    generalized_error = generalized.tracking_error(trace.xs, burn_in=burn)
    return WorkflowResult(
        figures={"example_6_7_multivariate_generalized_coordinates": fig},
        arrays=arrays,
        metadata={"seed": seed, "dt": dt, "gamma": gamma, "embedding_dim": embedding_dim},
        summary={"ordinary_error": ordinary_error, "generalized_error": generalized_error},
    )


def build_multivariate_active_agent_env(
    kappa_a: float,
    *,
    embedding_dim: int,
    gamma: float,
) -> tuple[MultivariateActiveInferenceAgent, MultivariateActiveEnvironment]:
    """Create the vector agent/environment used by Chapter 7 section 7.5."""
    model = GeneralizedVectorModel(
        f=LinearVectorFunction(-np.eye(2), MULTIVARIATE_AIF_PREFERENCE),
        g=LinearVectorFunction(np.eye(2), np.zeros(2)),
        precision_x=correlated_embedding_precision(np.eye(2), embedding_dim, gamma=gamma),
        precision_y=correlated_embedding_precision(np.eye(2) * 20.0, embedding_dim, gamma=gamma),
        embedding_dim=embedding_dim,
        dim_x=2,
        dim_y=2,
    )
    forward = np.zeros((embedding_dim * 2, 2))
    forward[[0, embedding_dim], :] = np.eye(2)
    agent = MultivariateActiveInferenceAgent(perception_model=model, forward_model=forward, kappa_x=0.4, kappa_a=kappa_a)
    env = MultivariateActiveEnvironment(
        drift=LinearVectorFunction(-np.eye(2), MULTIVARIATE_AIF_EXOGENOUS),
        g=LinearVectorFunction(np.eye(2), np.zeros(2)),
        action_matrix=np.eye(2),
        omega_x=0.0,
        omega_y=0.0,
    )
    return agent, env


def build_example_7_5_multivariate_active_inference(
    *,
    seed: int,
    n_steps: int,
    dt: float,
    embedding_dim: int,
    gamma: float,
) -> WorkflowResult:
    """Build Example 7.5's multivariate active-inference figure/data."""
    action_start = n_steps // 5
    agent, env = build_multivariate_active_agent_env(0.5, embedding_dim=embedding_dim, gamma=gamma)
    passive_agent, _ = build_multivariate_active_agent_env(0.0, embedding_dim=embedding_dim, gamma=gamma)
    active = simulate_multivariate_active_inference(
        agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((embedding_dim, 2)),
        n_steps=n_steps,
        dt=dt,
        action_start=action_start,
        rng=np.random.default_rng(seed),
    )
    passive = simulate_multivariate_active_inference(
        passive_agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((embedding_dim, 2)),
        n_steps=n_steps,
        dt=dt,
        action_start=action_start,
        rng=np.random.default_rng(seed),
    )
    fig = plot_multivariate_active_inference(
        active,
        preference=MULTIVARIATE_AIF_PREFERENCE,
        exogenous=MULTIVARIATE_AIF_EXOGENOUS,
        baseline=passive,
        dt=dt,
    )
    arrays = {
        "time": np.arange(active.xs.shape[0]) * dt,
        "active_xs": active.xs,
        "active_beliefs": active.mus,
        "active_actions": active.actions,
        "active_observations": active.ys,
        "active_generalized_measurements": active.y_tildes,
        "active_eps_y": active.eps_y,
        "active_free_energy": active.free_energies,
        "passive_xs": passive.xs,
        "passive_beliefs": passive.mus,
        "passive_free_energy": passive.free_energies,
        "preference": MULTIVARIATE_AIF_PREFERENCE,
        "exogenous": MULTIVARIATE_AIF_EXOGENOUS,
    }
    return WorkflowResult(
        figures={"example_7_5_multivariate_active_inference": fig},
        arrays=arrays,
        metadata={"seed": seed, "dt": dt, "gamma": gamma, "action_start": action_start},
        summary={
            "active_preference_error": active.preference_error(MULTIVARIATE_AIF_PREFERENCE),
            "passive_preference_error": passive.preference_error(MULTIVARIATE_AIF_PREFERENCE),
        },
    )


def build_animation_7_5_multivariate_active_inference(
    *,
    seed: int,
    n_steps: int,
    dt: float,
    embedding_dim: int,
    gamma: float,
) -> AnimationWorkflowResult:
    """Build the Chapter 7 section 7.5 active-inference animation/data."""
    action_start = n_steps // 5
    agent, env = build_multivariate_active_agent_env(0.5, embedding_dim=embedding_dim, gamma=gamma)
    result = simulate_multivariate_active_inference(
        agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((embedding_dim, 2)),
        n_steps=n_steps,
        dt=dt,
        action_start=action_start,
        rng=np.random.default_rng(seed),
    )
    anim = animate_multivariate_active_inference(
        result,
        preference=MULTIVARIATE_AIF_PREFERENCE,
        exogenous=MULTIVARIATE_AIF_EXOGENOUS,
        dt=dt,
        frame_stride=25,
        title="Fig. 7.5 animated · vector action-perception loop",
    )
    arrays = {
        "time": np.arange(result.xs.shape[0]) * dt,
        "xs": result.xs,
        "beliefs": result.mus,
        "actions": result.actions,
        "generalized_measurements": result.y_tildes,
        "eps_y": result.eps_y,
        "free_energy": result.free_energies,
        "preference": MULTIVARIATE_AIF_PREFERENCE,
        "exogenous": MULTIVARIATE_AIF_EXOGENOUS,
    }
    return AnimationWorkflowResult(
        animation=anim,
        arrays=arrays,
        metadata={"seed": seed, "dt": dt, "gamma": gamma, "action_start": action_start},
        summary={"preference_error": result.preference_error(MULTIVARIATE_AIF_PREFERENCE)},
    )


def build_variational_free_energy_visualization(*, y: float) -> WorkflowResult:
    """Build the extras variational-free-energy topic figure/data."""
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model, x_grid).infer(y)
    mus = np.linspace(4.5, exact.posterior_mean, 36)
    variances = np.linspace(1.5, exact.posterior_variance, 36)
    components = [variational_free_energy(GaussianBelief(float(mu), float(var)), model, y, x_grid) for mu, var in zip(mus, variances)]
    states = [vfe_thermodynamic_state(GaussianBelief(float(mu), float(var)), model, y, x_grid) for mu, var in zip(mus, variances)]
    free_energy = np.array([c.free_energy for c in components])
    surprisal = np.array([c.surprisal for c in components])
    bound_gap = np.array([c.bound_gap for c in components])
    complexity = np.array([c.complexity for c in components])
    accuracy = np.array([c.accuracy for c in components])
    internal_energy = np.array([s.energy for s in states])
    entropy = np.array([s.entropy for s in states])
    iteration = np.arange(free_energy.size)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(iteration, free_energy, lw=2, label="F", color=COLORS["posterior"])
    axes[0].plot(iteration, surprisal, lw=2, ls="--", label="-log p(y)", color=COLORS["likelihood"])
    axes[0].fill_between(iteration, surprisal, free_energy, alpha=0.2, color=COLORS["posterior"], label="bound gap")
    axes[0].set_xlabel("belief interpolation step")
    axes[0].set_ylabel("nats")
    axes[0].set_title("VFE upper-bounds surprisal")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(iteration, complexity, lw=2, label="complexity", color=COLORS["prior"])
    axes[1].plot(iteration, -accuracy, lw=2, label="-accuracy", color=COLORS["likelihood"])
    axes[1].plot(iteration, internal_energy - entropy, lw=2, ls=":", label="U - S", color=COLORS["posterior"])
    axes[1].set_xlabel("belief interpolation step")
    axes[1].set_ylabel("nats")
    axes[1].set_title("Equivalent decompositions")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Variational free energy as a thermodynamic bridge")

    return WorkflowResult(
        figures={"visualize_variational_free_energy": fig},
        arrays={
            "iteration": iteration,
            "mu": mus,
            "variance": variances,
            "free_energy": free_energy,
            "surprisal": surprisal,
            "bound_gap": bound_gap,
            "complexity": complexity,
            "accuracy": accuracy,
            "internal_energy": internal_energy,
            "entropy": entropy,
        },
        metadata={
            "script": "visualize_variational_free_energy.py",
            "y": y,
            "source_apis": list(extra_topic_spec("variational_free_energy").source_apis),
        },
    )


__all__ = [
    "AnimationWorkflowResult",
    "MULTIVARIATE_AIF_EXOGENOUS",
    "MULTIVARIATE_AIF_PREFERENCE",
    "WorkflowResult",
    "build_animation_7_5_multivariate_active_inference",
    "build_example_2_7_multiple_samples",
    "build_example_2_10_gradient_descent",
    "build_example_3_5_bayesian_linear_regression",
    "build_example_3_7_factor_analysis_em",
    "build_example_6_7_multivariate_generalized_coordinates",
    "build_example_7_5_multivariate_active_inference",
    "build_multivariate_active_agent_env",
    "build_variational_free_energy_visualization",
    "hooke_vector_functions",
    "sequential_inference_posteriors",
]
