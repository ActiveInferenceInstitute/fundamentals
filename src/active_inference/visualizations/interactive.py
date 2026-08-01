"""Interactive widget-based simulations using matplotlib's slider widget.

These functions are zero-dependency beyond ``matplotlib`` itself — they avoid
the heavier ``ipywidgets`` stack so chapter scripts run in any environment
that has a display.
"""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ..core.generative_model import LinearGaussianModel
from ..core.inference import GridBayesianInference
from ..core.predictive_coding import (
    LinearFunction,
    PredictiveCodingModel,
    pc_linear_fixed_point,
    predictive_coding_free_energy,
    sensory_prediction_error,
    state_prediction_error,
)
from ..core.variational import GaussianBelief, variational_free_energy
from ..utils.grids import make_grid
from .style import COLORS, annotate_stat_box, stat_box_bbox


def interactive_inference(
    *,
    x_low: float = 0.0,
    x_high: float = 5.0,
    n_grid: int = 500,
    beta0: float = 3.0,
    beta1: float = 2.0,
    sigma2_y_init: float = 0.25,
    s2_x_init: float = 0.25,
    m_x_init: float = 4.0,
    y_init: float = 7.0,
) -> plt.Figure:
    """Live sliders for the four levers of the linear-Gaussian agent.

    Drag the observation, prior mean, prior variance, and likelihood variance
    sliders to watch prior / likelihood / posterior update in real time.
    """
    x_grid = make_grid(x_low, x_high, n_grid)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.32, wspace=0.35)

    def compute(y, m_x, s2_x, sigma2_y):
        """Compute the current inference result from interactive control values."""
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            m_x=m_x, s2_x=s2_x, prior_kind="gaussian",
        )
        return GridBayesianInference(model, x_grid).infer(y)

    res = compute(y_init, m_x_init, s2_x_init, sigma2_y_init)

    line_prior, = axes[0].plot(x_grid, res.prior, color=COLORS["prior"], lw=2)
    line_lik, = axes[1].plot(x_grid, res.likelihood, color=COLORS["likelihood"], lw=2)
    line_post, = axes[2].plot(x_grid, res.posterior, color=COLORS["posterior"], lw=2)
    mode_marker = axes[2].axvline(res.posterior_mode, color=COLORS["data"], ls="--", lw=1)
    axes[0].set_title("Prior p(x)")
    axes[1].set_title("Likelihood p(y | x)")
    axes[2].set_title("Posterior p(x | y)")
    for ax in axes:
        ax.set_xlim(x_low, x_high)
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    stat_text = axes[2].text(
        0.02, 0.97, "", transform=axes[2].transAxes, fontsize=8,
        va="top", ha="left", bbox=stat_box_bbox(pad=0.25),
    )

    ax_y = plt.axes([0.10, 0.18, 0.80, 0.03])
    ax_mx = plt.axes([0.10, 0.13, 0.80, 0.03])
    ax_sx = plt.axes([0.10, 0.08, 0.80, 0.03])
    ax_sy = plt.axes([0.10, 0.03, 0.80, 0.03])

    s_y = Slider(ax_y, "observation y",
                 beta0 + beta1 * x_low, beta0 + beta1 * x_high,
                 valinit=y_init)
    s_mx = Slider(ax_mx, "prior mean m_x", x_low, x_high, valinit=m_x_init)
    s_sx = Slider(ax_sx, "prior var s2_x", 1e-3, 4.0, valinit=s2_x_init)
    s_sy = Slider(ax_sy, "lik var sigma2_y", 1e-3, 4.0, valinit=sigma2_y_init)

    def update(_event=None):
        """Update interactive or animated artists for the current state."""
        r = compute(s_y.val, s_mx.val, s_sx.val, s_sy.val)
        line_prior.set_ydata(r.prior)
        line_lik.set_ydata(r.likelihood)
        line_post.set_ydata(r.posterior)
        mode_marker.set_xdata([r.posterior_mode, r.posterior_mode])
        for ax, ydata in zip(axes, (r.prior, r.likelihood, r.posterior)):
            top = float(np.max(ydata))
            ax.set_ylim(0, max(top * 1.1, 1e-6))
        # Live diagnostic readout — moves with the sliders so users can watch
        # the statistics change, not just the curve shape.
        stat_text.set_text(
            f"mode = {r.posterior_mode:.3f}\n"
            f"mean = {r.posterior_mean:.3f}\n"
            f"std  = {r.posterior_variance ** 0.5:.3f}\n"
            f"H    = {r.entropy():.3f}\n"
            f"KL   = {r.kl_from_prior():.3f}"
        )
        fig.canvas.draw_idle()

    update()  # populate the stat readout immediately

    for s in (s_y, s_mx, s_sx, s_sy):
        s.on_changed(update)

    fig._sliders = (s_y, s_mx, s_sx, s_sy)  # keep references alive
    return fig


def interactive_precision(
    *,
    x_low: float = 0.0,
    x_high: float = 5.0,
    beta0: float = 3.0,
    beta1: float = 2.0,
    y_obs: float = 7.0,
    m_x: float = 4.0,
) -> plt.Figure:
    """Single-slider sweep showing the prior-vs-data trade-off.

    A single ``log10(precision_ratio)`` slider sweeps from "trust the prior" to
    "trust the data". This is a stripped-down version of
    :func:`interactive_inference` that highlights only the precision lever.
    """
    x_grid = make_grid(x_low, x_high, 500)

    def compute(log_ratio: float):
        """Compute the current inference result from interactive control values."""
        ratio = 10 ** log_ratio
        # Split the requested log10(s2_x/sigma2_y) ratio symmetrically so the
        # label matches the actual precision ratio: s2_x = √ratio, σ²_y = 1/√ratio
        # ⇒ s2_x/σ²_y = ratio = 10^(log_ratio).
        s2_x = np.sqrt(ratio)
        sigma2_y = 1.0 / np.sqrt(ratio)
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            m_x=m_x, s2_x=s2_x, prior_kind="gaussian",
        )
        return GridBayesianInference(model, x_grid).infer(y_obs)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=False)
    plt.subplots_adjust(bottom=0.22, top=0.9)

    res = compute(0.0)
    line_prior, = ax.plot(x_grid, res.prior / np.max(res.prior),
                          lw=2, label="prior", color=COLORS["prior"])
    line_lik, = ax.plot(x_grid, res.likelihood / np.max(res.likelihood),
                        lw=2, label="likelihood", color=COLORS["likelihood"])
    line_post, = ax.plot(x_grid, res.posterior / np.max(res.posterior),
                         lw=2, label="posterior", color=COLORS["posterior"])
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density (peak = 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Posterior is a precision-weighted compromise (drag slider)")

    ax_slider = plt.axes([0.15, 0.07, 0.7, 0.04])
    s = Slider(ax_slider, "log10(s2_x / sigma2_y)", -2.0, 2.0, valinit=0.0)

    stat_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
        va="top", ha="left", bbox=stat_box_bbox(pad=0.25),
    )

    def update(_event=None):
        """Update interactive or animated artists for the current state."""
        r = compute(s.val)
        line_prior.set_ydata(r.prior / np.max(r.prior))
        line_lik.set_ydata(r.likelihood / np.max(r.likelihood))
        line_post.set_ydata(r.posterior / np.max(r.posterior))
        stat_text.set_text(
            f"log ratio = {s.val:+.2f}\n"
            f"posterior mode = {r.posterior_mode:.3f}\n"
            f"posterior std  = {r.posterior_variance ** 0.5:.3f}\n"
            f"KL[post||prior] = {r.kl_from_prior():.3f}"
        )
        fig.canvas.draw_idle()

    update()
    s.on_changed(update)
    fig._slider = s
    return fig


def interactive_topic_demo(slug: str) -> plt.Figure:
    """Interactive one-slider explorer for a simulation-capable extras topic.

    The widget reuses :func:`active_inference.extra_topics.build_topic_demo`
    in ``simulate`` mode and exposes a continuous blend between the baseline
    and the main simulation curve. This keeps every extras interactive tied to
    the same deterministic, validated data builder as the static and simulation
    wrappers.
    """
    from active_inference.extra_topics import build_topic_demo, extra_topic_spec

    spec = extra_topic_spec(slug)
    if not spec.has_simulation:
        raise ValueError(f"Extras topic {slug!r} does not declare a simulation mode")
    demo = build_topic_demo(slug, mode="simulate")
    x = np.asarray(demo.arrays["x"], dtype=float)
    primary = np.asarray(demo.arrays[demo.line_keys[0]], dtype=float)
    if primary.shape != x.shape:
        x = np.linspace(0.0, 1.0, primary.size)
    if len(demo.line_keys) > 1:
        baseline = np.asarray(demo.arrays[demo.line_keys[1]], dtype=float)
        if baseline.shape != primary.shape:
            baseline = np.full_like(primary, float(np.nanmean(primary)))
        baseline_label = str(demo.metadata.get(f"{demo.line_keys[1]}_label", "baseline"))
    else:
        baseline = np.zeros_like(primary) + float(np.nanmean(primary))
        baseline_label = "baseline"

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=False)
    plt.subplots_adjust(bottom=0.22, top=0.88)
    ax.set_title(f"{spec.title}: interactive parameter sweep", loc="left")
    ax.set_xlabel(str(demo.metadata.get("x_label", "parameter")))
    ax.set_ylabel(str(demo.metadata.get(f"{demo.line_keys[0]}_label", "value")))
    ax.grid(alpha=0.3)
    baseline_line, = ax.plot(x, baseline, color=COLORS["neutral"], lw=2.0, ls="--", label=baseline_label)
    target_line, = ax.plot(x, primary, color=COLORS["prior"], lw=2.0, alpha=0.55, label="target")
    current_line, = ax.plot(x, baseline, color=COLORS["posterior"], lw=3.0, label="interactive blend")
    ax.legend(fontsize=10)
    values = np.concatenate([baseline, primary])
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    pad = 0.08 * max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)

    stat_text = annotate_stat_box(ax, "", loc="upper left", fontsize=9, monospace=False)
    ax_slider = plt.axes([0.16, 0.08, 0.68, 0.04])
    slider = Slider(ax_slider, "simulation blend", 0.0, 1.0, valinit=0.0)

    def update(_event=None) -> None:
        """Update the extras topic curve from the slider value."""
        blend = float(slider.val)
        current = (1.0 - blend) * baseline + blend * primary
        current_line.set_ydata(current)
        stat_text.set_text(
            f"{spec.slug.replace('_', ' ')}\n"
            f"blend = {blend:.2f}\n"
            f"mean = {np.mean(current):.3f}\n"
            f"source = {spec.source_apis[0].split('.')[-1]}"
        )
        fig.canvas.draw_idle()

    update()
    slider.on_changed(update)
    fig._slider = slider  # type: ignore[attr-defined]
    fig._topic_demo = demo  # type: ignore[attr-defined]
    fig._interactive_lines = (baseline_line, target_line, current_line)  # type: ignore[attr-defined]
    return fig


def interactive_inverse_problem(
    *,
    beta0: float = 3.0,
    beta1: float = 2.0,
    x_extent: float = 2.5,
    n_grid: int = 601,
    y_init: float = 5.0,
    sigma2_y_init: float = 0.05,
) -> plt.Figure:
    """Chapter 1 · the inverse problem, live.

    A non-injective quadratic generator ``y = β₀ + β₁ x²`` maps ``+x`` and ``−x`` to
    the same observation, so with a symmetric uniform prior the posterior over the
    hidden state is **bi-modal**. Drag the **observation ``y``** slider to watch the
    two modes separate as ``y`` grows, and the **likelihood variance ``σ_y²``**
    slider to watch them sharpen or merge — the interactive companion to
    ``04_inverse_problem.py``. The readout reports both modes and their separation.
    """
    x_grid = make_grid(-x_extent, x_extent, n_grid)

    def compute(y: float, sigma2_y: float):
        """Grid posterior for the quadratic generator at the given controls."""
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            prior_kind="uniform", uniform_low=-x_extent, uniform_high=x_extent,
            psi=lambda x: x ** 2,
        )
        return GridBayesianInference(model, x_grid).infer(y)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.28)
    res = compute(y_init, sigma2_y_init)
    (post_line,) = ax.plot(x_grid, res.posterior, color=COLORS["posterior"], lw=2.2)
    fill = ax.fill_between(x_grid, res.posterior, alpha=0.2, color=COLORS["posterior"])
    left_marker = ax.axvline(0.0, color=COLORS["data"], ls="--", lw=1.2)
    right_marker = ax.axvline(0.0, color=COLORS["data"], ls="--", lw=1.2)
    ax.set_xlim(-x_extent, x_extent)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("posterior density")
    ax.set_title("Inverse problem: two states explain one observation")
    ax.grid(alpha=0.3)
    stat_text = annotate_stat_box(ax, "", loc="upper right", fontsize=9, monospace=False)

    ax_y = plt.axes([0.12, 0.14, 0.76, 0.04])
    ax_sy = plt.axes([0.12, 0.07, 0.76, 0.04])
    s_y = Slider(ax_y, "observation y", beta0, beta0 + beta1 * x_extent ** 2, valinit=y_init)
    s_sy = Slider(ax_sy, "lik var σ_y²", 1e-3, 1.0, valinit=sigma2_y_init)

    def update(_event=None) -> None:
        """Refresh the bimodal posterior and mode readout from the sliders."""
        nonlocal fill
        r = compute(s_y.val, s_sy.val)
        post = r.posterior
        post_line.set_ydata(post)
        for coll in [c for c in ax.collections if c is fill]:
            coll.remove()
        fill = ax.fill_between(x_grid, post, alpha=0.2, color=COLORS["posterior"])
        ax.set_ylim(0, max(float(np.max(post)) * 1.15, 1e-6))
        left = x_grid < 0
        right = x_grid >= 0
        mode_l = float(x_grid[left][np.argmax(post[left])])
        mode_r = float(x_grid[right][np.argmax(post[right])])
        left_marker.set_xdata([mode_l, mode_l])
        right_marker.set_xdata([mode_r, mode_r])
        stat_text.set_text(
            f"left mode  = {mode_l:.3f}\n"
            f"right mode = {mode_r:.3f}\n"
            f"separation = {mode_r - mode_l:.3f}\n"
            f"entropy H  = {r.entropy():.3f}"
        )
        fig.canvas.draw_idle()

    update()
    for s in (s_y, s_sy):
        s.on_changed(update)
    fig._sliders = (s_y, s_sy)  # type: ignore[attr-defined]
    return fig


def interactive_bayesian_regression(
    *,
    beta0_true: float = 3.0,
    beta1_true: float = 2.0,
    x_low: float = 0.0,
    x_high: float = 5.0,
    n_max: int = 60,
    noise_std: float = 1.0,
    seed: int = 0,
) -> plt.Figure:
    """Chapter 3 · Bayesian linear regression, live.

    Slide the **sample size ``N``** to watch the posterior over the regression line
    tighten as data accumulates, and the **prior precision** to see how a stronger
    prior regularizes the fit. A fixed pool of ``n_max`` noisy observations is drawn
    once; the ``N`` slider uses the first ``N`` of them, so the animation is a true
    sequential-evidence story. The shaded band is the ±2σ posterior-predictive
    interval; the readout reports the recovered slope/intercept with their posterior
    standard deviations. Companion to ``example_3_5_bayesian_linear_regression.py``.
    """
    from ..estimators.linear_regression import BayesianLinearRegression

    rng = np.random.default_rng(seed)
    x_pool = rng.uniform(x_low, x_high, size=n_max)
    y_pool = beta0_true + beta1_true * x_pool + rng.normal(0.0, noise_std, size=n_max)
    x_line = np.linspace(x_low, x_high, 200)

    def fit(n: int, prior_precision: float):
        """Fit BLR on the first ``n`` pooled points at the given prior precision."""
        n = max(int(n), 2)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(2), prior_cov=np.eye(2) / prior_precision,
            sigma2_y=noise_std ** 2, intercept=True,
        )
        post = blr.fit(x_pool[:n].reshape(-1, 1), y_pool[:n])
        mean, var = post.predictive(x_line.reshape(-1, 1), sigma2_y=noise_std ** 2)
        return post, mean, np.sqrt(np.maximum(var, 0.0)), n

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.26)
    blr, mean, sd, n0 = fit(10, 1.0)
    scatter = ax.scatter(x_pool[:n0], y_pool[:n0], s=20, color=COLORS["sensory"],
                         alpha=0.7, zorder=3)
    ax.plot(x_line, beta0_true + beta1_true * x_line, color=COLORS["truth"], ls=":",
            lw=1.8, label="true line")
    (mean_line,) = ax.plot(x_line, mean, color=COLORS["posterior"], lw=2.2,
                           label="posterior mean")
    band = ax.fill_between(x_line, mean - 2 * sd, mean + 2 * sd, alpha=0.2,
                           color=COLORS["posterior"], label="±2σ predictive")
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bayesian linear regression — posterior tightens with N")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    stat_text = annotate_stat_box(ax, "", loc="lower right", fontsize=9, monospace=False)

    ax_n = plt.axes([0.12, 0.12, 0.76, 0.04])
    ax_p = plt.axes([0.12, 0.05, 0.76, 0.04])
    s_n = Slider(ax_n, "sample size N", 2, n_max, valinit=10, valstep=1)
    s_p = Slider(ax_p, "prior precision", 1e-2, 10.0, valinit=1.0)

    def update(_event=None) -> None:
        """Refit BLR and refresh the band and coefficient readout from the sliders."""
        nonlocal band
        post, mean, sd, n = fit(s_n.val, s_p.val)
        scatter.set_offsets(np.column_stack([x_pool[:n], y_pool[:n]]))
        mean_line.set_ydata(mean)
        for coll in [c for c in ax.collections if c is band]:
            coll.remove()
        band = ax.fill_between(x_line, mean - 2 * sd, mean + 2 * sd, alpha=0.2,
                               color=COLORS["posterior"])
        post_sd = post.std()
        stat_text.set_text(
            f"N = {n}\n"
            f"β0 = {post.mean[0]:.3f} ± {post_sd[0]:.3f}\n"
            f"β1 = {post.mean[1]:.3f} ± {post_sd[1]:.3f}\n"
            f"(true {beta0_true:.3f}, {beta1_true:.3f})"
        )
        fig.canvas.draw_idle()

    update()
    for s in (s_n, s_p):
        s.on_changed(update)
    fig._sliders = (s_n, s_p)  # type: ignore[attr-defined]
    return fig


def interactive_predictive_coding(
    *,
    y_init: float = 7.0,
    m_x_init: float = 4.0,
    s2_x_init: float = 0.25,
    sigma2_y_init: float = 0.25,
    mu_low: float = 0.0,
    mu_high: float = 6.0,
    n_grid: int = 601,
) -> plt.Figure:
    """Chapter 5 · predictive coding, live.

    The free-energy landscape ``F(μ) = ½(λ_y ε_y² + λ_x ε_x²) + const`` for the
    linear model ``g(x)=2x+3``. Drag the **observation ``y``**, **prior mean
    ``m_x``**, **prior variance ``s_x²``**, and **likelihood variance ``σ_y²``**
    sliders to watch the two precision-weighted prediction errors trade off and the
    free-energy minimum ``μ*`` slide between the data-consistent state and the prior
    mean (the interactive form of Example 5.2). The minimum is marked with the
    closed-form :func:`~active_inference.pc_linear_fixed_point`, and the readout
    reports ``μ*``, both prediction errors there, and the precision ratio.
    """
    mu_grid = make_grid(mu_low, mu_high, n_grid)

    def compute(y, m_x, s2_x, sigma2_y):
        """Free-energy curve, weighted PE curves, and closed-form minimum."""
        model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=sigma2_y,
                                      m_x=m_x, s2_x=s2_x)
        fe = np.array([predictive_coding_free_energy(model, y, float(m)).free_energy
                       for m in mu_grid])
        w_ey = model.lambda_y * (y - (2.0 * mu_grid + 3.0)) ** 2
        w_ex = model.lambda_x * (mu_grid - m_x) ** 2
        mu_star = pc_linear_fixed_point(model, y)
        return model, fe, w_ey, w_ex, mu_star

    fig, (ax_fe, ax_pe) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.34, wspace=0.28)
    model, fe, w_ey, w_ex, mu_star = compute(y_init, m_x_init, s2_x_init, sigma2_y_init)

    (fe_line,) = ax_fe.plot(mu_grid, fe, color=COLORS["data"], lw=2.2, label=r"$\mathcal{F}$")
    min_marker = ax_fe.axvline(mu_star, color=COLORS["posterior"], ls="--", lw=1.6,
                               label=r"$\mu^*$")
    ax_fe.set_xlim(mu_low, mu_high)
    ax_fe.set_xlabel(r"$\mu_x$")
    ax_fe.set_ylabel(r"$\mathcal{F}$")
    ax_fe.set_title("free-energy landscape")
    ax_fe.grid(alpha=0.3)
    ax_fe.legend(loc="upper center", fontsize=8)
    stat_text = annotate_stat_box(ax_fe, "", loc="upper right", fontsize=9, monospace=False)

    (ey_line,) = ax_pe.plot(mu_grid, w_ey, color=COLORS["sensory"], lw=2.0,
                            label=r"$\lambda_y\,\varepsilon_y^2$ (sensory)")
    (ex_line,) = ax_pe.plot(mu_grid, w_ex, color=COLORS["state"], lw=2.0,
                            label=r"$\lambda_x\,\varepsilon_x^2$ (state)")
    ax_pe.set_xlim(mu_low, mu_high)
    ax_pe.set_xlabel(r"$\mu_x$")
    ax_pe.set_ylabel("weighted squared error")
    ax_pe.set_title("the two prediction errors")
    ax_pe.grid(alpha=0.3)
    ax_pe.legend(loc="upper center", fontsize=8)

    ax_y = plt.axes([0.10, 0.20, 0.80, 0.03])
    ax_mx = plt.axes([0.10, 0.15, 0.80, 0.03])
    ax_sx = plt.axes([0.10, 0.10, 0.80, 0.03])
    ax_sy = plt.axes([0.10, 0.05, 0.80, 0.03])
    s_y = Slider(ax_y, "observation y", 0.0, 14.0, valinit=y_init)
    s_mx = Slider(ax_mx, "prior mean m_x", mu_low, mu_high, valinit=m_x_init)
    s_sx = Slider(ax_sx, "prior var s_x²", 1e-2, 4.0, valinit=s2_x_init)
    s_sy = Slider(ax_sy, "lik var σ_y²", 1e-2, 4.0, valinit=sigma2_y_init)

    def update(_event=None) -> None:
        """Refresh the landscape, PE curves, and readout from the sliders."""
        model, fe, w_ey, w_ex, mu_star = compute(s_y.val, s_mx.val, s_sx.val, s_sy.val)
        fe_line.set_ydata(fe)
        ey_line.set_ydata(w_ey)
        ex_line.set_ydata(w_ex)
        min_marker.set_xdata([mu_star, mu_star])
        fe_lo = float(np.min(fe))
        # F(μ) is quadratic near μ* with curvature L = λ_x + λ_y g'² (g=2x+3).
        # Size the vertical window to that curvature so the ±3σ basin stays in
        # view uniformly: a fixed window clips a steep V (small σ_y²) into a
        # wall or flattens a wide V (large σ_y²) into a line.
        curvature = model.lambda_x + 4.0 * model.lambda_y
        sigma_mu = 1.0 / np.sqrt(max(curvature, 1e-12))
        fe_top = fe_lo + 0.5 * curvature * (3.0 * sigma_mu) ** 2
        ax_fe.set_ylim(fe_lo - 0.5, fe_top if np.isfinite(fe_top) else fe_lo + 5.0)
        ax_pe.set_ylim(0, max(float(np.max(w_ey)), float(np.max(w_ex)), 1e-6) * 1.1)
        eps_y = sensory_prediction_error(model, s_y.val, mu_star)
        eps_x = state_prediction_error(model, mu_star)
        stat_text.set_text(
            f"μ* = {mu_star:.3f}\n"
            f"ε_y = {eps_y:.3f}\n"
            f"ε_x = {eps_x:.3f}\n"
            f"λ_x/λ_y = {model.lambda_x / model.lambda_y:.2f}"
        )
        fig.canvas.draw_idle()

    update()
    for s in (s_y, s_mx, s_sx, s_sy):
        s.on_changed(update)
    fig._sliders = (s_y, s_mx, s_sx, s_sy)  # type: ignore[attr-defined]
    return fig


def interactive_variational_free_energy(
    *,
    beta0: float = 3.0,
    beta1: float = 2.0,
    sigma2_y: float = 0.25,
    m_x: float = 4.0,
    s2_x: float = 0.25,
    y_obs: float = 7.0,
    x_low: float = -6.0,
    x_high: float = 12.0,
    n_grid: int = 1201,
    mu_low: float = 0.0,
    mu_high: float = 5.0,
    mu_init: float = 4.0,
    var_low: float = 0.02,
    var_high: float = 2.0,
    var_init: float = 0.25,
) -> plt.Figure:
    """Chapter 4 (bonus) · variational free energy, live.

    Two sliders move the variational density ``q(x) = N(μ_x, σ²_x)``. The top
    panel overlays ``q(x)`` on the exact grid posterior; the bottom panel shows
    the live decomposition of variational free energy (free energy, divergence
    from the posterior, complexity, accuracy) so you can *feel* how the terms
    trade off as you drag the belief around. Free energy bottoms out exactly
    when ``q`` sits on the posterior — at which point divergence is zero and
    ``F = −log p(y)``.
    """
    x_grid = make_grid(x_low, x_high, n_grid)
    model = LinearGaussianModel(beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
                                m_x=m_x, s2_x=s2_x)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(y_obs)
    le, post = float(exact.log_evidence), np.asarray(exact.posterior)

    fig, (ax_q, ax_bar) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=False)
    plt.subplots_adjust(bottom=0.22, hspace=0.35)
    ax_q.plot(x_grid, post, color="black", ls="--", lw=2, label="posterior")
    (line_q,) = ax_q.plot([], [], color=COLORS["posterior"], lw=2, label="q(x)")
    ax_q.set_xlim(x_low, x_high)
    ax_q.set_xlabel("food size x")
    ax_q.set_ylabel("density")
    ax_q.legend(loc="upper right")
    stat_text = annotate_stat_box(ax_q, "", loc="upper left", fontsize=9,
                                  monospace=False)

    labels = ["F", "divergence", "complexity", "accuracy"]
    bars = ax_bar.bar(labels, [0, 0, 0, 0],
                      color=[COLORS["prior"], COLORS["posterior"],
                             COLORS["truth"], COLORS["state"]])
    ax_bar.set_ylabel("nats")
    ax_bar.axhline(-le, color=COLORS["likelihood"], ls=":",
                   label=f"−log p(y) = {-le:.2f}")
    ax_bar.legend(loc="upper right", fontsize=9)

    ax_mu = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_var = plt.axes([0.15, 0.05, 0.7, 0.03])
    s_mu = Slider(ax_mu, r"$\mu_x$", mu_low, mu_high, valinit=mu_init)
    s_var = Slider(ax_var, r"$\sigma_x^2$", var_low, var_high, valinit=var_init)

    def update(_event=None) -> None:
        """Update interactive or animated artists for the current state."""
        q = GaussianBelief(s_mu.val, max(s_var.val, 1e-3))
        line_q.set_data(x_grid, q.pdf(x_grid))
        c = variational_free_energy(q, model, y_obs, x_grid,
                                    log_evidence=le, posterior=post)
        for bar, val in zip(bars, [c.free_energy, c.divergence,
                                   c.complexity, c.accuracy]):
            bar.set_height(val)
        ax_bar.relim()
        ax_bar.autoscale_view()
        stat_text.set_text(
            f"μ = {s_mu.val:.3f}\n"
            f"σ² = {q.var:.4f}\n"
            f"F = {c.free_energy:.3f}\n"
            f"div = {c.divergence:.3f}\n"
            f"cplx = {c.complexity:.3f}\n"
            f"acc = {c.accuracy:.3f}"
        )
        fig.canvas.draw_idle()

    s_mu.on_changed(update)
    s_var.on_changed(update)
    update()
    fig.suptitle("Interactive VFE explorer — drag μ and σ² onto the posterior")
    fig._sliders = (s_mu, s_var)  # type: ignore[attr-defined]
    return fig


def interactive_gradient_descent(
    *,
    beta0: float = 3.0,
    beta1: float = 2.0,
    sigma2_y: float = 0.5,
    x_true: float = 2.5,
    seed: int = 12,
    n_samples: int = 200,
    x0: float = 5.0,
    max_iter: int = 60,
    x_low: float = -2.0,
    x_high: float = 7.0,
    n_grid: int = 500,
    log_lr_low: float = -6.0,
    log_lr_high: float = -2.5,
    log_lr_init: float = -4.0,
) -> plt.Figure:
    """Chapter 2 · gradient descent trajectory scrubber.

    Reuses the exact MLE loss/gradient (:func:`~active_inference.estimators.mle.mle_loss`,
    :func:`~active_inference.estimators.mle.mle_grad_x`) and optimizer
    (:func:`~active_inference.estimators.gradient_descent.gradient_descent`) as
    ``example_2_10_gradient_descent.py`` and ``animation_gradient_descent.py``, on a
    fixed pool of observations drawn once from :class:`~active_inference.LinearGaussianProcess`.

    Two sliders give a different view than the fixed-speed animation:

    * **log10(learning rate)** — moving this slider *recomputes the whole
      trajectory* from the same fixed start point ``x0``. Drag it past the
      stability threshold ``2 / (n · β₁² / σ_y²)`` and the iterate overshoots
      and diverges instead of descending — something a single fixed-speed
      animation cannot show side by side with convergent behaviour.
    * **iteration index** — scrubs through the *current* trajectory, redrawing
      the iterate on the loss surface and the loss-vs-iteration trace up to
      that step, so you can freeze on any point instead of watching it play.

    The readout reports the current iterate, its loss, and whether the
    trajectory is still shrinking its step size (a simple converging/diverging
    heuristic on the last two steps).
    """
    from ..core.generative_process import LinearGaussianProcess
    from ..estimators.gradient_descent import gradient_descent
    from ..estimators.mle import mle_grad_x, mle_loss

    rng = np.random.default_rng(seed)
    process = LinearGaussianProcess(beta0=beta0, beta1=beta1, sigma2_y=sigma2_y, rng=rng)
    ys = process.sample(x_true, n=n_samples).flatten()

    x_grid = np.linspace(x_low, x_high, n_grid)
    loss_grid = np.asarray(mle_loss(x_grid, ys, beta0, beta1, sigma2_y), dtype=float)

    def trajectory(log_lr: float):
        """Run gradient descent from the fixed start point at the given learning rate."""
        lr = 10.0 ** log_lr
        return gradient_descent(
            loss_fn=lambda x: float(mle_loss(x, ys, beta0, beta1, sigma2_y)),
            grad_fn=lambda x: mle_grad_x(x, ys, beta0, beta1, sigma2_y),
            x0=x0,
            learning_rate=lr,
            max_iter=max_iter,
        )

    state = {"result": trajectory(log_lr_init)}

    fig, (ax_surf, ax_trace) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.32, wspace=0.28)

    ax_surf.plot(x_grid, loss_grid, color="#888", lw=1.5)
    ax_surf.axvline(x_true, color=COLORS["truth"], ls=":", lw=1.2, label="x* (truth)")
    ax_surf.set_xlim(x_low, x_high)
    ax_surf.set_xlabel("x")
    ax_surf.set_ylabel("loss")
    ax_surf.set_title("Loss surface + iterate")
    ax_surf.grid(alpha=0.3)
    ax_surf.legend(loc="upper right", fontsize=8)
    (trail_line,) = ax_surf.plot([], [], color=COLORS["likelihood"], lw=1, alpha=0.5)
    (point_marker,) = ax_surf.plot([], [], "o", color=COLORS["likelihood"], ms=8, zorder=5)

    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("loss")
    ax_trace.set_title("Loss vs iteration (scrubbed)")
    ax_trace.grid(alpha=0.3)
    (loss_line,) = ax_trace.plot([], [], color=COLORS["prior"], lw=2)
    scrub_marker = ax_trace.axvline(0.0, color=COLORS["data"], ls="--", lw=1.0)

    stat_text = annotate_stat_box(ax_surf, "", loc="upper left", fontsize=9, monospace=False)

    ax_lr = plt.axes([0.10, 0.16, 0.80, 0.04])
    ax_it = plt.axes([0.10, 0.07, 0.80, 0.04])
    s_lr = Slider(ax_lr, "log10(learning rate)", log_lr_low, log_lr_high, valinit=log_lr_init)
    s_it = Slider(ax_it, "iteration", 0, max_iter, valinit=max_iter, valstep=1)

    def scrub(_event=None) -> None:
        """Redraw the current iterate and loss trace up to the scrubbed iteration."""
        result = state["result"]
        idx = int(np.clip(s_it.val, 0, len(result.history) - 1))
        hist = result.history[: idx + 1]
        losses = result.losses[: idx + 1]
        loss_here = np.asarray(mle_loss(hist, ys, beta0, beta1, sigma2_y), dtype=float)
        trail_line.set_data(hist, loss_here)
        point_marker.set_data([hist[-1]], [loss_here[-1]])
        loss_line.set_data(np.arange(idx + 1), losses)
        scrub_marker.set_xdata([idx, idx])
        step = float(hist[-1] - hist[-2]) if idx >= 1 else float("nan")
        # A learning rate above the stability threshold drives the iterate to
        # inf/NaN (the point of the slider); report those states without crashing.
        last_loss = losses[-1]
        loss_str = f"{last_loss:.3g}" if np.isfinite(last_loss) else "(∞/NaN)"
        last_x = hist[-1]
        x_str = f"{last_x:.3f}" if np.isfinite(last_x) else "(∞)"
        step_str = f"{step:.3g}" if np.isfinite(step) else "(∞)"
        stat_text.set_text(
            f"iter = {idx} / {len(result.history) - 1}\n"
            f"x    = {x_str}\n"
            f"loss = {loss_str}\n"
            f"Δx   = {step_str}\n"
            f"{'converged' if result.converged else 'diverging' if idx >= 2 and abs(step) > abs(float(hist[-2] - hist[-3])) else 'descending'}"
        )
        fig.canvas.draw_idle()

    def on_lr_change(_event=None) -> None:
        """Recompute the full trajectory from ``x0`` at the new learning rate."""
        result = trajectory(s_lr.val)
        state["result"] = result
        new_max = max(len(result.history) - 1, 1)
        s_it.valmax = new_max
        s_it.ax.set_xlim(s_it.valmin, new_max)
        finite_losses = result.losses[np.isfinite(result.losses)]
        if finite_losses.size:
            ax_trace.set_xlim(0, new_max)
            ax_trace.set_ylim(float(np.min(finite_losses)) * 0.95,
                              float(np.max(finite_losses)) * 1.05 + 1e-6)
        s_it.set_val(new_max)  # also triggers scrub()

    s_lr.on_changed(on_lr_change)
    s_it.on_changed(scrub)
    on_lr_change(log_lr_init)

    fig._sliders = (s_lr, s_it)  # type: ignore[attr-defined]
    fig._state = state  # type: ignore[attr-defined]  # {"result": GradientDescentResult}
    return fig


def interactive_lgs_localization(
    *,
    Theta: "np.ndarray | None" = None,
    cov_y: "np.ndarray | None" = None,
    prior_mean: "np.ndarray | None" = None,
    cov_x: "np.ndarray | None" = None,
    y1_low: float = -0.5,
    y1_high: float = 1.5,
    y2_low: float = -0.5,
    y2_high: float = 1.5,
    y_init: "tuple[float, float]" = (0.4, 0.6),
) -> plt.Figure:
    """Chapter 3 · 2-D LGS food-localization explorer.

    Reuses the exact multivariate closed-form update from
    ``example_3_6_lgs_food_localization.py`` — a single
    :class:`~active_inference.LinearGaussianSystem` with ``Θ = I`` and
    :meth:`~active_inference.LinearGaussianSystem.posterior` (the
    single-observation sensor-fusion update, not the batch form).

    The two sliders are the **coordinates of a single noisy observation**
    ``y = (y₁, y₂)`` — the most direct hands-on view of vector-valued
    sensor fusion: as you drag the observation around, the posterior mean
    ellipse slides along the precision-weighted line between the fixed prior
    mean and ``y`` (mirroring the scalar precision-weighting story from
    Chapters 2 and 5, now in 2-D). The prior ellipse stays fixed as a visual
    anchor; the posterior ellipse (shape set by ``Θ``, ``Σ_y``, ``Σ_x`` — all
    fixed here) moves and the live readout reports the posterior mean/std and
    how far it sits from both the prior mean and the dragged observation.
    """
    from ..core.distributions import diagonal_cov, isotropic_cov
    from ..core.lgs import LinearGaussianSystem
    from .plotting import confidence_ellipse

    Theta = np.eye(2) if Theta is None else np.asarray(Theta, dtype=float)
    cov_y = diagonal_cov([0.07, 0.06]) if cov_y is None else np.asarray(cov_y, dtype=float)
    prior_mean = np.array([0.5, 0.5]) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    cov_x = isotropic_cov(2, 0.5) if cov_x is None else np.asarray(cov_x, dtype=float)

    lgs = LinearGaussianSystem(Theta=Theta, cov_y=cov_y, mx=prior_mean, cov_x=cov_x)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=False)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.26)

    prior_patches = []
    for n_std, alpha in zip((1, 2), (0.18, 0.07)):
        patch = confidence_ellipse(lgs.mx, lgs.cov_x, n_std=n_std,
                                   fc=COLORS["prior"], ec=COLORS["prior"], alpha=alpha, lw=1.0)
        ax.add_patch(patch)
        prior_patches.append(patch)

    post0 = lgs.posterior(np.asarray(y_init, dtype=float))
    post_patches = []

    def draw_posterior(mean, cov):
        """Add the two posterior confidence ellipses for the given mean/cov."""
        added = []
        for n_std, alpha in zip((1, 2), (0.5, 0.22)):
            patch = confidence_ellipse(mean, cov, n_std=n_std, fc=COLORS["posterior"],
                                       ec=COLORS["posterior"], alpha=alpha, lw=1.5)
            ax.add_patch(patch)
            added.append(patch)
        return added

    post_patches.extend(draw_posterior(post0.mean, post0.cov))

    obs_marker = ax.scatter(*y_init, color="white", edgecolor="black", s=40, lw=0.8,
                            zorder=4, label="observation y")
    mean_marker = ax.scatter(*post0.mean, color=COLORS["posterior"], s=70, marker="o",
                             zorder=5, label="posterior mean")
    ax.scatter(*prior_mean, color=COLORS["prior"], marker="s", s=70, label="prior mean")
    ax.scatter(0, 0, marker="^", color="black", s=80, label="agent")

    ax.set_xlim(min(y1_low, prior_mean[0]) - 0.3, max(y1_high, prior_mean[0]) + 0.3)
    ax.set_ylim(min(y2_low, prior_mean[1]) - 0.3, max(y2_high, prior_mean[1]) + 0.3)
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("LGS sensor fusion — drag the observation")
    ax.legend(loc="lower right", fontsize=8)

    stat_text = annotate_stat_box(ax, "", loc="upper left", fontsize=9, monospace=False)

    ax_y1 = plt.axes([0.15, 0.13, 0.75, 0.03])
    ax_y2 = plt.axes([0.15, 0.07, 0.75, 0.03])
    s_y1 = Slider(ax_y1, "observation y₁", y1_low, y1_high, valinit=float(y_init[0]))
    s_y2 = Slider(ax_y2, "observation y₂", y2_low, y2_high, valinit=float(y_init[1]))

    def update(_event=None) -> None:
        """Recompute the single-observation posterior and refresh the ellipses/readout."""
        y = np.array([s_y1.val, s_y2.val])
        post = lgs.posterior(y)
        for patch in post_patches:
            patch.remove()
        post_patches.clear()
        post_patches.extend(draw_posterior(post.mean, post.cov))
        obs_marker.set_offsets([y])
        mean_marker.set_offsets([post.mean])
        std = post.std()
        dist_to_prior = float(np.linalg.norm(post.mean - prior_mean))
        dist_to_obs = float(np.linalg.norm(post.mean - y))
        stat_text.set_text(
            f"y = ({y[0]:.2f}, {y[1]:.2f})\n"
            f"posterior mean = ({post.mean[0]:.3f}, {post.mean[1]:.3f})\n"
            f"posterior std  = ({std[0]:.3f}, {std[1]:.3f})\n"
            f"‖mean−prior‖ = {dist_to_prior:.3f}\n"
            f"‖mean−y‖     = {dist_to_obs:.3f}"
        )
        fig.canvas.draw_idle()

    update()
    s_y1.on_changed(update)
    s_y2.on_changed(update)
    fig._sliders = (s_y1, s_y2)  # type: ignore[attr-defined]
    fig._markers = (obs_marker, mean_marker)  # type: ignore[attr-defined]
    return fig
