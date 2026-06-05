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

    s_y = Slider(ax_y, "observation y", x_low, x_low + (x_high - x_low) * 4,
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
        sigma2_y = 1.0 / ratio
        s2_x = ratio
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
