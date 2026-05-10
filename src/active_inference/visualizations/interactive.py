"""Interactive widget-based simulations using matplotlib's slider widget.

These functions are zero-dependency beyond ``matplotlib`` itself — they avoid
the heavier ``ipywidgets`` stack so chapter scripts run in any environment
that has a display.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ..core.generative_model import LinearGaussianModel
from ..core.inference import GridBayesianInference
from ..utils.grids import make_grid


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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.32, wspace=0.35)

    def compute(y, m_x, s2_x, sigma2_y):
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            m_x=m_x, s2_x=s2_x, prior_kind="gaussian",
        )
        return GridBayesianInference(model, x_grid).infer(y)

    res = compute(y_init, m_x_init, s2_x_init, sigma2_y_init)

    line_prior, = axes[0].plot(x_grid, res.prior, color="#1f77b4", lw=2)
    line_lik, = axes[1].plot(x_grid, res.likelihood, color="#d62728", lw=2)
    line_post, = axes[2].plot(x_grid, res.posterior, color="#2ca02c", lw=2)
    mode_marker = axes[2].axvline(res.posterior_mode, color="black", ls="--", lw=1)
    axes[0].set_title("Prior p(x)")
    axes[1].set_title("Likelihood p(y | x)")
    axes[2].set_title("Posterior p(x | y)")
    for ax in axes:
        ax.set_xlim(x_low, x_high)
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    from .style import stat_box_bbox
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
        ratio = 10 ** log_ratio
        sigma2_y = 1.0 / ratio
        s2_x = ratio
        model = LinearGaussianModel(
            beta0=beta0, beta1=beta1, sigma2_y=sigma2_y,
            m_x=m_x, s2_x=s2_x, prior_kind="gaussian",
        )
        return GridBayesianInference(model, x_grid).infer(y_obs)

    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(bottom=0.22, top=0.9)

    res = compute(0.0)
    line_prior, = ax.plot(x_grid, res.prior / np.max(res.prior),
                          lw=2, label="prior", color="#1f77b4")
    line_lik, = ax.plot(x_grid, res.likelihood / np.max(res.likelihood),
                        lw=2, label="likelihood", color="#d62728")
    line_post, = ax.plot(x_grid, res.posterior / np.max(res.posterior),
                         lw=2, label="posterior", color="#2ca02c")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density (peak = 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Posterior is a precision-weighted compromise (drag slider)")

    ax_slider = plt.axes([0.15, 0.07, 0.7, 0.04])
    s = Slider(ax_slider, "log10(s2_x / sigma2_y)", -2.0, 2.0, valinit=0.0)

    from .style import stat_box_bbox
    stat_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
        va="top", ha="left", bbox=stat_box_bbox(pad=0.25),
    )

    def update(_event=None):
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
