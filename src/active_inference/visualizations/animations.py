"""Matplotlib animations for the active-inference workflow.

Each function returns a :class:`matplotlib.animation.FuncAnimation` so the
caller can either ``.save("file.gif", writer="pillow")`` or display it
interactively. We avoid heavier dependencies (ImageMagick, FFmpeg) by sticking
to pillow output, which ships with matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from ..utils import save_animation_data
from .style import COLORS, annotate_stat_box
from .unified import layer_colors


def save_animation(
    anim: FuncAnimation,
    path: Path | str,
    *,
    fps: int = 12,
    dpi: int = 110,
) -> Path:
    """Save ``anim`` as a GIF using the pillow writer (no FFmpeg required)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    animation_metadata = getattr(anim, "_metadata", {})
    if not isinstance(animation_metadata, dict):
        animation_metadata = {}
    save_animation_data(anim, path, metadata={**animation_metadata, "fps": fps, "dpi": dpi})
    anim._draw_was_started = True
    plt.close(anim._fig)  # avoid leaking figure handles
    return path


def animate_sequential_posterior(
    x_grid: np.ndarray,
    posteriors: Sequence[np.ndarray],
    *,
    truth: Optional[float] = None,
    prior: Optional[np.ndarray] = None,
    title: str = "Sequential Bayesian update",
    interval_ms: int = 80,
) -> FuncAnimation:
    """Animate a sequence of posterior densities on a shared x-grid.

    Parameters
    ----------
    x_grid : np.ndarray
    posteriors : sequence of arrays
        Each entry is a posterior density on ``x_grid`` after assimilating
        one more observation.
    truth : float, optional
        Vertical reference line for the true hidden state.
    prior : np.ndarray, optional
        If provided, drawn as a faint dashed reference for context.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(x_grid.min(), x_grid.max())
    peak = max(np.max(p) for p in posteriors)
    ax.set_ylim(0, peak * 1.1)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.set_title(title)

    if prior is not None:
        ax.plot(x_grid, prior, color="#1f77b4", ls="--", lw=1.2,
                alpha=0.5, label="prior")
    if truth is not None:
        ax.axvline(truth, color="red", ls=":", lw=1.5,
                   label=f"x* = {truth:.3f}")

    line, = ax.plot([], [], color="#2ca02c", lw=2, label="posterior")
    fill = ax.fill_between(x_grid, np.zeros_like(x_grid), alpha=0.25,
                           color="#2ca02c")
    txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))
    ax.legend(loc="upper right", fontsize=9)

    def init():
        """Initialize animation artists before the first rendered frame."""
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        nonlocal fill
        post = posteriors[frame]
        line.set_data(x_grid, post)
        # Re-draw fill (matplotlib does not animate `PolyCollection` updates).
        for coll in [c for c in ax.collections if c is fill]:
            coll.remove()
        fill = ax.fill_between(x_grid, post, alpha=0.25, color="#2ca02c")
        mode = float(x_grid[int(np.argmax(post))])
        txt.set_text(f"N = {frame + 1}\nposterior mode = {mode:.3f}")
        return line, txt

    anim = FuncAnimation(fig, update, frames=len(posteriors),
                         init_func=init, interval=interval_ms, blit=False,
                         repeat_delay=800)
    anim._fig = fig
    return anim


def animate_gradient_descent(
    loss_grid: np.ndarray,
    x_grid: np.ndarray,
    history: np.ndarray,
    losses: np.ndarray,
    *,
    truth: Optional[float] = None,
    title: str = "Gradient descent",
    interval_ms: int = 60,
) -> FuncAnimation:
    """Animate a 1-D gradient descent trajectory rolling down a loss curve."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(x_grid, loss_grid, color="#888", lw=1.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss surface + iterate")
    if truth is not None:
        axes[0].axvline(truth, color="red", ls=":", lw=1, label="x*")
        axes[0].legend()
    axes[0].grid(alpha=0.3)
    point, = axes[0].plot([], [], "o", color="#d62728", ms=8)
    trail, = axes[0].plot([], [], color="#d62728", lw=1, alpha=0.4)

    axes[1].set_xlim(0, len(history))
    axes[1].set_ylim(min(losses) * 0.95, max(losses) * 1.05)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss vs iteration")
    axes[1].grid(alpha=0.3)
    loss_line, = axes[1].plot([], [], color="#1f77b4", lw=2)

    def init():
        """Initialize animation artists before the first rendered frame."""
        point.set_data([], [])
        trail.set_data([], [])
        loss_line.set_data([], [])
        return point, trail, loss_line

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        x = history[frame]
        # Read loss off the precomputed grid for visual consistency.
        idx = int(np.clip(np.searchsorted(x_grid, x), 0, len(x_grid) - 1))
        y = float(loss_grid[idx])
        point.set_data([x], [y])
        trail.set_data(history[: frame + 1],
                       [loss_grid[int(np.clip(np.searchsorted(x_grid, h), 0,
                                              len(x_grid) - 1))]
                        for h in history[: frame + 1]])
        loss_line.set_data(np.arange(frame + 1), losses[: frame + 1])
        return point, trail, loss_line

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=len(history),
                         init_func=init, interval=interval_ms, blit=False)
    anim._fig = fig
    return anim


def _ellipse_from_cov(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n_std: float = 2.0,
    **kwargs,
) -> Ellipse:
    """Confidence ellipse for a 2-D Gaussian.

    Standard derivation: the eigenvectors of the covariance give the axes,
    and the eigenvalues set the half-lengths scaled by ``n_std``.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def animate_2d_posterior(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    truth: Optional[np.ndarray] = None,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    labels: Tuple[str, str] = (r"$\theta_0$", r"$\theta_1$"),
    title: str = "Posterior tightening",
    interval_ms: int = 100,
) -> FuncAnimation:
    """Animate a sequence of 2-D Gaussian posteriors as confidence ellipses.

    Parameters
    ----------
    means : np.ndarray, shape ``(K, 2)``
    covs : np.ndarray, shape ``(K, 2, 2)``
    """
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    if prior_mean is not None and prior_cov is not None:
        for n_std, alpha in zip((1, 2), (0.18, 0.08)):
            ax.add_patch(_ellipse_from_cov(
                np.asarray(prior_mean), np.asarray(prior_cov),
                n_std=n_std, fc="#1f77b4", ec="#1f77b4", alpha=alpha,
                lw=1.2,
            ))
    if truth is not None:
        ax.scatter(*truth, marker="x", color="red", s=80, lw=2, label="true θ")

    mean_dot, = ax.plot([], [], "o", color="#2ca02c", ms=6, label="posterior mean")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"),
                  va="top")
    ellipses: list[Ellipse] = []

    def init():
        """Initialize animation artists before the first rendered frame."""
        mean_dot.set_data([], [])
        txt.set_text("")
        return mean_dot, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        nonlocal ellipses
        for e in ellipses:
            e.remove()
        ellipses = []
        m, c = means[frame], covs[frame]
        for n_std, alpha in zip((1, 2), (0.4, 0.18)):
            e = _ellipse_from_cov(
                m, c, n_std=n_std, fc="#2ca02c", ec="#2ca02c",
                alpha=alpha, lw=1.5,
            )
            ax.add_patch(e)
            ellipses.append(e)
        mean_dot.set_data([m[0]], [m[1]])
        txt.set_text(
            f"N = {frame + 1}\n"
            f"mean = ({m[0]:.2f}, {m[1]:.2f})\n"
            f"std  = ({np.sqrt(c[0, 0]):.3f}, {np.sqrt(c[1, 1]):.3f})"
        )
        return mean_dot, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=means.shape[0],
                         init_func=init, interval=interval_ms, blit=False,
                         repeat_delay=800)
    if truth is not None:
        ax.legend(loc="upper right", fontsize=9)
    anim._fig = fig
    return anim


def animate_sufficient_statistics(
    n_axis: np.ndarray,
    running_mean: np.ndarray,
    running_std: np.ndarray,
    running_kl: np.ndarray,
    *,
    truth: Optional[float] = None,
    title: str = "Sufficient statistics over time",
    interval_ms: int = 60,
) -> FuncAnimation:
    """Animate three statistics jointly: posterior mean, std, KL from prior.

    Each frame extends the trace by one observation. ``running_mean``,
    ``running_std``, and ``running_kl`` must all share length with ``n_axis``.
    """
    n_axis = np.asarray(n_axis, dtype=float)
    if not (running_mean.shape == running_std.shape == running_kl.shape == n_axis.shape):
        raise ValueError("traces must share shape with n_axis")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    axes[0].set_title("Posterior mean")
    axes[1].set_title("Posterior std")
    axes[2].set_title("KL[post || prior]  (nats)")
    for ax in axes:
        ax.set_xlabel("samples assimilated")
        ax.grid(alpha=0.3)

    if truth is not None:
        axes[0].axhline(truth, color="red", ls=":", lw=1.5, label="truth")
        axes[0].legend(loc="upper right", fontsize=8)

    line_mean, = axes[0].plot([], [], color="#2ca02c", lw=2)
    line_std,  = axes[1].plot([], [], color="#1f77b4", lw=2)
    line_kl,   = axes[2].plot([], [], color="#d62728", lw=2)
    pad = 0.05
    axes[0].set_xlim(n_axis.min(), n_axis.max())
    axes[0].set_ylim(running_mean.min() - pad, running_mean.max() + pad)
    axes[1].set_xlim(n_axis.min(), n_axis.max())
    axes[1].set_ylim(0, max(running_std) * 1.05 + 1e-6)
    axes[2].set_xlim(n_axis.min(), n_axis.max())
    axes[2].set_ylim(0, max(running_kl.max(), 1e-3) * 1.05)

    txt = axes[2].text(0.02, 0.97, "", transform=axes[2].transAxes,
                       fontsize=9, va="top",
                       bbox=dict(boxstyle="round,pad=0.2",
                                 fc="white", ec="black"))

    def init():
        """Initialize animation artists before the first rendered frame."""
        line_mean.set_data([], [])
        line_std.set_data([], [])
        line_kl.set_data([], [])
        txt.set_text("")
        return line_mean, line_std, line_kl, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        x = n_axis[: frame + 1]
        line_mean.set_data(x, running_mean[: frame + 1])
        line_std.set_data(x, running_std[: frame + 1])
        line_kl.set_data(x, running_kl[: frame + 1])
        txt.set_text(
            f"N = {int(n_axis[frame])}\n"
            f"mean = {running_mean[frame]:.3f}\n"
            f"std  = {running_std[frame]:.3f}\n"
            f"KL   = {running_kl[frame]:.3f}"
        )
        return line_mean, line_std, line_kl, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=n_axis.size, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=800)
    anim._fig = fig
    return anim


def animate_calibration_growth(
    nominal: np.ndarray,
    empirical_history: np.ndarray,
    *,
    title: str = "Calibration curve filling in",
    interval_ms: int = 120,
) -> FuncAnimation:
    """Animate a reliability diagram as the trial count grows.

    Parameters
    ----------
    nominal : (L,) array
        Nominal credible-interval levels.
    empirical_history : (K, L) array
        Each row is the empirical-coverage curve after the first ``i+1``
        trials have been used to estimate it.
    """
    nominal = np.asarray(nominal, dtype=float)
    empirical_history = np.asarray(empirical_history, dtype=float)
    if empirical_history.ndim != 2 or empirical_history.shape[1] != nominal.size:
        raise ValueError("empirical_history must be (K, len(nominal))")
    K = empirical_history.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="red", ls="--", lw=1, label="perfect")
    line, = ax.plot([], [], "o-", color="#1f77b4", lw=2, ms=5,
                    label="empirical")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=10,
                  va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))

    def init():
        """Initialize animation artists before the first rendered frame."""
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        emp = empirical_history[frame]
        line.set_data(nominal, emp)
        ece = float(np.mean(np.abs(emp - nominal)))
        txt.set_text(f"trials so far = {frame + 1}\nECE = {ece:.3f}")
        return line, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=K, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=800)
    anim._fig = fig
    return anim


def animate_precision_sweep(
    x_grid: np.ndarray,
    priors: Sequence[np.ndarray],
    likelihoods: Sequence[np.ndarray],
    posteriors: Sequence[np.ndarray],
    log_ratios: Sequence[float],
    *,
    truth: Optional[float] = None,
    title: str = "Precision sweep · trust prior ↔ trust data",
    interval_ms: int = 90,
) -> FuncAnimation:
    """Animate prior / likelihood / posterior as the precision ratio is swept.

    Each frame shows the three densities at one value of
    ``log10(s2_x / sigma2_y)``. The most subtle effect in Bayesian
    inference — the smooth interpolation between prior-dominated and
    data-dominated posteriors — is rendered as the posterior glides
    between its two limits.
    """
    n = len(priors)
    if not (n == len(likelihoods) == len(posteriors) == len(log_ratios)):
        raise ValueError(
            "priors, likelihoods, posteriors, log_ratios must share length"
        )

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    line_p, = ax.plot([], [], color="#1f77b4", lw=2, label="prior")
    line_l, = ax.plot([], [], color="#d62728", lw=2, label="likelihood")
    line_q, = ax.plot([], [], color="#2ca02c", lw=2, label="posterior")
    if truth is not None:
        ax.axvline(truth, color="black", ls=":", lw=1, label=f"x* = {truth:.3f}")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density (peak normalized)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                  va="top", bbox=dict(boxstyle="round,pad=0.2",
                                       fc="white", ec="black"))

    def init():
        """Initialize animation artists before the first rendered frame."""
        for ln in (line_p, line_l, line_q):
            ln.set_data([], [])
        txt.set_text("")
        return line_p, line_l, line_q, txt

    def update(frame: int):
        # Peak-normalize for visual comparability across frames.
        """Update interactive or animated artists for the current state."""
        pr = priors[frame] / max(np.max(priors[frame]), 1e-12)
        lk = likelihoods[frame] / max(np.max(likelihoods[frame]), 1e-12)
        ps = posteriors[frame] / max(np.max(posteriors[frame]), 1e-12)
        line_p.set_data(x_grid, pr)
        line_l.set_data(x_grid, lk)
        line_q.set_data(x_grid, ps)
        ratio = log_ratios[frame]
        if ratio < 0:
            verdict = "prior dominates"
        elif ratio > 0:
            verdict = "data dominates"
        else:
            verdict = "balanced"
        txt.set_text(
            f"log10(s²_x / σ²_y) = {ratio:+.2f}\n"
            f"({verdict})"
        )
        return line_p, line_l, line_q, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=n, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


def animate_bimodal_emergence(
    x_grid: np.ndarray,
    posteriors: Sequence[np.ndarray],
    prior_means: Sequence[float],
    *,
    truths: Optional[Sequence[float]] = None,
    title: str = "Inverse problem · bimodality emerges as the prior shifts",
    interval_ms: int = 100,
) -> FuncAnimation:
    """Animate a posterior on a non-injective generator as the prior moves.

    Each frame uses a different prior mean; with the right placement the
    posterior is unimodal, with the wrong placement it splits. The
    transition between the two regimes is what the animation surfaces.
    """
    n = len(posteriors)
    if n != len(prior_means):
        raise ValueError("posteriors and prior_means must share length")
    if truths is not None and len(truths) != n:
        raise ValueError("truths must share length with posteriors")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    line, = ax.plot([], [], color="#2ca02c", lw=2)
    ax.set_xlim(x_grid.min(), x_grid.max())
    peak = max(np.max(p) for p in posteriors) * 1.1
    ax.set_ylim(0, peak)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("posterior density")
    ax.grid(alpha=0.3)
    prior_marker = ax.axvline(0.0, color="#1f77b4", ls="--", lw=1.5,
                              label="prior mean")
    truth_marker = None
    if truths is not None:
        truth_marker = ax.axvline(truths[0], color="red", ls=":",
                                  lw=1.5, label="true |x|")
    ax.legend(loc="upper right", fontsize=9)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                  va="top", bbox=dict(boxstyle="round,pad=0.2",
                                       fc="white", ec="black"))
    fill_state = {"poly": None}

    def init():
        """Initialize animation artists before the first rendered frame."""
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        if fill_state["poly"] is not None:
            fill_state["poly"].remove()
        line.set_data(x_grid, posteriors[frame])
        fill_state["poly"] = ax.fill_between(
            x_grid, posteriors[frame], alpha=0.25, color="#2ca02c"
        )
        prior_marker.set_xdata([prior_means[frame], prior_means[frame]])
        if truth_marker is not None:
            truth_marker.set_xdata([truths[frame], truths[frame]])
        # Detect bimodality from the posterior shape.
        peak_idx = int(np.argmax(posteriors[frame]))
        is_bimodal = _is_bimodal(posteriors[frame])
        verdict = "BIMODAL" if is_bimodal else "unimodal"
        txt.set_text(
            f"prior mean = {prior_means[frame]:+.2f}\n"
            f"mode at x ≈ {x_grid[peak_idx]:+.2f}\n"
            f"{verdict}"
        )
        return line, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=n, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


def _is_bimodal(p: np.ndarray, *, peak_ratio: float = 0.4) -> bool:
    """Tiny heuristic for bimodality detection in the bimodal-emergence anim."""
    p = np.asarray(p, dtype=float)
    if p.size < 5:
        return False
    # Find local maxima.
    maxima_mask = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    maxima_indices = np.flatnonzero(maxima_mask) + 1
    if maxima_indices.size < 2:
        return False
    sorted_peaks = sorted(p[maxima_indices], reverse=True)
    return sorted_peaks[1] >= peak_ratio * sorted_peaks[0]


def animate_lgs_online(
    means: np.ndarray,
    covs: np.ndarray,
    observations: np.ndarray,
    *,
    truth: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    title: str = "LGS sensor fusion · one observation at a time",
    interval_ms: int = 100,
) -> FuncAnimation:
    """Animate a 2-D LGS posterior tightening as each new observation arrives.

    Parameters
    ----------
    means : (T, 2)
    covs  : (T, 2, 2)
    observations : (T, 2)
        The observation that produced each posterior.
    """
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    observations = np.asarray(observations, dtype=float)
    T = means.shape[0]
    if covs.shape != (T, 2, 2) or observations.shape != (T, 2):
        raise ValueError("shape mismatch: expected (T, 2), (T, 2, 2), (T, 2)")

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.3)
    if truth is not None:
        ax.scatter(*truth, marker="x", color="red", s=100, lw=2,
                   label=f"truth ({truth[0]:.2f}, {truth[1]:.2f})", zorder=5)

    obs_scatter = ax.scatter([], [], s=18, color="black",
                             alpha=0.5, label="observations", zorder=3)
    mean_dot, = ax.plot([], [], "o", color="#2ca02c", ms=8,
                       label="posterior mean")
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                  va="top", bbox=dict(boxstyle="round,pad=0.2",
                                       fc="white", ec="black"))
    ellipses: list[Ellipse] = []

    def init():
        """Initialize animation artists before the first rendered frame."""
        mean_dot.set_data([], [])
        obs_scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return mean_dot, obs_scatter, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        nonlocal ellipses
        for e in ellipses:
            e.remove()
        ellipses = []
        m, c = means[frame], covs[frame]
        for n_std, alpha in zip((1, 2), (0.45, 0.18)):
            e = _ellipse_from_cov(
                m, c, n_std=n_std, fc="#2ca02c", ec="#2ca02c",
                alpha=alpha, lw=1.5,
            )
            ax.add_patch(e)
            ellipses.append(e)
        mean_dot.set_data([m[0]], [m[1]])
        obs_scatter.set_offsets(observations[: frame + 1])
        # Geometric mean of the posterior ellipse axes.
        eigvals = np.linalg.eigvalsh(c)
        gm = float(np.sqrt(np.prod(np.maximum(eigvals, 0))))
        txt.set_text(
            f"N = {frame + 1}\n"
            f"mean = ({m[0]:+.2f}, {m[1]:+.2f})\n"
            f"|Σ|^{1/2} = {gm:.3f}"
        )
        return mean_dot, obs_scatter, txt

    if truth is not None:
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


def animate_em_steps(
    e_step_means: Sequence[np.ndarray],
    m_step_thetas: Sequence[np.ndarray],
    log_likelihoods: np.ndarray,
    *,
    title: str = "EM · alternating E and M steps",
    interval_ms: int = 250,
) -> FuncAnimation:
    """Animate the alternation between E and M steps with monotone LL.

    Each frame shows two side-by-side panels:
    - Left: scatter of the latent posterior means at the current E-step.
    - Right: heat-map of the M-step loadings matrix.
    A bottom inset traces the marginal log-likelihood, which must be
    monotone non-decreasing across frames.
    """
    if not (len(e_step_means) == len(m_step_thetas)
            == log_likelihoods.size):
        raise ValueError("E-step / M-step / log-likelihood lengths must match")
    K = len(e_step_means)

    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_e = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_ll = fig.add_subplot(gs[1, :])

    # E-step panel
    e_means_all = np.asarray(e_step_means[0])
    if e_means_all.shape[1] >= 2:
        e_scatter = ax_e.scatter([], [], s=12, c="#1f77b4", alpha=0.6)
    else:
        e_scatter = ax_e.scatter([], [], s=12, c="#1f77b4", alpha=0.6)
    pad = 1.2
    arr_first = np.asarray(e_step_means[0])
    if arr_first.shape[1] >= 2:
        x_lo, x_hi = arr_first[:, 0].min() - pad, arr_first[:, 0].max() + pad
        y_lo, y_hi = arr_first[:, 1].min() - pad, arr_first[:, 1].max() + pad
        ax_e.set_xlim(x_lo, x_hi)
        ax_e.set_ylim(y_lo, y_hi)
    ax_e.set_title("E-step · latent posterior means")
    ax_e.set_xlabel("factor 1")
    ax_e.set_ylabel("factor 2")
    ax_e.grid(alpha=0.3)

    # M-step panel
    vmax = max(np.max(np.abs(t)) for t in m_step_thetas)
    im = ax_m.imshow(np.asarray(m_step_thetas[0]), cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax_m, shrink=0.85)
    ax_m.set_title("M-step · loadings Θ")
    ax_m.set_xlabel("factor")
    ax_m.set_ylabel("output dim")

    # Log-likelihood panel
    ax_ll.set_xlim(0, K)
    ax_ll.set_ylim(min(log_likelihoods) - 1, max(log_likelihoods) + 1)
    ax_ll.set_xlabel("EM iteration")
    ax_ll.set_ylabel("incomplete log p(Y)")
    ax_ll.grid(alpha=0.3)
    ll_line, = ax_ll.plot([], [], color="#2ca02c", lw=2)
    txt = ax_ll.text(0.02, 0.85, "", transform=ax_ll.transAxes, fontsize=10,
                     va="top", bbox=dict(boxstyle="round,pad=0.2",
                                          fc="white", ec="black"))

    def init():
        """Initialize animation artists before the first rendered frame."""
        e_scatter.set_offsets(np.empty((0, 2)))
        ll_line.set_data([], [])
        txt.set_text("")
        return e_scatter, im, ll_line, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        arr = np.asarray(e_step_means[frame])
        if arr.shape[1] >= 2:
            e_scatter.set_offsets(arr[:, :2])
        else:
            e_scatter.set_offsets(np.column_stack([arr[:, 0],
                                                   np.zeros(arr.shape[0])]))
        im.set_data(np.asarray(m_step_thetas[frame]))
        ll_line.set_data(np.arange(frame + 1), log_likelihoods[: frame + 1])
        txt.set_text(
            f"iter = {frame + 1}\n"
            f"log p(Y) = {log_likelihoods[frame]:.3f}"
        )
        return e_scatter, im, ll_line, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=K, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


def animate_blr_predictive_band(
    x_grid: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    sigma2_y: float,
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    truth_line: Optional[Tuple[float, float]] = None,
    intercept: bool = True,
    title: str = "BLR · predictive band collapsing onto the truth",
    interval_ms: int = 110,
) -> FuncAnimation:
    """Animate the BLR predictive 95 % band shrinking as data accumulates.

    Each frame: the parameter posterior at step ``t`` produces a
    predictive mean and variance over a fixed ``x_grid``; we plot the
    mean line and its 95 % envelope, plus all data assimilated so far.
    """
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    T = means.shape[0]
    if covs.shape[0] != T:
        raise ValueError("means and covs must share leading dimension")
    if x_data.size != y_data.size or x_data.size != T:
        raise ValueError("x_data and y_data must have length T")

    if intercept:
        X_aug = np.column_stack([np.ones_like(x_grid), x_grid])
    else:
        X_aug = x_grid.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if truth_line is not None:
        b0, b1 = truth_line
        ax.plot(x_grid, b0 + b1 * x_grid, color="red", ls=":", lw=1.5,
                label="true line")
    band = ax.fill_between(x_grid, np.zeros_like(x_grid), np.zeros_like(x_grid),
                           alpha=0.25, color="#2ca02c")
    line, = ax.plot([], [], color="#2ca02c", lw=2, label="predictive mean")
    scatter = ax.scatter([], [], s=14, color="black", alpha=0.6,
                         label="data so far")
    ax.set_xlim(x_grid.min(), x_grid.max())
    pad = max(abs(y_data).max() * 0.2, 1.0)
    ax.set_ylim(y_data.min() - pad, y_data.max() + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    txt = ax.text(0.97, 0.96, "", transform=ax.transAxes, fontsize=10,
                  va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))

    def init():
        """Initialize animation artists before the first rendered frame."""
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return line, scatter, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        nonlocal band
        m, c = means[frame], covs[frame]
        mean_pred = X_aug @ m
        var_pred = sigma2_y + np.einsum("nd,de,ne->n", X_aug, c, X_aug)
        std_pred = np.sqrt(var_pred)
        band.remove()
        band = ax.fill_between(x_grid, mean_pred - 1.96 * std_pred,
                               mean_pred + 1.96 * std_pred,
                               alpha=0.25, color="#2ca02c")
        line.set_data(x_grid, mean_pred)
        scatter.set_offsets(np.column_stack([x_data[: frame + 1],
                                             y_data[: frame + 1]]))
        txt.set_text(
            f"N = {frame + 1}\n"
            f"avg band width = {2 * 1.96 * std_pred.mean():.3f}"
        )
        return line, scatter, txt

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


def animate_em_convergence(
    log_likelihoods: np.ndarray,
    Theta_history: Iterable[np.ndarray],
    *,
    title: str = "EM for factor analysis",
    interval_ms: int = 100,
) -> FuncAnimation:
    """Animate the marginal log-likelihood and a heat-map of ``Θ`` per iter."""
    Theta_history = [np.asarray(t) for t in Theta_history]
    K = len(Theta_history)
    if log_likelihoods.size != K:
        raise ValueError("log_likelihoods length must match Theta_history")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].set_xlim(0, K)
    axes[0].set_ylim(min(log_likelihoods) - 1, max(log_likelihoods) + 1)
    axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("incomplete log p(Y)")
    axes[0].set_title("Marginal log-likelihood")
    axes[0].grid(alpha=0.3)
    line, = axes[0].plot([], [], color="#1f77b4", lw=2)

    vmax = max(np.max(np.abs(t)) for t in Theta_history)
    im = axes[1].imshow(Theta_history[0], cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    axes[1].set_xlabel("latent factor")
    axes[1].set_ylabel("output dim")
    axes[1].set_title("loadings  Θ")

    def init():
        """Initialize animation artists before the first rendered frame."""
        line.set_data([], [])
        return line, im

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        line.set_data(np.arange(frame + 1), log_likelihoods[: frame + 1])
        im.set_data(Theta_history[frame])
        return line, im

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=K, init_func=init,
                         interval=interval_ms, blit=False)
    anim._fig = fig
    return anim


def animate_vfe_descent(
    x_grid: np.ndarray,
    beliefs: Sequence["object"],
    free_energies: Sequence[float],
    *,
    posterior: Optional[np.ndarray] = None,
    surprisal: Optional[float] = None,
    title: str = "Variational free energy descent",
    interval_ms: int = 120,
) -> FuncAnimation:
    """Animate ``q(x)`` tightening onto the posterior as VFE falls (Chapter 4).

    Left panel: the variational density ``q(x)`` at each iteration, with the exact
    ``posterior`` (dashed) as the target. Right panel: variational free energy per
    iteration, with the surprisal bound ``-log p(y)`` (dashed) it descends toward.

    Parameters
    ----------
    beliefs : sequence of GaussianBelief
        One belief per frame (must expose ``.pdf(x)``).
    free_energies : sequence of float
        VFE at each frame, same length as ``beliefs``.
    posterior : ndarray, optional
        Exact posterior density on ``x_grid`` (drawn as the dashed target).
    surprisal : float, optional
        ``-log p(y)``, drawn as the horizontal bound on the free-energy panel.
    """
    n = len(beliefs)
    if n != len(free_energies):
        raise ValueError("beliefs and free_energies must share length")
    x_grid = np.asarray(x_grid, dtype=float)
    fe = np.asarray(free_energies, dtype=float)

    fig, (ax_q, ax_f) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                     constrained_layout=True)
    q0 = beliefs[0].pdf(x_grid)
    ymax = max(float(np.max(q0)),
               float(np.max(posterior)) if posterior is not None else 0.0)
    if posterior is not None:
        ax_q.plot(x_grid, posterior, color=COLORS["data"], ls="--", lw=2.2,
                  label=r"posterior $p(x\,|\,y)$")
    line_q, = ax_q.plot([], [], color=COLORS["posterior"], lw=2.6, label=r"$q(x)$")
    ax_q.set_xlim(x_grid.min(), x_grid.max())
    ax_q.set_ylim(0, 1.25 * max(ymax, 1e-6))
    ax_q.set_xlabel("hidden state x")
    ax_q.set_ylabel("density")
    ax_q.set_title("variational density")
    ax_q.grid(alpha=0.3)
    ax_q.legend(loc="upper right")

    ax_f.plot(np.arange(n), fe, color=COLORS["neutral"], lw=1.0, alpha=0.4)
    pt_f, = ax_f.plot([], [], color=COLORS["prior"], marker="o", ms=9,
                      markeredgecolor="white", markeredgewidth=1.2)
    (f_line,) = ax_f.plot([], [], color=COLORS["prior"], lw=2.4)
    if surprisal is not None:
        ax_f.axhline(surprisal, color=COLORS["likelihood"], ls="--", lw=1.8,
                     label=rf"$-\log p(y)={surprisal:.3f}$ (bound)")
        ax_f.legend(loc="upper right")
    ax_f.set_xlim(-0.5, n - 0.5)
    pad = 0.05 * (fe.max() - fe.min() + 1e-9)
    lo = min(fe.min(), surprisal if surprisal is not None else fe.min())
    ax_f.set_ylim(lo - pad, fe.max() + pad)
    ax_f.set_xlabel("iteration")
    ax_f.set_ylabel(r"$\mathcal{F}$")
    ax_f.set_title("free energy descent")
    ax_f.grid(alpha=0.3)

    txt = annotate_stat_box(ax_q, "", loc="upper left")

    def init():
        """Initialize animation artists before the first rendered frame."""
        line_q.set_data([], [])
        pt_f.set_data([], [])
        f_line.set_data([], [])
        txt.set_text("")
        return line_q, pt_f, f_line, txt

    def update(frame: int):
        """Update interactive or animated artists for the current state."""
        b = beliefs[frame]
        line_q.set_data(x_grid, b.pdf(x_grid))
        pt_f.set_data([frame], [fe[frame]])
        f_line.set_data(np.arange(frame + 1), fe[:frame + 1])
        txt.set_text(f"iter {frame}\nμ  = {b.mu:.3f}\nσ² = {b.var:.3f}\nF  = {fe[frame]:.3f}")
        return line_q, pt_f, f_line, txt

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=n, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=900)
    anim._fig = fig
    return anim


# ---------------------------------------------------------------------------
# Composable recognition-dynamics animator (Chapter 4 fixed-form OR Chapter 5 PC)
# ---------------------------------------------------------------------------


def _frame_indices(n: int, stride: int) -> List[int]:
    """Subsampled frame indices that always include the final iteration."""
    stride = max(1, int(stride))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def animate_recognition_dynamics(
    result,
    *,
    truth: Optional[float] = None,
    oracle: Optional[float] = None,
    surprisal: Optional[float] = None,
    label: str = r"$\mu_x$",
    title: str = "Recognition dynamics",
    interval_ms: int = 90,
    stride: int = 1,
) -> FuncAnimation:
    r"""Animate a recognition / inference descent from any result object.

    The animation counterpart of
    :func:`~active_inference.visualizations.unified.plot_recognition_dynamics`:
    it is **composable** — duck-typed on ``.mus`` / ``.free_energies`` (and the
    optional ``.eps_x`` / ``.eps_y``), so the *same* function animates a Chapter 4
    :class:`~active_inference.estimators.variational.FixedFormResult` (2 panels) and
    a Chapter 5
    :class:`~active_inference.estimators.predictive_coding.PredictiveCodingResult`
    (3 panels). It reuses the package palette and bold typography, draws a legend on
    every panel, and updates a live statistics box (iteration, current ``μ`` and
    ``𝓕``) each frame.

    Parameters
    ----------
    result
        Any object exposing 1-D ``mus`` and ``free_energies`` traces.
    truth, oracle, surprisal : float, optional
        Reference lines: the true state ``x*`` (blue), an oracle such as the grid
        posterior mean (red), and the surprisal floor ``-log p(y)``.
    label : str
        Legend/axis label for the belief-mean trace.
    title : str
        Figure suptitle.
    interval_ms : int
        Delay between frames in milliseconds (playback speed).
    stride : int
        Animate every ``stride``-th iteration (the final frame is always shown).
        Use ``stride > 1`` to keep GIFs small for long descents.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        With ``._fig`` set so :func:`save_animation` can close it cleanly.
    """
    mus = np.asarray(result.mus, dtype=float)
    fes = np.asarray(result.free_energies, dtype=float)
    if mus.ndim != 1 or fes.ndim != 1:
        raise TypeError("result.mus and result.free_energies must be 1-D traces")
    eps_x = getattr(result, "eps_x", None)
    eps_y = getattr(result, "eps_y", None)
    has_errors = eps_x is not None and eps_y is not None
    if has_errors:
        eps_x = np.asarray(eps_x, dtype=float)
        eps_y = np.asarray(eps_y, dtype=float)

    n = mus.shape[0]
    frames = _frame_indices(n, stride)
    it = np.arange(n)
    n_panels = 3 if has_errors else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8),
                             constrained_layout=True)
    ax_mu = axes[0]
    ax_f = axes[-1]

    def _pad(arr):
        """Return padded plot limits for a numeric time-series."""
        lo, hi = float(np.min(arr)), float(np.max(arr))
        m = 0.08 * (hi - lo + 1e-9)
        return lo - m, hi + m

    # --- belief-mean panel (static refs + growing trace + moving marker) ---
    if truth is not None:
        ax_mu.axhline(truth, color=COLORS["prior"], ls="--", lw=1.8,
                      label=rf"truth $x^*={truth:.3g}$")
    if oracle is not None:
        ax_mu.axhline(oracle, color=COLORS["likelihood"], ls=":", lw=2.0,
                      label=rf"oracle $={oracle:.4g}$")
    (mu_line,) = ax_mu.plot([], [], color=COLORS["posterior"], lw=2.6, label=label)
    (mu_dot,) = ax_mu.plot([], [], "o", color=COLORS["posterior"], ms=10,
                           markeredgecolor="white", markeredgewidth=1.3, zorder=6)
    ax_mu.set_xlim(-0.5, n - 0.5)
    ax_mu.set_ylim(*_pad(mus if truth is None else np.r_[mus, truth]))
    ax_mu.set_xlabel("iteration")
    ax_mu.set_ylabel(label)
    ax_mu.set_title("belief mean (recognition)")
    ax_mu.grid(alpha=0.3)
    ax_mu.legend(loc="upper right")
    stat = annotate_stat_box(ax_mu, "", loc="lower left")

    # --- prediction-error panel (Chapter 5 only) ---
    if has_errors:
        ax_e = axes[1]
        ax_e.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
        (ex_line,) = ax_e.plot([], [], color=COLORS["state"], lw=2.6,
                               label=r"$\varepsilon_x$ (state)")
        (ey_line,) = ax_e.plot([], [], color=COLORS["sensory"], lw=2.6,
                               label=r"$\varepsilon_y$ (sensory)")
        ax_e.set_xlim(-0.5, n - 0.5)
        ax_e.set_ylim(*_pad(np.r_[eps_x, eps_y]))
        ax_e.set_xlabel("iteration")
        ax_e.set_ylabel("prediction error")
        ax_e.set_title("precision-weighted prediction errors")
        ax_e.grid(alpha=0.3)
        ax_e.legend(loc="upper right")

    # --- free-energy panel ---
    if surprisal is not None:
        ax_f.axhline(surprisal, color=COLORS["likelihood"], ls="--", lw=1.8,
                     label=rf"$-\log p(y)={surprisal:.3f}$")
    ax_f.plot(it, fes, color=COLORS["neutral"], lw=1.0, alpha=0.35)  # ghost of full path
    (f_line,) = ax_f.plot([], [], color=COLORS["data"], lw=2.6, label=r"$\mathcal{F}$")
    (f_dot,) = ax_f.plot([], [], "o", color=COLORS["data"], ms=9,
                         markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    ax_f.set_xlim(-0.5, n - 0.5)
    lo = min(float(fes.min()), surprisal if surprisal is not None else float(fes.min()))
    ax_f.set_ylim(*_pad(np.r_[fes, lo]))
    ax_f.set_xlabel("iteration")
    ax_f.set_ylabel(r"$\mathcal{F}$")
    ax_f.set_title("free energy descent")
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="upper right")

    artists = [mu_line, mu_dot, f_line, f_dot, stat]
    if has_errors:
        artists += [ex_line, ey_line]

    def init():
        """Initialize animation artists before the first rendered frame."""
        mu_line.set_data([], [])
        mu_dot.set_data([], [])
        f_line.set_data([], [])
        f_dot.set_data([], [])
        if has_errors:
            ex_line.set_data([], [])
            ey_line.set_data([], [])
        stat.set_text("")
        return artists

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        sl = slice(0, k + 1)
        mu_line.set_data(it[sl], mus[sl])
        mu_dot.set_data([it[k]], [mus[k]])
        f_line.set_data(it[sl], fes[sl])
        f_dot.set_data([it[k]], [fes[k]])
        if has_errors:
            ex_line.set_data(it[sl], eps_x[sl])
            ey_line.set_data(it[sl], eps_y[sl])
        stat.set_text(f"iter {k}\nμ = {mus[k]:.4f}\nF = {fes[k]:.4f}")
        return artists

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


def animate_hierarchical_pc(
    result,
    *,
    truth: Optional[Sequence[float]] = None,
    title: str = "Hierarchical predictive coding",
    interval_ms: int = 60,
    stride: int = 1,
) -> FuncAnimation:
    r"""Animate a hierarchical PC run converging (Chapter 5 §5.4, Fig. 5.4.4).

    Composable counterpart of
    :func:`~active_inference.visualizations.unified.plot_hierarchical_pc`: takes a
    :class:`~active_inference.estimators.predictive_coding.HierarchicalPCResult` and
    grows three synchronized panels — each node belief ``μ^{[l]}``, each layer error
    ``ε^{[l]}`` decaying to zero, and the per-layer / summed free energy — using one
    perceptually-ordered colour per layer and bold typography.

    Parameters
    ----------
    result
        A ``HierarchicalPCResult`` (needs ``mus`` ``(T, L+1)``, ``errors``,
        ``free_energies``, ``layer_free_energies``).
    truth : sequence of float, optional
        Per-node target values, drawn as dashed reference lines.
    interval_ms, stride :
        Playback speed and frame subsampling (final frame always shown).
    """
    mus = np.asarray(result.mus, dtype=float)            # (T, L+1)
    errs = np.asarray(result.errors, dtype=float)
    lfes = np.asarray(result.layer_free_energies, dtype=float)
    fes = np.asarray(result.free_energies, dtype=float)
    T, n_nodes = mus.shape
    it = np.arange(T)
    colors = layer_colors(n_nodes)
    frames = _frame_indices(T, stride)

    fig, (ax_mu, ax_e, ax_f) = plt.subplots(1, 3, figsize=(15.5, 4.8),
                                            constrained_layout=True)

    def _pad(arr):
        """Return padded plot limits for a numeric time-series."""
        lo, hi = float(np.min(arr)), float(np.max(arr))
        m = 0.08 * (hi - lo + 1e-9)
        return lo - m, hi + m

    mu_lines, e_lines, lf_lines = [], [], []
    for lvl in range(n_nodes):
        lab = r"$y=\mu^{[0]}$" if lvl == 0 else rf"$\mu^{{[{lvl}]}}$"
        (lm,) = ax_mu.plot([], [], color=colors[lvl], lw=2.4, label=lab)
        mu_lines.append(lm)
        (le,) = ax_e.plot([], [], color=colors[lvl], lw=2.4,
                          label=rf"$\varepsilon^{{[{lvl}]}}$")
        e_lines.append(le)
        (lf,) = ax_f.plot([], [], color=colors[lvl], lw=1.8,
                          label=rf"$\mathcal{{F}}^{{[{lvl}]}}$")
        lf_lines.append(lf)
    (sum_line,) = ax_f.plot([], [], color=COLORS["likelihood"], ls="--", lw=2.6,
                            label=r"$\sum_l \mathcal{F}^{[l]}$")
    if truth is not None:
        for lvl, t in enumerate(truth):
            ax_mu.axhline(t, color=colors[lvl], ls="--", lw=1.2)
    ax_e.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)

    for ax, ylab, ttl, data in (
        (ax_mu, r"$\mu^{[l]}$", "layer beliefs (top-down)", mus),
        (ax_e, r"$\varepsilon^{[l]}$", "layer prediction errors → 0", errs),
        (ax_f, r"$\mathcal{F}$", "per-layer & total free energy",
         np.c_[lfes, fes]),
    ):
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(*_pad(data))
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=1)
    stat = annotate_stat_box(ax_mu, "", loc="lower left")

    artists = mu_lines + e_lines + lf_lines + [sum_line, stat]

    def init():
        """Initialize animation artists before the first rendered frame."""
        for ln in artists[:-1]:
            ln.set_data([], [])
        stat.set_text("")
        return artists

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        sl = slice(0, k + 1)
        for lvl in range(n_nodes):
            mu_lines[lvl].set_data(it[sl], mus[sl, lvl])
            e_lines[lvl].set_data(it[sl], errs[sl, lvl])
            lf_lines[lvl].set_data(it[sl], lfes[sl, lvl])
        sum_line.set_data(it[sl], fes[sl])
        mu_str = "[" + ", ".join(f"{v:.2f}" for v in mus[k]) + "]"
        stat.set_text(f"iter {k}\nμ = {mu_str}\nΣF = {fes[k]:.3e}")
        return artists

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


# ---------------------------------------------------------------------------
# Chapter 9 — dynamic discrete POMDP filtering
# ---------------------------------------------------------------------------


def animate_discrete_beliefs(
    beliefs: np.ndarray,
    *,
    observations: Optional[Sequence[str | int]] = None,
    state_labels: Optional[Sequence[str]] = None,
    title: str = "Dynamic POMDP filtering",
    interval_ms: int = 350,
) -> FuncAnimation:
    r"""Animate a Chapter 9 forward-filtering belief sequence.

    The left panel redraws the current categorical posterior over hidden states, while the
    right panel accumulates each state's belief trajectory over time. It is the animated
    counterpart of :func:`active_inference.visualizations.unified.plot_discrete_belief_sequence`.
    """
    beliefs = np.asarray(beliefs, dtype=float)
    if beliefs.ndim != 2 or beliefs.shape[0] < 1:
        raise ValueError("beliefs must be a non-empty (T, C) array")
    if not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each belief row must sum to 1")
    T, C = beliefs.shape
    labels = list(state_labels) if state_labels is not None else [f"s{k}" for k in range(C)]
    if len(labels) != C:
        raise ValueError(f"state_labels must have length {C}")
    obs_labels = None if observations is None else [str(o) for o in observations]
    if obs_labels is not None and len(obs_labels) != T:
        raise ValueError(f"observations must have length {T}")

    colors = layer_colors(C)
    t = np.arange(T)
    fig, (ax_bar, ax_trace) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    bars = ax_bar.bar(labels, beliefs[0], color=colors, alpha=0.9)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_ylabel("probability")
    ax_bar.set_title("current posterior")
    ax_bar.grid(alpha=0.3, axis="y")
    stat = annotate_stat_box(ax_bar, "", loc="upper right")

    lines = []
    for k in range(C):
        (line,) = ax_trace.plot([], [], color=colors[k], lw=2.4, marker="o", ms=4,
                                label=labels[k])
        lines.append(line)
    ax_trace.set_xlim(0, max(T - 1, 1))
    ax_trace.set_ylim(-0.04, 1.04)
    ax_trace.set_xlabel("time")
    ax_trace.set_ylabel("probability")
    ax_trace.set_title("belief trajectory")
    ax_trace.grid(alpha=0.3)
    ax_trace.legend(loc="upper right", fontsize=9)

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        for rect, h in zip(bars, beliefs[k]):
            rect.set_height(h)
        for s, line in enumerate(lines):
            line.set_data(t[: k + 1], beliefs[: k + 1, s])
        obs = "" if obs_labels is None else f"\nobs = {obs_labels[k]}"
        stat.set_text(
            f"t = {k}{obs}\n"
            f"MAP = {labels[int(np.argmax(beliefs[k]))]}\n"
            f"P = {beliefs[k].max():.2f}"
        )
        return list(bars) + lines + [stat]

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms,
                         blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


def animate_policy_efe_tradeoff(
    risks: np.ndarray,
    ambiguities: np.ndarray,
    *,
    posteriors: Optional[np.ndarray] = None,
    novelties: Optional[np.ndarray] = None,
    frame_labels: Optional[Sequence[str]] = None,
    policy_labels: Optional[Sequence[str]] = None,
    title: str = "Risk/ambiguity policy trade-off",
    interval_ms: int = 300,
) -> FuncAnimation:
    r"""Animate Chapter 9 policy selection as risk and ambiguity change.

    ``risks`` and ``ambiguities`` are ``(K, P)`` arrays: ``K`` frames, ``P`` policies. Each
    frame redraws the decomposition ``G = risk + ambiguity - novelty``. If policy posteriors
    are supplied, the right panel shows ``Q(π)``; otherwise it shows total ``G``.
    """
    risks = np.asarray(risks, dtype=float)
    ambiguities = np.asarray(ambiguities, dtype=float)
    if risks.ndim != 2 or ambiguities.shape != risks.shape:
        raise ValueError("risks and ambiguities must be matching (n_frames, n_policies) arrays")
    K, P = risks.shape
    nov = np.zeros_like(risks) if novelties is None else np.asarray(novelties, dtype=float)
    if nov.shape != risks.shape:
        raise ValueError("novelties must match risks shape")
    posts = None if posteriors is None else np.asarray(posteriors, dtype=float)
    if posts is not None and posts.shape != risks.shape:
        raise ValueError("posteriors must match risks shape")
    labels = list(policy_labels) if policy_labels is not None else [f"π{k}" for k in range(P)]
    if len(labels) != P:
        raise ValueError(f"policy_labels must have length {P}")
    frame_text = list(frame_labels) if frame_labels is not None else [f"frame {k}" for k in range(K)]
    if len(frame_text) != K:
        raise ValueError(f"frame_labels must have length {K}")

    totals = risks + ambiguities - nov
    x = np.arange(P)
    colors = layer_colors(P)
    fig, (ax_decomp, ax_choice) = plt.subplots(1, 2, figsize=(12.8, 4.9),
                                               constrained_layout=True)

    risk_bars = ax_decomp.bar(x, risks[0], color=COLORS["prior"], label="risk")
    amb_bars = ax_decomp.bar(x, ambiguities[0], bottom=risks[0], color=COLORS["sensory"],
                             label="ambiguity")
    novelty_bars = ax_decomp.bar(x, -nov[0], color=COLORS["posterior"], label="novelty")
    total_markers, = ax_decomp.plot(x, totals[0], "o", color=COLORS["data"], ms=9,
                                    label="total G")
    ax_decomp.set_xticks(x)
    ax_decomp.set_xticklabels(labels)
    ax_decomp.set_ylabel("nats")
    ax_decomp.set_title("G = risk + ambiguity − novelty")
    ax_decomp.grid(alpha=0.3, axis="y")
    ax_decomp.legend(loc="upper right", fontsize=9)
    ymin = min(0.0, float((-nov).min())) - 0.05 * float(np.ptp(totals) + 1.0)
    ymax = float((risks + ambiguities).max()) * 1.12 + 1e-9
    ax_decomp.set_ylim(ymin, ymax)
    stat = annotate_stat_box(ax_decomp, "", loc="upper left")

    heights0 = posts[0] if posts is not None else totals[0]
    choice_bars = ax_choice.bar(x, heights0, color=colors, alpha=0.9)
    ax_choice.set_xticks(x)
    ax_choice.set_xticklabels(labels)
    if posts is not None:
        ax_choice.set_ylim(0.0, 1.05)
        ax_choice.set_ylabel(r"$Q(\pi)$")
        ax_choice.set_title("policy posterior")
    else:
        ax_choice.set_ylim(max(0.0, float(totals.min()) * 0.95), float(totals.max()) * 1.08)
        ax_choice.set_ylabel(r"$G(\pi)$")
        ax_choice.set_title("policy totals")
    ax_choice.grid(alpha=0.3, axis="y")

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        for i in range(P):
            risk_bars[i].set_height(risks[k, i])
            amb_bars[i].set_y(risks[k, i])
            amb_bars[i].set_height(ambiguities[k, i])
            novelty_bars[i].set_height(-nov[k, i])
            choice_bars[i].set_height(posts[k, i] if posts is not None else totals[k, i])
        total_markers.set_data(x, totals[k])
        best = int(np.argmin(totals[k]))
        top = int(np.argmax(posts[k])) if posts is not None else best
        stat.set_text(f"{frame_text[k]}\nmin G: {labels[best]}\nshown: {labels[top]}")
        return list(risk_bars) + list(amb_bars) + list(novelty_bars) + list(choice_bars) + [
            total_markers, stat]

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=K, interval=interval_ms,
                         blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


def animate_learning_attention(
    mus: np.ndarray,
    thetas: np.ndarray,
    zetas: np.ndarray,
    free_energies: np.ndarray,
    *,
    truth_x: Optional[np.ndarray] = None,
    truth_theta: Optional[float] = None,
    title: str = "Learning and attention in continuous states",
    interval_ms: int = 160,
) -> FuncAnimation:
    """Animate Chapter 8 hidden-state, parameter, precision, and VFE convergence."""
    mus = np.asarray(mus, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    zetas = np.asarray(zetas, dtype=float)
    fes = np.asarray(free_energies, dtype=float)
    if not (mus.ndim == thetas.ndim == zetas.ndim == fes.ndim == 1):
        raise ValueError("mus, thetas, zetas, and free_energies must be 1-D arrays")
    if not (mus.shape == thetas.shape == zetas.shape == fes.shape):
        raise ValueError("mus, thetas, zetas, and free_energies must share one shape")
    n = mus.shape[0]
    if n < 2:
        raise ValueError("at least two frames are required")
    truth = None if truth_x is None else np.asarray(truth_x, dtype=float)
    if truth is not None and truth.shape != mus.shape:
        raise ValueError("truth_x must match mus shape")

    t = np.arange(n)
    variances = np.exp(-zetas)
    fig, (ax_state, ax_terms) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                             constrained_layout=True)

    ax_state.set_xlim(0, n - 1)
    ymin = min(float(mus.min()), float(thetas.min()), float(zetas.min()))
    ymax = max(float(mus.max()), float(thetas.max()), float(zetas.max()))
    if truth is not None:
        ymin = min(ymin, float(truth.min()))
        ymax = max(ymax, float(truth.max()))
    if truth_theta is not None:
        ymin = min(ymin, float(truth_theta))
        ymax = max(ymax, float(truth_theta))
    pad = 0.08 * (ymax - ymin + 1e-9)
    ax_state.set_ylim(ymin - pad, ymax + pad)
    ax_state.set_xlabel("frame")
    ax_state.set_ylabel("mean / log precision")
    ax_state.set_title("state, learning, and attention")
    ax_state.grid(alpha=0.3)
    mu_line, = ax_state.plot([], [], color=COLORS["posterior"], label=r"$\mu_x$")
    theta_line, = ax_state.plot([], [], color=COLORS["likelihood"], label=r"$\mu_\theta$")
    zeta_line, = ax_state.plot([], [], color=COLORS["neutral"], ls=":", label=r"$\mu_\zeta$")
    if truth is not None:
        ax_state.plot(t, truth, color=COLORS["truth"], lw=1.4, alpha=0.55,
                      label=r"true $x^*$")
    if truth_theta is not None:
        ax_state.axhline(truth_theta, color=COLORS["truth"], ls="--", lw=1.4,
                         label=r"true $\theta$")
    ax_state.legend(loc="best", fontsize=9)

    ax_terms.set_xlim(0, n - 1)
    term_max = max(float(fes.max()), float(variances.max()))
    ax_terms.set_ylim(0.0, term_max * 1.1 + 1e-9)
    ax_terms.set_xlabel("frame")
    ax_terms.set_ylabel("nats / variance")
    ax_terms.set_title("objective and learned variance")
    ax_terms.grid(alpha=0.3)
    f_line, = ax_terms.plot([], [], color=COLORS["data"], label=r"$\mathcal{F}$")
    var_line, = ax_terms.plot([], [], color=COLORS["state"], label=r"$s_x^2=e^{-\zeta}$")
    ax_terms.legend(loc="best", fontsize=9)
    stat = annotate_stat_box(ax_terms, "", loc="upper right")

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        sl = slice(0, k + 1)
        mu_line.set_data(t[sl], mus[sl])
        theta_line.set_data(t[sl], thetas[sl])
        zeta_line.set_data(t[sl], zetas[sl])
        f_line.set_data(t[sl], fes[sl])
        var_line.set_data(t[sl], variances[sl])
        stat.set_text(
            f"frame {k}\n"
            f"mu_x={mus[k]:.2f}\n"
            f"theta={thetas[k]:.2f}\n"
            f"var={variances[k]:.3f}"
        )
        return [mu_line, theta_line, zeta_line, f_line, var_line, stat]

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=n, interval=interval_ms,
                         blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


# ---------------------------------------------------------------------------
# Chapter 10 — learning animations (Dirichlet convergence + precision sweep)
# ---------------------------------------------------------------------------


def animate_parameter_learning(
    history: np.ndarray,
    confidence: np.ndarray,
    *,
    truth: Optional[np.ndarray] = None,
    symbol: str = "A",
    title: str = "Learning a POMDP array",
    interval_ms: int = 220,
    stride: int = 1,
) -> FuncAnimation:
    r"""Animate Dirichlet parameter learning over trials (Chapter 10 §10.1, Figs 10.1.3/10.1.4).

    The animated counterpart of
    :func:`~active_inference.visualizations.unified.plot_parameter_learning`: two synchronized
    panels grow trial-by-trial — each learned matrix entry converging on its true value
    (left, with target dots) and the matching Dirichlet concentration parameters
    ("confidence") accumulating (right). Duck-typed on plain arrays so it animates either the
    likelihood ``A`` or the transition ``B``.

    Parameters
    ----------
    history : ndarray, shape ``(n_trials+1, ...)``
        Per-trial expected probability arrays.
    confidence : ndarray, shape ``(n_trials+1, ...)``
        Per-trial raw concentration parameters (pseudocounts).
    truth : ndarray, optional
        True array (same trailing shape) — drawn as target dots.
    symbol : str
        Array name for labels (``"A"`` or ``"B"``).
    interval_ms, stride :
        Playback speed and frame subsampling (final frame always shown).
    """
    history = np.asarray(history, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if history.ndim < 2:
        raise TypeError("history must be (n_trials+1, ...) with at least one trailing axis")
    T = history.shape[0]
    trials = np.arange(T)
    flat = history.reshape(T, -1)
    conf_flat = confidence.reshape(T, -1)
    n_entries = flat.shape[1]
    colors = layer_colors(n_entries)
    truth_flat = None if truth is None else np.asarray(truth, dtype=float).ravel()
    idx = list(np.ndindex(history.shape[1:]))
    frames = _frame_indices(T, stride)

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    p_lines, c_lines = [], []
    for k in range(n_entries):
        lbl = rf"${symbol}_{{{idx[k][0]}{idx[k][1]}}}$" if len(idx[k]) == 2 else f"{symbol}[{k}]"
        (lp,) = ax_p.plot([], [], color=colors[k], lw=2.4, label=lbl)
        p_lines.append(lp)
        (lc,) = ax_c.plot([], [], color=colors[k], lw=2.4)
        c_lines.append(lc)
        if truth_flat is not None:
            ax_p.plot(T - 1, truth_flat[k], "o", color=colors[k], ms=9,
                      markeredgecolor="black", zorder=5)

    ax_p.set_xlim(-0.3, T - 0.7)
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_xlabel("trial")
    ax_p.set_ylabel("probability")
    ax_p.set_title(f"learned {symbol} entries → truth (dots)")
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc="center right", fontsize=9)

    ax_c.set_xlim(-0.3, T - 0.7)
    ax_c.set_ylim(0.0, float(conf_flat.max()) * 1.08 + 1e-9)
    ax_c.set_xlabel("trial")
    ax_c.set_ylabel("concentration (pseudocount)")
    ax_c.set_title("confidence (Dirichlet pseudocounts) grows")
    ax_c.grid(alpha=0.3)
    stat = annotate_stat_box(ax_p, "", loc="upper left")

    artists = p_lines + c_lines + [stat]

    def init():
        """Initialize animation artists before the first rendered frame."""
        for ln in p_lines + c_lines:
            ln.set_data([], [])
        stat.set_text("")
        return artists

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        sl = slice(0, k + 1)
        for e in range(n_entries):
            p_lines[e].set_data(trials[sl], flat[sl, e])
            c_lines[e].set_data(trials[sl], conf_flat[sl, e])
        err = ("" if truth_flat is None
               else f"\nmax |{symbol}−{symbol}*| = {np.max(np.abs(flat[k] - truth_flat)):.3f}")
        stat.set_text(f"trial {k}\nΣ counts = {conf_flat[k].sum():.0f}{err}")
        return artists

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=interval_ms, blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


def animate_policy_precision(
    G: np.ndarray,
    gammas: np.ndarray,
    *,
    E: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
    title: str = "Policy precision sweep",
    interval_ms: int = 90,
) -> FuncAnimation:
    r"""Animate the policy posterior as precision ``γ`` sweeps (Chapter 10 §10.2, Fig 10.2.2).

    Bars for ``Q(π) = σ(log E − F − γ G)`` (book Eq. 22) redraw as ``γ`` ramps up: at ``γ = 0``
    the distribution is the (uniform or habit-shaped) prior, and as ``γ`` grows it concentrates
    on the lowest-EFE policy. A companion marker traces the swept ``γ`` value, and a live box
    reports the current ``γ`` and the dominant policy.

    Parameters
    ----------
    G : ndarray, shape ``(P,)``
        Expected free energy per policy.
    gammas : ndarray, shape ``(K,)``
        Precision values to sweep (animation frames).
    E, F : ndarray, optional
        Baseline habit prior and policy-dependent VFE (see ``policy_posterior_full``).
    """
    from ..core.pomdp import policy_posterior_full

    G = np.asarray(G, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    P = G.shape[0]
    posts = np.array([policy_posterior_full(G, F=F, E=E, gamma=float(g)) for g in gammas])
    colors = layer_colors(P)

    fig, (ax_bar, ax_g) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    x = np.arange(P)
    bars = ax_bar.bar(x, posts[0], color=colors, alpha=0.9)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([rf"$\pi^{{({k})}}$" for k in range(P)])
    ax_bar.set_xlabel("policy")
    ax_bar.set_ylabel(r"$Q(\pi)$")
    ax_bar.set_title("policy posterior concentrates as γ rises")
    ax_bar.grid(alpha=0.3, axis="y")
    stat = annotate_stat_box(ax_bar, "", loc="upper right")

    ax_g.plot(gammas, gammas, color=COLORS["neutral"], lw=1.4, ls="--")
    (gmark,) = ax_g.plot([], [], "o", color=COLORS["posterior"], ms=12)
    ax_g.set_xlim(float(gammas.min()), float(gammas.max()) + 1e-9)
    ax_g.set_ylim(float(gammas.min()), float(gammas.max()) + 1e-9)
    ax_g.set_xlabel(r"swept $\gamma$")
    ax_g.set_ylabel(r"current $\gamma$")
    ax_g.set_title("precision (confidence) ramp")
    ax_g.grid(alpha=0.3)

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        for rect, h in zip(bars, posts[k]):
            rect.set_height(h)
        gmark.set_data([gammas[k]], [gammas[k]])
        stat.set_text(f"γ = {gammas[k]:.2f}\ntop policy: π^({int(np.argmax(posts[k]))})")
        return list(bars) + [gmark, stat]

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=len(gammas),
                         interval=interval_ms, blit=False, repeat_delay=1200)
    anim._fig = fig
    return anim


def animate_two_armed_bandit(
    result,
    *,
    title: str = "Two-armed bandit (factorial active inference)",
    interval_ms: int = 350,
) -> FuncAnimation:
    r"""Animate a two-armed bandit run (Chapter 10 §10.3, Figs 10.3.6/10.3.7 in motion).

    Two synchronized panels evolve step-by-step from a ``TwoArmedBanditResult``: the agent's
    **context belief** (which machine is better) converging on the truth, and the **policy
    posterior** bars over the four choice actions, with a live stat box reporting the running
    win/hint counts. Shows the explore-then-exploit trajectory unfold.
    """
    ctx = np.asarray(result.context_belief, dtype=float)
    pol = np.asarray(result.policy_posterior, dtype=float)
    rew = np.asarray(result.reward_obs, dtype=int)
    n = pol.shape[0]
    truth = int(result.true_context)
    actions = ["start", "hint", "left", "right"]

    fig, (ax_ctx, ax_pol) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    (l0,) = ax_ctx.plot([], [], color=COLORS["prior"], lw=2.6, label="L-better")
    (l1,) = ax_ctx.plot([], [], color=COLORS["likelihood"], lw=2.6, label="R-better")
    ax_ctx.set_xlim(0, ctx.shape[0] - 1)
    ax_ctx.set_ylim(-0.05, 1.05)
    ax_ctx.set_xlabel("time")
    ax_ctx.set_ylabel("probability")
    ax_ctx.set_title(f"context belief (true: {'right' if truth else 'left'}-better)")
    ax_ctx.grid(alpha=0.3)
    ax_ctx.legend(loc="center right", fontsize=9)

    colors = layer_colors(pol.shape[1])
    bars = ax_pol.bar(actions, pol[0], color=colors, alpha=0.9)
    ax_pol.set_ylim(0.0, 1.05)
    ax_pol.set_ylabel(r"$Q(\pi)$")
    ax_pol.set_title("policy posterior")
    ax_pol.grid(alpha=0.3, axis="y")
    stat = annotate_stat_box(ax_pol, "", loc="upper left")

    def update(k: int):
        """Update interactive or animated artists for the current state."""
        l0.set_data(np.arange(k + 2), ctx[: k + 2, 0])
        l1.set_data(np.arange(k + 2), ctx[: k + 2, 1])
        for rect, h in zip(bars, pol[k]):
            rect.set_height(h)
        wins = int(np.sum(rew[: k + 1] == 2))
        hints = int(np.sum(np.asarray(result.choices)[: k + 1] == 1))
        stat.set_text(f"step {k}\nwins {wins}\nhints {hints}")
        return [l0, l1, stat, *bars]

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=n, interval=interval_ms,
                         blit=False, repeat_delay=1500)
    anim._fig = fig
    return anim


def animate_multivariate_active_inference(
    result,
    *,
    preference: np.ndarray,
    exogenous: Optional[np.ndarray] = None,
    dt: float = 1.0,
    frame_stride: int = 25,
    title: str = "Multivariate active generalized filtering",
    interval_ms: int = 80,
) -> FuncAnimation:
    """Animate Chapter 7 §7.5 vector action-perception dynamics.

    The animation shows the 2-D external state moving from the exogenous attractor
    toward the preferred state, alongside live action, sensory error, and free-energy
    traces. Raw time-series are attached to the animation for NPZ+JSON export.
    """
    xs = np.asarray(result.xs, dtype=float)
    mus = np.asarray(result.mus, dtype=float)[:, 0, :]
    actions = np.asarray(result.actions, dtype=float)
    eps = np.linalg.norm(np.asarray(result.eps_y, dtype=float)[:, 0, :], axis=1)
    fes = np.asarray(result.free_energies, dtype=float)
    preference = np.asarray(preference, dtype=float)
    frames = np.arange(0, xs.shape[0], max(1, int(frame_stride)))
    if frames[-1] != xs.shape[0] - 1:
        frames = np.append(frames, xs.shape[0] - 1)
    t = np.arange(xs.shape[0]) * float(dt)

    fig, (ax_path, ax_traces) = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    pad = 1.5
    all_xy = np.vstack([xs, mus, preference[None, :]])
    if exogenous is not None:
        all_xy = np.vstack([all_xy, np.asarray(exogenous, dtype=float)[None, :]])
    ax_path.set_xlim(float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad))
    ax_path.set_ylim(float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad))
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel(r"$x_0$")
    ax_path.set_ylabel(r"$x_1$")
    ax_path.set_title("external state and belief path")
    ax_path.grid(alpha=0.3)
    ax_path.scatter(preference[0], preference[1], marker="*", s=180,
                    color=COLORS["likelihood"], edgecolor="black", linewidth=0.8,
                    label="preference")
    if exogenous is not None:
        ex = np.asarray(exogenous, dtype=float)
        ax_path.scatter(ex[0], ex[1], marker="x", s=110, color=COLORS["neutral"],
                        label="exogenous")
    (truth_path,) = ax_path.plot([], [], color=COLORS["truth"], lw=2.0, label=r"$x^*$")
    (belief_path,) = ax_path.plot([], [], color=COLORS["posterior"], ls="--", lw=1.5,
                                  label=r"$\mu_x$")
    truth_dot = ax_path.scatter([], [], s=95, color=COLORS["truth"], edgecolor="white", zorder=4)
    belief_dot = ax_path.scatter([], [], s=80, color=COLORS["posterior"], edgecolor="white", zorder=4)
    ax_path.legend(loc="best", fontsize=9)

    (action_line,) = ax_traces.plot([], [], color=COLORS["state"], lw=2.0,
                                    label=r"$||a||$")
    (eps_line,) = ax_traces.plot([], [], color=COLORS["sensory"], lw=2.0,
                                 label=r"$||\epsilon_y||$")
    (fe_line,) = ax_traces.plot([], [], color=COLORS["data"], lw=2.0,
                                label=r"$\mathcal{F}$")
    ax_traces.axvline(result.action_start * float(dt), color="black", ls="--", lw=1.2,
                      label="action on")
    ax_traces.set_xlim(float(t[0]), float(t[-1]))
    ymax = max(float(np.linalg.norm(actions, axis=1).max()), float(eps.max()), float(fes.max()))
    ax_traces.set_ylim(0.0, ymax * 1.1 + 1e-9)
    ax_traces.set_xlabel("time")
    ax_traces.set_ylabel("magnitude")
    ax_traces.set_title("action, sensory error, and VFE")
    ax_traces.grid(alpha=0.3)
    ax_traces.legend(loc="upper right", fontsize=9)
    stat = annotate_stat_box(ax_traces, "", loc="center right")

    def update(frame_index: int):
        """Update animated artists for a saved frame."""
        k = int(frames[frame_index])
        truth_path.set_data(xs[: k + 1, 0], xs[: k + 1, 1])
        belief_path.set_data(mus[: k + 1, 0], mus[: k + 1, 1])
        truth_dot.set_offsets(xs[k][None, :])
        belief_dot.set_offsets(mus[k][None, :])
        action_line.set_data(t[: k + 1], np.linalg.norm(actions[: k + 1], axis=1))
        eps_line.set_data(t[: k + 1], eps[: k + 1])
        fe_line.set_data(t[: k + 1], fes[: k + 1])
        stat.set_text(
            f"t={t[k]:.2f}\n"
            f"pref err={np.linalg.norm(xs[k] - preference):.2f}\n"
            f"|a|={np.linalg.norm(actions[k]):.2f}"
        )
        return truth_path, belief_path, truth_dot, belief_dot, action_line, eps_line, fe_line, stat

    fig.suptitle(title)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms,
                         blit=False, repeat_delay=1200)
    anim._fig = fig
    anim._raw_data = {
        "xs": xs,
        "mus_order0": mus,
        "actions": actions,
        "eps_y_norm": eps,
        "free_energies": fes,
        "preference": preference,
        "frames": frames,
        "dt": np.array([dt], dtype=float),
    }
    return anim
