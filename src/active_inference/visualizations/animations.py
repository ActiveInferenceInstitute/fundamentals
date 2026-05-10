"""Matplotlib animations for the active-inference workflow.

Each function returns a :class:`matplotlib.animation.FuncAnimation` so the
caller can either ``.save("file.gif", writer="pillow")`` or display it
interactively. We avoid heavier dependencies (ImageMagick, FFmpeg) by sticking
to pillow output, which ships with matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse


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
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
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
        point.set_data([], [])
        trail.set_data([], [])
        loss_line.set_data([], [])
        return point, trail, loss_line

    def update(frame: int):
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
        mean_dot.set_data([], [])
        txt.set_text("")
        return mean_dot, txt

    def update(frame: int):
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
        line_mean.set_data([], [])
        line_std.set_data([], [])
        line_kl.set_data([], [])
        txt.set_text("")
        return line_mean, line_std, line_kl, txt

    def update(frame: int):
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
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
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
        for ln in (line_p, line_l, line_q):
            ln.set_data([], [])
        txt.set_text("")
        return line_p, line_l, line_q, txt

    def update(frame: int):
        # Peak-normalize for visual comparability across frames.
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
        line.set_data([], [])
        txt.set_text("")
        return line, txt

    def update(frame: int):
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
        mean_dot.set_data([], [])
        obs_scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return mean_dot, obs_scatter, txt

    def update(frame: int):
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
        e_scatter.set_offsets(np.empty((0, 2)))
        ll_line.set_data([], [])
        txt.set_text("")
        return e_scatter, im, ll_line, txt

    def update(frame: int):
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
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return line, scatter, txt

    def update(frame: int):
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
        line.set_data([], [])
        return line, im

    def update(frame: int):
        line.set_data(np.arange(frame + 1), log_likelihoods[: frame + 1])
        im.set_data(Theta_history[frame])
        return line, im

    fig.suptitle(title, fontsize=12)
    anim = FuncAnimation(fig, update, frames=K, init_func=init,
                         interval=interval_ms, blit=False)
    anim._fig = fig
    return anim
