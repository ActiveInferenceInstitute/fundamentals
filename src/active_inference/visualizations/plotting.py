"""Static matplotlib helpers used by the chapter orchestrators.

Every function here accepts an optional ``ax`` so that callers can compose
multi-panel figures, and an optional ``save_path`` so that batch scripts can
render to disk without opening a window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..core.inference import InferenceResult
from ..utils import save_figure_data
from .style import COLORS, stat_box_bbox


_SERIES_COLORS = (
    COLORS["prior"],
    COLORS["likelihood"],
    COLORS["posterior"],
    COLORS["state"],
    COLORS["sensory"],
    COLORS["truth"],
    COLORS["data"],
    COLORS["neutral"],
)


def save_or_show(
    fig: plt.Figure,
    save_path: Optional[Path | str] = None,
    *,
    show: bool = False,
    dpi: int = 150,
) -> Optional[Path]:
    """Either save ``fig`` to ``save_path`` or call ``plt.show()``.

    Returns the resolved path on save, otherwise ``None``.
    """
    out: Optional[Path] = None
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        save_figure_data(fig, out, metadata={"dpi": dpi})
    if show:
        plt.show()
    if save_path is not None and not show:
        plt.close(fig)
    return out


def plot_prior_likelihood_posterior(
    result: InferenceResult,
    *,
    title: Optional[str] = None,
    truth: Optional[float] = None,
    annotate_stats: bool = True,
    credible_mass: float = 0.95,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 3.4),
) -> plt.Figure:
    """Three-panel figure showing prior, likelihood, and posterior densities.

    When ``annotate_stats`` is true, the posterior panel is overlaid with the
    credible interval, mean, mode, and a stat-box reporting KL(post || prior),
    differential entropy, and (if ``truth`` is supplied) the bias.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].plot(result.x_grid, result.prior, color=COLORS["prior"], lw=2)
    axes[0].fill_between(result.x_grid, result.prior, alpha=0.25, color=COLORS["prior"])
    axes[0].set_title("Prior  p(x)")
    axes[0].set_ylabel("density")

    axes[1].plot(result.x_grid, result.likelihood, color=COLORS["likelihood"], lw=2)
    axes[1].fill_between(
        result.x_grid, result.likelihood, alpha=0.25, color=COLORS["likelihood"]
    )
    axes[1].set_title("Likelihood  p(y | x)  (unnormalized)")
    axes[1].set_ylabel("credibility")

    axes[2].plot(result.x_grid, result.posterior, color=COLORS["posterior"], lw=2)
    axes[2].fill_between(
        result.x_grid, result.posterior, alpha=0.25, color=COLORS["posterior"]
    )

    if annotate_stats:
        lo, hi = result.credible_interval(credible_mass)
        ci_mask = (result.x_grid >= lo) & (result.x_grid <= hi)
        axes[2].fill_between(result.x_grid[ci_mask],
                             result.posterior[ci_mask],
                             alpha=0.45, color=COLORS["posterior"],
                             label=f"{int(credible_mass * 100)}% CI")
        axes[2].axvline(result.posterior_mean, color=COLORS["prior"], lw=1, ls="-",
                        label=f"mean = {result.posterior_mean:.3f}")

    axes[2].axvline(result.posterior_mode, color=COLORS["data"], lw=1, ls="--",
                    label=f"mode = {result.posterior_mode:.3f}")
    if truth is not None:
        axes[2].axvline(truth, color=COLORS["truth"], lw=1, ls=":",
                        label=f"x* = {truth:.3f}")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].set_title("Posterior  p(x | y)")
    axes[2].set_ylabel("density")

    if annotate_stats:
        kl = result.kl_from_prior()
        h = result.entropy()
        text = (
            f"H[post] = {h:.3f}\n"
            f"KL[post||prior] = {kl:.3f}\n"
            f"log p(y) = {result.log_evidence:.3f}"
        )
        if truth is not None:
            text = f"bias = {result.posterior_mean - truth:+.3f}\n" + text
        axes[2].text(0.02, 0.97, text, transform=axes[2].transAxes,
                     fontsize=8, va="top", ha="left",
                     bbox=stat_box_bbox(pad=0.3))

    for ax in axes:
        ax.set_xlabel("hidden state x")
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_generating_function(
    x_grid: np.ndarray,
    f_x: np.ndarray,
    samples_x: Optional[np.ndarray] = None,
    samples_y: Optional[np.ndarray] = None,
    *,
    title: Optional[str] = None,
    xlabel: str = "hidden state x",
    ylabel: str = "observation y",
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (5.5, 4),
) -> plt.Figure:
    """Plot ``y = g(x)`` with optional scatter of noisy samples.

    When ``samples_x`` and ``samples_y`` are supplied the figure overlays a
    stat-box reporting the residual mean / std / RMSE so readers can see
    how noisy the realized observations are around the noise-free curve.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(x_grid, f_x, color=COLORS["data"], lw=2, label="g(x)")
    if samples_x is not None and samples_y is not None:
        ax.scatter(samples_x, samples_y, s=22, color=COLORS["state"],
                   alpha=0.7, label="samples", edgecolor="white", lw=0.4)
        # Residual statistics: project each sample to the curve via interp.
        f_at_samples = np.interp(np.asarray(samples_x), x_grid, f_x)
        residuals = np.asarray(samples_y) - f_at_samples
        if residuals.size == 0:
            residual_summary = "N = 0\nresidual statistics unavailable"
        else:
            residual_std = (
                f"{float(residuals.std(ddof=1)):.3f}"
                if residuals.size > 1
                else "n/a (N < 2)"
            )
            residual_summary = (
                f"N = {residuals.size}\n"
                f"residual mean = {float(residuals.mean()):+.3f}\n"
                f"residual std  = {residual_std}\n"
                f"RMSE          = {float(np.sqrt((residuals ** 2).mean())):.3f}"
            )
        ax.text(
            0.02, 0.97,
            residual_summary,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=stat_box_bbox(),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_likelihood_ridge(
    x_grid: np.ndarray,
    likelihoods: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    *,
    title: Optional[str] = None,
    show_cumulative: bool = True,
    truth: Optional[float] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 5.5),
) -> plt.Figure:
    """Vertically stacked ridge plot of per-sample likelihoods.

    When ``show_cumulative`` is true a thin top panel overlays the
    *joint* (un-normalized) likelihood — the product of every per-sample
    likelihood — so readers can see the inversion concentrating onto the
    true state as samples accumulate.
    """
    n = len(likelihoods)
    if show_cumulative:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 3]},
            sharex=True,
        )
        ax_top, ax = axes
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_top = None

    offsets = np.linspace(0, n - 1, n)
    for i, lik in enumerate(likelihoods):
        scaled = lik / np.max(lik) if np.max(lik) > 0 else lik
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.fill_between(x_grid, offsets[i], offsets[i] + scaled,
                        color=color, alpha=0.7, lw=0.5)
        ax.plot(x_grid, offsets[i] + scaled, color=color, lw=1)
    if labels is None:
        labels = [f"sample {i + 1}" for i in range(n)]
    ax.set_yticks(offsets + 0.5)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("hidden state x")
    ax.set_ylabel("per-sample likelihood")
    ax.grid(alpha=0.3, axis="x")
    if truth is not None:
        ax.axvline(truth, color=COLORS["truth"], ls=":", lw=1.5, alpha=0.7)

    if ax_top is not None:
        # Cumulative likelihood = product of per-sample likelihoods,
        # computed in log-space then peak-normalized for visual clarity.
        log_cum = np.zeros_like(x_grid, dtype=float)
        for lik in likelihoods:
            with np.errstate(divide="ignore"):
                log_cum = log_cum + np.log(np.maximum(lik, 1e-300))
        log_cum -= np.max(log_cum)
        cum = np.exp(log_cum)
        cum = cum / np.trapezoid(cum, x_grid)
        ax_top.plot(x_grid, cum, color=COLORS["posterior"], lw=2)
        ax_top.fill_between(x_grid, cum, alpha=0.3, color=COLORS["posterior"])
        peak = float(x_grid[int(np.argmax(cum))])
        ax_top.axvline(peak, color=COLORS["data"], ls="--", lw=1,
                       label=f"argmax = {peak:.3f}")
        if truth is not None:
            ax_top.axvline(truth, color=COLORS["truth"], ls=":", lw=1.5,
                           label=f"x* = {truth:.3f}")
        ax_top.set_ylabel("joint p(y_{1:N}|x)")
        ax_top.legend(loc="upper right", fontsize=9)
        ax_top.grid(alpha=0.3, axis="x")

    if title:
        fig.suptitle(title, fontsize=13)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_joint_heatmap(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    joint: np.ndarray,
    *,
    title: Optional[str] = None,
    show_marginals: bool = True,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (7.5, 6.5),
) -> plt.Figure:
    """Heatmap of the joint density ``p(x, y)`` with optional marginals.

    ``joint`` is expected with shape ``(len(y_grid), len(x_grid))``. When
    ``show_marginals`` is true the figure adds the marginal ``p(x)`` and
    ``p(y)`` panels on the top and right edges plus a stat-box reporting
    ``E[x]``, ``E[y]``, ``cov(x, y)``, and ``corr(x, y)`` computed by
    trapezoid integration over the joint.
    """
    if not show_marginals:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
        im = ax.imshow(joint, extent=extent, origin="lower", aspect="auto",
                       cmap="magma")
        fig.colorbar(im, ax=ax, label="p(x, y)")
        ax.set_xlabel("hidden state x")
        ax.set_ylabel("observation y")
        if title:
            ax.set_title(title)
        save_or_show(fig, save_path, show=show)
        return fig

    # Composed layout: heatmap + top marginal + right marginal.
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
        wspace=0.05, hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
    im = ax_main.imshow(joint, extent=extent, origin="lower", aspect="auto",
                        cmap="magma")
    ax_main.set_xlabel("hidden state x")
    ax_main.set_ylabel("observation y")

    # Marginals via trapezoid integration over the joint.
    p_x = np.trapezoid(joint, y_grid, axis=0)
    p_y = np.trapezoid(joint, x_grid, axis=1)
    ax_top.plot(x_grid, p_x, color=COLORS["prior"], lw=2)
    ax_top.fill_between(x_grid, p_x, alpha=0.3, color=COLORS["prior"])
    ax_top.set_ylabel("p(x)")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(alpha=0.3)
    ax_right.plot(p_y, y_grid, color=COLORS["likelihood"], lw=2)
    ax_right.fill_betweenx(y_grid, 0, p_y, alpha=0.3, color=COLORS["likelihood"])
    ax_right.set_xlabel("p(y)")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(alpha=0.3)

    # Joint moments.
    ex = float(np.trapezoid(x_grid * p_x, x_grid))
    ey = float(np.trapezoid(y_grid * p_y, y_grid))
    var_x = float(np.trapezoid((x_grid - ex) ** 2 * p_x, x_grid))
    var_y = float(np.trapezoid((y_grid - ey) ** 2 * p_y, y_grid))
    # cov(x, y) = ∫∫ (x − E[x])(y − E[y]) p(x, y) dx dy
    cov_grid = np.outer(y_grid - ey, x_grid - ex) * joint
    cov_marg_y = np.trapezoid(cov_grid, x_grid, axis=1)
    cov_xy = float(np.trapezoid(cov_marg_y, y_grid))
    corr_xy = cov_xy / np.sqrt(var_x * var_y) if var_x * var_y > 0 else 0.0

    ax_right.text(
        0.5, 1.02,
        f"E[x] = {ex:.3f}\n"
        f"E[y] = {ey:.3f}\n"
        f"cov  = {cov_xy:+.3f}\n"
        f"corr = {corr_xy:+.3f}",
        transform=ax_right.transAxes, ha="center", va="bottom",
        fontsize=9, bbox=stat_box_bbox(),
    )

    fig.colorbar(im, ax=[ax_main, ax_right], label="p(x, y)",
                 location="right", shrink=0.8, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=13)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_gradient_descent(
    history: np.ndarray,
    losses: np.ndarray,
    *,
    truth: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (11, 4.5),
) -> plt.Figure:
    """Side-by-side loss-vs-iter and iterate-vs-iter view with stat annotations.

    Both panels carry an in-figure stat box: the loss panel reports the
    initial / final loss and the relative drop; the iterate panel reports
    the initial / final estimate, the move from start to finish, and (if
    ``truth`` is supplied) the residual bias against the true value.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].plot(losses, color=COLORS["prior"], lw=2)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)

    if losses.size:
        rel_drop = (
            (losses[0] - losses[-1]) / losses[0] if losses[0] != 0 else np.nan
        )
        axes[0].text(
            0.97, 0.97,
            f"K = {len(losses) - 1}\n"
            f"loss[0] = {losses[0]:.3g}\n"
            f"loss[K] = {losses[-1]:.3g}\n"
            f"Δ rel. = {rel_drop:.2%}",
            transform=axes[0].transAxes,
            ha="right", va="top", fontsize=9, bbox=stat_box_bbox(),
        )

    axes[1].plot(history, color=COLORS["posterior"], lw=2, label="iterate")
    if truth is not None:
        axes[1].axhline(truth, color=COLORS["truth"], ls=":", lw=1.5, label="x*")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("estimate")
    axes[1].set_title("Hidden-state estimate")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    if history.size:
        bias_line = (
            f"\nbias = {history[-1] - truth:+.3g}"
            if truth is not None else ""
        )
        axes[1].text(
            0.03, 0.97,
            f"x[0] = {history[0]:.3g}\n"
            f"x[K] = {history[-1]:.3g}\n"
            f"move = {history[-1] - history[0]:+.3g}{bias_line}",
            transform=axes[1].transAxes,
            ha="left", va="top", fontsize=9, bbox=stat_box_bbox(),
        )

    if title:
        fig.suptitle(title, fontsize=13)
    save_or_show(fig, save_path, show=show)
    return fig


def confidence_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n_std: float = 2.0,
    **kwargs,
) -> Ellipse:
    """Return a confidence ellipse patch for a 2-D Gaussian.

    Standard derivation: the eigenvectors of the covariance give the
    principal axes; the eigenvalues set the half-lengths scaled by ``n_std``.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    return Ellipse(xy=tuple(np.asarray(mean).reshape(-1)),
                   width=width, height=height, angle=angle, **kwargs)


def plot_2d_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    samples: Optional[np.ndarray] = None,
    truth: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (5.5, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualize a 2-D Gaussian via 1- and 2-σ confidence ellipses.

    ``samples`` (shape ``(N, 2)``) and ``truth`` (shape ``(2,)``) are optional
    overlays.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    for n_std, alpha in zip((1, 2), (0.35, 0.15)):
        ax.add_patch(confidence_ellipse(
            mean,
            cov,
            n_std=n_std,
            fc=COLORS["posterior"],
            ec=COLORS["posterior"],
            alpha=alpha, lw=1.5,
        ))
    ax.scatter(*mean, color=COLORS["posterior"], s=40, zorder=5, label="posterior mean")
    if truth is not None:
        ax.scatter(*truth, marker="x", color=COLORS["truth"], s=80, lw=2.0,
                   label=f"true ({truth[0]:.2f}, {truth[1]:.2f})")
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5,
                   color=COLORS["data"], label="samples")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_precision_comparison(
    results: Iterable[tuple[str, InferenceResult]],
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Overlay multiple posteriors (named) for a precision study.

    The posterior panel adds an in-figure stat box reporting, for every
    overlay, the posterior mode and the KL[posterior‖prior] — readers
    can directly compare how far each precision setting moved the belief.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].set_title("Prior")
    axes[1].set_title("Likelihood (unnormalized)")
    axes[2].set_title("Posterior")
    rows = []
    for i, (label, res) in enumerate(results):
        c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        axes[0].plot(res.x_grid, res.prior, color=c, lw=2, label=label)
        axes[1].plot(res.x_grid, res.likelihood, color=c, lw=2, label=label)
        axes[2].plot(res.x_grid, res.posterior, color=c, lw=2, label=label)
        rows.append((label, res.posterior_mode, res.kl_from_prior()))
    for ax in axes:
        ax.set_xlabel("hidden state x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    if rows:
        lines = ["mode    KL[post‖prior]"]
        for label, mode, kl in rows:
            short = label if len(label) <= 22 else label[:19] + "..."
            lines.append(f"{short:<22} {mode:6.3f}  {kl:6.3f}")
        axes[2].text(
            0.02, 0.97, "\n".join(lines),
            transform=axes[2].transAxes,
            ha="left", va="top", fontsize=8,
            family="monospace",
            bbox=stat_box_bbox(),
        )

    if title:
        fig.suptitle(title, fontsize=13)
    save_or_show(fig, save_path, show=show)
    return fig
