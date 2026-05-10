"""Visualizations that surface *statistics* — CDFs, QQ plots, calibration,
posterior-predictive checks, coverage curves, KL traces.

These complement ``plotting.py`` (which renders generic curves and ellipses)
by communicating concrete numerical diagnostics readers can use to evaluate
an inference run, not just admire its shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..core.diagnostics import (
    CalibrationCurve,
    PosteriorPredictiveCheck,
)
from .plotting import save_or_show
from .style import stat_box_bbox


# ---------------------------------------------------------------------------
# CDF / QQ plots
# ---------------------------------------------------------------------------


def plot_cdf_comparison(
    x_grid: np.ndarray,
    cdfs: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    truth: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (7, 4.5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Overlay several CDFs on a shared x-axis.

    Useful for showing a posterior CDF tightening as N grows or for
    comparing prior/posterior CDFs side by side.
    """
    if len(cdfs) != len(labels):
        raise ValueError("cdfs and labels must have the same length")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    cmap = plt.get_cmap("viridis")
    for i, (cdf, label) in enumerate(zip(cdfs, labels)):
        if cdf.shape != x_grid.shape:
            raise ValueError(f"cdfs[{i}] must share shape with x_grid")
        ax.plot(x_grid, cdf, color=cmap(i / max(len(cdfs) - 1, 1)),
                lw=2, label=label)
    if truth is not None:
        ax.axvline(truth, color="red", ls=":", lw=1.5,
                   label=f"x* = {truth:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_qq(
    samples: np.ndarray,
    *,
    distribution: str = "normal",
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (5.5, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Quantile–quantile plot vs the standard normal (or another reference).

    A diagonal line indicates perfect agreement; deviations highlight
    skew / heavy tails / location bias in the empirical distribution.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    if samples.size < 2:
        raise ValueError("need at least two samples for a QQ plot")
    if distribution != "normal":
        raise NotImplementedError(f"distribution={distribution!r} not supported")

    from scipy.special import erfinv
    n = samples.size
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.sqrt(2.0) * erfinv(2.0 * quantiles - 1.0)
    empirical = np.sort(samples)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.scatter(theoretical, empirical, s=10, color="#1f77b4", alpha=0.7)
    lim_lo = min(theoretical.min(), empirical.min())
    lim_hi = max(theoretical.max(), empirical.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="red", ls="--",
            lw=1.5, label="y = x")
    ax.set_xlabel("theoretical quantile (N(0, 1))")
    ax.set_ylabel("empirical quantile")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


# ---------------------------------------------------------------------------
# Calibration / coverage
# ---------------------------------------------------------------------------


def plot_calibration(
    curve: CalibrationCurve,
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (5.5, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot empirical vs nominal coverage (the "reliability diagram")."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot([0, 1], [0, 1], color="red", ls="--", lw=1, label="perfect")
    ax.plot(curve.nominal, curve.empirical, "o-", color="#1f77b4",
            lw=2, ms=6, label=f"empirical (T = {curve.n_trials})")
    ax.fill_between(curve.nominal, curve.nominal, curve.empirical,
                    alpha=0.18, color="#1f77b4")
    ece = curve.calibration_error()
    ax.text(0.02, 0.93, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=10, bbox=stat_box_bbox())
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_coverage_curve(
    n_axis: np.ndarray,
    coverages: np.ndarray,
    *,
    nominal: float = 0.95,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (7, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot empirical coverage of a fixed-mass interval vs sample size."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.axhline(nominal, color="red", ls="--", lw=1,
               label=f"nominal = {nominal:.2f}")
    ax.plot(n_axis, coverages, "o-", color="#1f77b4", lw=2, ms=5,
            label="empirical")
    ax.set_xlabel("samples assimilated N")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------


def plot_posterior_predictive_check(
    check: PosteriorPredictiveCheck,
    *,
    label: str = "test statistic",
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (7, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Histogram of replicated test statistics with the observed value marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.hist(check.replicated, bins=40, alpha=0.7, color="#1f77b4",
            edgecolor="white", label=f"replicates (M = {check.replicated.size})")
    ax.axvline(check.observed, color="red", lw=2.5,
               label=f"observed = {check.observed:.3f}")
    ax.text(0.02, 0.93,
            f"two-sided p = {check.p_value:.3f}",
            transform=ax.transAxes, fontsize=10, bbox=stat_box_bbox())
    ax.set_xlabel(label)
    ax.set_ylabel("frequency")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


# ---------------------------------------------------------------------------
# Trace / convergence diagnostics
# ---------------------------------------------------------------------------


def plot_kl_trace(
    n_axis: np.ndarray,
    kl_values: np.ndarray,
    *,
    title: str = "KL(posterior || prior) vs N",
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (7, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot how much the data has moved the posterior away from the prior."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(n_axis, kl_values, "o-", color="#2ca02c", lw=2, ms=4)
    ax.set_xlabel("samples assimilated N")
    ax.set_ylabel("KL[posterior || prior]  (nats)")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    save_or_show(fig, save_path, show=show)
    return fig


def plot_running_statistics(
    n_axis: np.ndarray,
    running_mean: np.ndarray,
    running_std: np.ndarray,
    *,
    truth: Optional[float] = None,
    title: str = "Running posterior summaries",
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Two-panel running posterior mean and standard deviation."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                             sharex=True)
    axes[0].plot(n_axis, running_mean, "-", color="#2ca02c", lw=2)
    if truth is not None:
        axes[0].axhline(truth, color="red", ls=":", lw=1.5, label="truth")
        axes[0].legend()
    axes[0].set_ylabel("posterior mean")
    axes[0].set_xlabel("samples assimilated N")
    axes[0].grid(alpha=0.3)

    axes[1].plot(n_axis, running_std, "-", color="#1f77b4", lw=2)
    axes[1].set_ylabel("posterior std")
    axes[1].set_xlabel("samples assimilated N")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    save_or_show(fig, save_path, show=show)
    return fig


# ---------------------------------------------------------------------------
# Score traces (log-score, CRPS)
# ---------------------------------------------------------------------------


def plot_score_trace(
    n_axis: np.ndarray,
    log_scores: np.ndarray,
    crps_scores: Optional[np.ndarray] = None,
    *,
    title: str = "Predictive scores vs N",
    save_path: Optional[Path | str] = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Side-by-side cumulative log-score and CRPS as the inference proceeds."""
    if crps_scores is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                                 sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] * 0.6, figsize[1]),
                               constrained_layout=True)
        axes = [ax]

    axes[0].plot(n_axis, log_scores, "-", color="#2ca02c", lw=2,
                 label="log score (higher better)")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("cumulative log score")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    if crps_scores is not None:
        axes[1].plot(n_axis, crps_scores, "-", color="#d62728", lw=2,
                     label="CRPS (lower better)")
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("mean CRPS")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    fig.suptitle(title, fontsize=12)
    save_or_show(fig, save_path, show=show)
    return fig
