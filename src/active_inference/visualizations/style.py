"""Shared visual style for figures and animations.

This module collects the small style decisions that were previously
duplicated across ``plotting.py``, ``diagnostics.py``, and
``animations.py``: stat-box bbox kwargs, the brand color palette, the
default ellipse drawing settings, and the global font / line / grid
settings used across every chapter orchestrator. Centralising them
prevents drift and makes it easy to re-skin the whole package.

The default font sizes are intentionally larger than matplotlib's
defaults — chapter figures are intended to be projected, embedded in
slides, and skim-readable in print, not just rendered at native
resolution on a laptop. Override individual entries via
``set_default_style(...)`` if you need tighter typography.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional

import matplotlib as mpl

# Palette — keep in sync with `tab10` ordering used throughout chapter scripts.
COLORS: Mapping[str, str] = {
    "prior":     "#1f77b4",   # blue
    "likelihood": "#d62728",  # red
    "posterior": "#2ca02c",   # green
    "data":      "#000000",   # black
    "truth":     "#ff0000",   # bright red
    "neutral":   "#888888",
}


def stat_box_bbox(
    *,
    facecolor: str = "white",
    edgecolor: str = "black",
    pad: float = 0.25,
    alpha: float = 0.85,
) -> dict:
    """The bbox style used for all in-figure stat-readout text boxes.

    Returns a dict suitable for `matplotlib.text.Text(bbox=...)`. Keep
    every stat-box readout in the package on this single style so they
    look like siblings.
    """
    return dict(
        boxstyle=f"round,pad={pad}",
        fc=facecolor,
        ec=edgecolor,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Global defaults — applied at import time so every figure picks them up.
# ---------------------------------------------------------------------------


DEFAULT_RC: Mapping[str, object] = {
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "figure.titlesize":  14,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.0,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "figure.constrained_layout.use": True,
}


def set_default_style(overrides: Optional[Mapping[str, object]] = None) -> None:
    """Apply the package's default rcParams (large fonts, grid on, etc.).

    Idempotent. Pass ``overrides`` to tweak specific entries — useful when
    a script wants oversized fonts for a slide deck or smaller fonts for
    print embedding.
    """
    rc = dict(DEFAULT_RC)
    if overrides:
        rc.update(overrides)
    mpl.rcParams.update(rc)


@contextmanager
def figure_style(overrides: Optional[Mapping[str, object]] = None):
    """Temporarily apply package defaults inside a ``with`` block.

    Restores the previous rcParams on exit so chapter scripts can opt
    into different styles per figure without leaking state.
    """
    saved = mpl.rcParams.copy()
    try:
        set_default_style(overrides)
        yield
    finally:
        mpl.rcParams.update(saved)


# Apply once at import. Chapter scripts can re-call set_default_style
# with overrides; this gives every figure a sensible baseline.
set_default_style()


def annotate_stat_box(
    ax,
    text: str,
    *,
    loc: str = "upper left",
    fontsize: int = 10,
    **bbox_overrides,
):
    """Place a stat-box at a named corner of the axis.

    ``loc`` accepts the same names as `Axes.legend(loc=...)` (only the
    four corners and "center" — anything else falls back to top-left).
    """
    corners = {
        "upper left":  (0.02, 0.97, "left",  "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left":  (0.02, 0.03, "left",  "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "center":      (0.50, 0.50, "center", "center"),
    }
    x, y, ha, va = corners.get(loc, corners["upper left"])
    bbox = stat_box_bbox(**bbox_overrides)
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, ha=ha, va=va,
        bbox=bbox,
    )
