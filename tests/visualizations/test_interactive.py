"""Tests for matplotlib-slider interactive figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.patches as mpl_patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.extra_topics import extra_topic_spec  # noqa: E402
from active_inference.visualizations.interactive import (  # noqa: E402
    interactive_bayesian_regression,
    interactive_gradient_descent,
    interactive_inference,
    interactive_inverse_problem,
    interactive_lgs_localization,
    interactive_precision,
    interactive_predictive_coding,
    interactive_topic_demo,
    interactive_variational_free_energy,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close any figures created by slider tests."""
    yield
    plt.close("all")


def _finite_lines(fig: plt.Figure) -> bool:
    """Return whether every plotted line/collection/patch contains finite values.

    Checks ``ax.lines`` (line artists) plus ``ax.collections`` (band fills,
    scatter offsets) and ``ax.patches`` (Ellipse posterior shapes), so
    non-finite band or ellipse data can't slip through unchecked.
    """
    finite = True
    for ax in fig.axes:
        for line in ax.lines:
            if len(line.get_ydata()):
                finite &= bool(np.all(np.isfinite(
                    np.asarray(line.get_ydata(), dtype=float))))
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if offsets is not None and len(offsets):
                finite &= bool(np.all(np.isfinite(
                    np.asarray(offsets, dtype=float))))
        for patch in ax.patches:
            if isinstance(patch, mpl_patches.Ellipse):
                finite &= bool(np.isfinite(patch.center[0])
                               and np.isfinite(patch.center[1]))
    return finite


def _exercise_sliders(fig) -> None:
    """Drive every slider to both endpoints and the midpoint, asserting lines
    stay finite with a live readout at each extreme. This probes the degenerate
    states (variance→min, observation at support edge, precision extremes) that
    per-default single-step nudges never reach."""
    for slider in fig._sliders:  # type: ignore[attr-defined]
        for target in (slider.valmin, slider.valmax,
                       (slider.valmin + slider.valmax) / 2.0):
            slider.set_val(target)
            assert _finite_lines(fig)
    assert any(text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_inference_slider_updates_finite_lines() -> None:
    """The Chapter 2 full interactive updates line data when sliders move."""
    fig = interactive_inference()
    slider = fig._sliders[0]  # type: ignore[attr-defined]
    slider.set_val(slider.val + 0.25)
    assert _finite_lines(fig)
    assert any(text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_precision_slider_updates_finite_lines() -> None:
    """The precision-ratio interactive updates line data when its slider moves."""
    fig = interactive_precision()
    slider = fig._slider  # type: ignore[attr-defined]
    slider.set_val(1.0)
    assert _finite_lines(fig)
    assert any("posterior std" in text.get_text() for ax in fig.axes for text in ax.texts)
    # Endpoint probe: at the extremes of the precision ratio the curves must
    # stay finite and the posterior/std readout must remain present.
    for target in (slider.valmin, slider.valmax):
        slider.set_val(target)
        assert _finite_lines(fig)
        assert any("posterior std" in text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_inverse_problem_bimodal() -> None:
    """Ch1 inverse-problem interactive stays finite and reports the two modes."""
    fig = interactive_inverse_problem()
    assert len(fig._sliders) == 2  # type: ignore[attr-defined]
    _exercise_sliders(fig)
    assert any("mode" in text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_bayesian_regression_updates() -> None:
    """Ch3 BLR interactive refits and reports recovered coefficients."""
    fig = interactive_bayesian_regression()
    assert len(fig._sliders) == 2  # type: ignore[attr-defined]
    _exercise_sliders(fig)
    assert any("β1" in text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_predictive_coding_updates() -> None:
    """Ch5 predictive-coding interactive tracks the free-energy minimum μ*."""
    fig = interactive_predictive_coding()
    assert len(fig._sliders) == 4  # type: ignore[attr-defined]
    _exercise_sliders(fig)
    assert any("μ*" in text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_variational_free_energy_updates() -> None:
    """Ch4 (bonus) VFE explorer stays finite, its bar decomposition updates, and
    the added live readout reports μ/σ²/F/divergence/complexity/accuracy."""
    fig = interactive_variational_free_energy()
    assert len(fig._sliders) == 2  # type: ignore[attr-defined]
    ax_bar = fig.axes[1]
    before = [bar.get_height() for bar in ax_bar.patches]
    _exercise_sliders(fig)
    assert _finite_lines(fig)
    after = [bar.get_height() for bar in ax_bar.patches]
    assert all(np.isfinite(h) for h in after)
    assert after != before
    assert any("σ²" in text.get_text() and "F =" in text.get_text()
               for ax in fig.axes for text in ax.texts)


def test_interactive_gradient_descent_scrubs_trajectory() -> None:
    """Ch2 gradient-descent scrubber recomputes on learning-rate changes and
    scrubs the loss trace without producing non-finite artefacts at a
    convergent learning rate."""
    fig = interactive_gradient_descent()
    s_lr, s_it = fig._sliders  # type: ignore[attr-defined]
    assert s_it.valmax >= 1

    # Move to a comfortably convergent learning rate and rescan the trajectory.
    s_lr.set_val(-5.0)
    assert _finite_lines(fig)
    assert any(text.get_text() for ax in fig.axes for text in ax.texts)

    # Scrub back to an earlier iteration on the (recomputed) trajectory.
    s_it.set_val(max(int(s_it.valmax) // 2, 0))
    assert _finite_lines(fig)


def test_interactive_gradient_descent_shows_divergence_at_high_learning_rate() -> None:
    """A learning rate above the stability threshold makes the trajectory grow
    instead of descend — the pedagogical point this slider exists for."""
    fig = interactive_gradient_descent()
    s_lr, s_it = fig._sliders  # type: ignore[attr-defined]
    s_lr.set_val(s_lr.valmax)  # top of the slider range: unstable learning rate
    assert _finite_lines(fig)
    result = fig._state["result"]  # type: ignore[attr-defined]
    # An unstable learning rate should not converge, and the iterate should
    # have moved substantially farther from the truth than it started.
    assert not result.converged
    assert abs(result.history[-1] - result.history[0]) > abs(result.history[1] - result.history[0])
    assert s_it.valmax >= 1
    # Even after the iterate diverges to inf/NaN, the readout must remain
    # present (guarded, not crashing).
    assert any(text.get_text() for ax in fig.axes for text in ax.texts)


def test_interactive_lgs_localization_moves_posterior() -> None:
    """Ch3 2-D LGS explorer's posterior ellipse follows the dragged observation."""
    fig = interactive_lgs_localization()
    assert len(fig._sliders) == 2  # type: ignore[attr-defined]
    _obs_marker, mean_marker = fig._markers  # type: ignore[attr-defined]
    mean_before = np.asarray(mean_marker.get_offsets())[0].copy()
    _exercise_sliders(fig)
    mean_after = np.asarray(mean_marker.get_offsets())[0]
    assert np.all(np.isfinite(mean_after))
    assert not np.allclose(mean_before, mean_after)
    assert any("posterior mean" in text.get_text() for ax in fig.axes for text in ax.texts)


@pytest.mark.parametrize("topic", ["entropy", "gradient_descent", "expected_free_energy", "temperature"])
def test_interactive_topic_demo_uses_registered_simulation_builder(topic: str) -> None:
    """Extras interactives reuse registered simulation-capable topic demos."""
    spec = extra_topic_spec(topic)
    assert spec.has_simulation
    fig = interactive_topic_demo(topic)
    slider = fig._slider  # type: ignore[attr-defined]
    slider.set_val(0.75)
    assert _finite_lines(fig)
    assert any(topic.replace("_", " ") in text.get_text().lower() for ax in fig.axes for text in ax.texts)
