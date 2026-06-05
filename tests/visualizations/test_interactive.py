"""Tests for matplotlib-slider interactive figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from active_inference.extra_topics import extra_topic_spec  # noqa: E402
from active_inference.visualizations.interactive import (  # noqa: E402
    interactive_inference,
    interactive_precision,
    interactive_topic_demo,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close any figures created by slider tests."""
    yield
    plt.close("all")


def _finite_lines(fig: plt.Figure) -> bool:
    """Return whether every plotted line with data contains finite values."""
    lines = [line for ax in fig.axes for line in ax.lines if len(line.get_ydata())]
    assert lines
    return all(np.all(np.isfinite(np.asarray(line.get_ydata(), dtype=float))) for line in lines)


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
