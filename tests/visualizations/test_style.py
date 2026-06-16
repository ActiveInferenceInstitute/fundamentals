"""Tests for shared visualization style helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from active_inference.visualizations.style import (  # noqa: E402
    COLORS,
    annotate_point,
    annotate_stat_box,
    figure_style,
    set_default_style,
    stat_box_bbox,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestStyleHelpers:
    def test_palette_and_stat_box_are_stable_mappings(self) -> None:
        assert {"prior", "likelihood", "posterior", "data", "truth"}.issubset(COLORS)
        bbox = stat_box_bbox(facecolor="white", alpha=0.5)
        assert bbox["boxstyle"].startswith("round")
        assert bbox["alpha"] == pytest.approx(0.5)

    def test_figure_style_restores_rcparams(self) -> None:
        original = matplotlib.rcParams["font.size"]
        with figure_style({"font.size": 22}):
            assert matplotlib.rcParams["font.size"] == 22
        assert matplotlib.rcParams["font.size"] == original
        set_default_style({"font.size": original})

    def test_annotations_attach_to_axes(self) -> None:
        fig, ax = plt.subplots()
        text = annotate_stat_box(ax, "F=1.0", loc="upper right")
        annotate_point(ax, 0.5, 0.25, "fixed", dx=0.1, dy=0.1)
        assert text in ax.texts
        assert len(ax.lines) == 1
        assert len(ax.texts) >= 1
