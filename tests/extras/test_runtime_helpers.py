"""In-process tests for extras topic rendering and CLI helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import active_inference.extra_topics as extra_topics
from active_inference.extra_topics import (
    ExtraTopicSpec,
    TopicDemo,
    build_topic_animation,
    main_animation,
    main_interactive,
    main_simulate,
    main_visualize,
    render_topic_figure,
    topic_artifact_path,
)


def test_render_topic_figure_covers_heatmap_and_bar_branches(monkeypatch) -> None:
    spec = ExtraTopicSpec(
        slug="fake_heatmap",
        title="Fake Heatmap",
        family="Test",
        chapters=(12,),
        sections=("12.0",),
        summary="Synthetic heatmap branch.",
        demo_kind="factor_graph",
    )

    def fake_demo(slug: str, *, mode: str = "visualize") -> TopicDemo:
        return TopicDemo(
            spec=spec,
            arrays={
                "x": np.arange(3, dtype=float),
                "primary": np.arange(3, dtype=float),
                "heat": np.arange(9, dtype=float).reshape(3, 3),
            },
            metadata={"x_label": "x", "primary_label": "primary", "source_apis": list(spec.source_apis)},
            line_keys=("primary",),
            heatmap_key="heat",
        )

    monkeypatch.setattr(extra_topics, "build_topic_demo", fake_demo)
    fig_heatmap, demo_heatmap = extra_topics.render_topic_figure("fake_heatmap")
    monkeypatch.undo()

    fig_bar, demo_bar = render_topic_figure("model_representation")
    assert demo_heatmap.heatmap_key == "heat"
    assert demo_bar.bar_key is not None
    assert len(fig_heatmap.axes) >= 2
    assert len(fig_bar.axes) >= 2
    plt.close(fig_heatmap)
    plt.close(fig_bar)


def test_build_topic_animation_callbacks_are_finite() -> None:
    anim, raw, metadata = build_topic_animation("entropy")
    init = getattr(anim, "_init_func", None)
    if init is not None:
        init()
    frames = list(anim.new_frame_seq())
    anim._func(frames[0])
    anim._func(frames[-1])
    assert metadata["trajectory_kind"]
    assert all(np.all(np.isfinite(values)) for values in raw.values())
    anim._draw_was_started = True
    plt.close(anim._fig)


def test_topic_artifact_path_uses_output_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    path = topic_artifact_path("entropy", "visualize_entropy", "png")
    assert path == tmp_path / "output" / "figures" / "extras" / "entropy" / "visualize_entropy.png"
    assert path.parent.is_dir()


def test_main_visualize_and_simulate_save_to_temp_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    assert main_visualize("entropy", ["--save"]) == 0
    assert main_simulate("entropy", ["--save"]) == 0
    assert (tmp_path / "output" / "figures" / "extras" / "entropy" / "visualize_entropy.png").exists()
    assert (tmp_path / "output" / "data" / "extras" / "entropy" / "simulate_entropy.json").exists()


def test_main_animation_save_and_interactive_non_save(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    assert main_animation("entropy", ["--save"]) == 0
    assert (tmp_path / "output" / "figures" / "extras" / "entropy" / "animation_entropy.gif").exists()

    shown = {"count": 0}

    def fake_show() -> None:
        shown["count"] += 1

    monkeypatch.setattr(plt, "show", fake_show)
    assert main_interactive("entropy", []) == 0
    assert shown["count"] == 1
    plt.close("all")


def test_main_interactive_save_is_gui_only(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    assert main_interactive("entropy", ["--save"]) == 0
    captured = capsys.readouterr()
    assert "GUI-only" in captured.out
