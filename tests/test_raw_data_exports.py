"""Contracts for chapter raw-data sidecars and their validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from active_inference.utils import (
    data_paths_for_figure,
    data_paths_for_extra_figure,
    extra_data_dir,
    extract_animation_data,
    extract_figure_data,
    infer_chapter_from_path,
    infer_extra_topic_from_path,
    save_animation_data,
    save_chapter_data,
    save_extra_data,
    save_figure_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_save_chapter_data_writes_npz_and_json_manifest(tmp_path: Path) -> None:
    """Persist numeric arrays plus machine-readable provenance and statistics."""
    npz_path, json_path = save_chapter_data(
        8,
        "example_8_1_learning_attention",
        arrays={
            "time": np.linspace(0.0, 1.0, 5),
            "beliefs": np.eye(3),
        },
        metadata={
            "script": "example_8_1_learning_attention.py",
            "args": {"save": True, "seed": 7},
            "seed": 7,
            "summary": {"final_error": 0.125},
        },
        figures=[Path("output/figures/chapter_08/example_8_1_learning_attention.png")],
        root=tmp_path,
    )

    assert npz_path == tmp_path / "chapter_08" / "example_8_1_learning_attention.npz"
    assert json_path == tmp_path / "chapter_08" / "example_8_1_learning_attention.json"
    assert npz_path.exists()
    assert json_path.exists()

    with np.load(npz_path) as data:
        assert set(data.files) == {"time", "beliefs"}
        np.testing.assert_allclose(data["time"], np.linspace(0.0, 1.0, 5))

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["chapter"] == 8
    assert manifest["script"] == "example_8_1_learning_attention.py"
    assert manifest["seed"] == 7
    assert manifest["figures"] == [
        "output/figures/chapter_08/example_8_1_learning_attention.png"
    ]
    assert manifest["arrays"]["time"]["shape"] == [5]
    assert manifest["arrays"]["beliefs"]["dtype"].startswith("float")
    assert manifest["summary"]["time"]["finite_fraction"] == pytest.approx(1.0)
    assert manifest["metadata"]["summary"]["final_error"] == pytest.approx(0.125)


def test_save_chapter_data_extracts_seed_from_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI seed provenance is recovered when metadata omits an explicit seed."""
    monkeypatch.setattr(sys, "argv", ["demo.py", "--seed", "42"])
    _, json_path = save_chapter_data(
        1,
        "seeded",
        arrays={"x": np.arange(3, dtype=float)},
        metadata={"script": "demo.py"},
        root=tmp_path,
    )
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42


def test_save_extra_data_writes_topic_npz_and_json_manifest(tmp_path: Path) -> None:
    """Persist extras topic arrays under output/data/extras/<topic>."""
    npz_path, json_path = save_extra_data(
        "temperature",
        "visualize_temperature",
        arrays={
            "temperature": np.array([0.5, 1.0, 2.0]),
            "free_energy": np.array([2.0, 1.0, 0.0]),
        },
        metadata={"script": "visualize_temperature.py", "summary": {"topic": "temperature"}},
        figures=[Path("output/figures/extras/temperature/visualize_temperature.png")],
        root=tmp_path,
    )

    assert npz_path == tmp_path / "extras" / "temperature" / "visualize_temperature.npz"
    assert json_path == tmp_path / "extras" / "temperature" / "visualize_temperature.json"
    assert npz_path.exists()
    assert json_path.exists()

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["section"] == "extras"
    assert manifest["topic"] == "temperature"
    assert manifest["script"] == "visualize_temperature.py"
    assert manifest["figures"] == [
        "output/figures/extras/temperature/visualize_temperature.png"
    ]
    assert manifest["arrays"]["temperature"]["shape"] == [3]
    assert manifest["metadata"]["summary"]["topic"] == "temperature"


@pytest.mark.parametrize(
    "arrays",
    [
        {"empty": np.array([])},
        {"nonfinite": np.array([0.0, np.inf])},
        {"object": np.array([{"bad": "shape"}], dtype=object)},
        {"string": np.array(["not", "numeric"])},
    ],
    ids=["empty", "nonfinite", "object", "string"],
)
def test_save_chapter_data_rejects_invalid_arrays(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Reject arrays that cannot support reproducible numeric reconstruction."""
    with pytest.raises(ValueError):
        save_chapter_data(1, "bad_export", arrays=arrays, root=tmp_path)


def test_extract_figure_data_captures_plot_reconstruction_arrays() -> None:
    """Extract line, image, scatter, and figure geometry arrays from a figure."""
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    x = np.linspace(0.0, 1.0, 4)
    ax.plot(x, x**2, label="curve")
    ax.scatter(x, x + 1.0, label="samples")
    ax.imshow(np.arange(4, dtype=float).reshape(2, 2), alpha=0.2)
    arrays, metadata = extract_figure_data(fig)

    assert "figure_size_inches" in arrays
    assert "axes_0_line_0_x" in arrays
    assert "axes_0_line_0_y" in arrays
    assert any(name.endswith("_image_0") for name in arrays)
    assert any(name.endswith("_collection_0_offsets") for name in arrays)
    assert metadata["axes"][0]["xlabel"] == ""
    assert "curve" in metadata["axes"][0]["legend_labels"]
    plt.close(fig)


def test_extract_figure_data_handles_bar_container_legends() -> None:
    """Extract metadata from bar charts whose legend handles are containers."""
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.bar([0, 1, 2], [1.0, 2.0, 1.5], label="bars")
    ax.legend()
    arrays, metadata = extract_figure_data(fig)

    assert arrays["axes_count"].shape == (1,)
    assert any(name.endswith("_patch_0_bounds") for name in arrays)
    assert "bars" in metadata["axes"][0]["legend_labels"]
    plt.close(fig)


def test_data_paths_for_figure_maps_output_figures_to_output_data() -> None:
    """Map a rendered artifact path to the matching chapter data sidecar paths."""
    npz_path, json_path = data_paths_for_figure(
        REPO_ROOT / "output" / "figures" / "chapter_08" / "demo.png"
    )
    assert npz_path == REPO_ROOT / "output" / "data" / "chapter_08" / "demo.npz"
    assert json_path == REPO_ROOT / "output" / "data" / "chapter_08" / "demo.json"


def test_data_paths_for_extra_figure_maps_output_figures_to_output_data() -> None:
    """Map extras rendered artifacts to matching topic raw-data sidecars."""
    npz_path, json_path = data_paths_for_extra_figure(
        REPO_ROOT / "output" / "figures" / "extras" / "entropy" / "demo.png"
    )
    assert npz_path == REPO_ROOT / "output" / "data" / "extras" / "entropy" / "demo.npz"
    assert json_path == REPO_ROOT / "output" / "data" / "extras" / "entropy" / "demo.json"
    assert extra_data_dir("entropy") == REPO_ROOT / "output" / "data" / "extras" / "entropy"


def test_infer_path_helpers_reject_unmatched_paths() -> None:
    """Inference helpers fail closed when a path lacks a chapter/extras marker."""
    with pytest.raises(ValueError):
        infer_chapter_from_path("figures/demo.png")
    with pytest.raises(ValueError):
        infer_extra_topic_from_path("figures/demo.png")


def test_save_figure_data_routes_chapter_and_extras_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic figure extraction writes sidecars for chapter and extras paths."""
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0], label="line")
    chapter_result = save_figure_data(
        fig,
        tmp_path / "output" / "figures" / "chapter_11" / "demo.png",
        {"script": "demo.py"},
    )
    extras_result = save_figure_data(
        fig,
        tmp_path / "output" / "figures" / "extras" / "entropy" / "demo.png",
        {"script": "demo.py"},
    )
    ignored = save_figure_data(fig, tmp_path / "elsewhere" / "demo.png")
    assert chapter_result is not None and chapter_result[0].exists()
    assert extras_result is not None and extras_result[0].exists()
    assert ignored is None
    plt.close(fig)


def test_extract_and_save_animation_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Animation extraction captures raw data and writes chapter sidecars."""
    monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "output"))
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    xs = np.linspace(0.0, 1.0, 4)
    ys = np.vstack([xs * i for i in range(3)])

    def update(frame: int):
        line.set_data(xs, ys[frame])
        return (line,)

    anim = FuncAnimation(fig, update, frames=3)
    anim._raw_data = {"xs": xs, "ys": ys}  # type: ignore[attr-defined]
    arrays, metadata = extract_animation_data(anim)
    assert "raw_xs" in arrays
    assert "raw_ys" in arrays
    assert metadata["animation_class"] == "FuncAnimation"

    result = save_animation_data(
        anim,
        tmp_path / "output" / "figures" / "chapter_12" / "demo.gif",
        {"script": "demo.py"},
    )
    assert result is not None and result[0].exists()
    ignored = save_animation_data(anim, tmp_path / "elsewhere" / "demo.gif")
    assert ignored is None
    anim._draw_was_started = True
    plt.close(fig)


def test_validate_raw_data_exports_accepts_good_and_rejects_bad(
    tmp_path: Path,
) -> None:
    """Exercise the CLI validator on valid and intentionally corrupted fixtures."""
    save_chapter_data(
        2,
        "good",
        arrays={"x": np.arange(3, dtype=float)},
        metadata={"script": "good.py"},
        root=tmp_path,
    )
    validator = REPO_ROOT / "scripts" / "validate_raw_data_exports.py"
    good = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path), "--chapters", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0, good.stderr

    bad_json = tmp_path / "chapter_02" / "good.json"
    manifest = json.loads(bad_json.read_text(encoding="utf-8"))
    manifest["arrays"]["x"]["shape"] = [99]
    bad_json.write_text(json.dumps(manifest), encoding="utf-8")

    bad = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path), "--chapters", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "shape" in bad.stderr


def test_validate_raw_data_exports_accepts_extras_topics(tmp_path: Path) -> None:
    """The raw-data validator can require extras topic exports."""
    save_extra_data(
        "entropy",
        "good",
        arrays={"x": np.arange(3, dtype=float)},
        metadata={"script": "good.py"},
        root=tmp_path,
    )
    validator = REPO_ROOT / "scripts" / "validate_raw_data_exports.py"

    good = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path), "--extras", "entropy"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert good.returncode == 0, good.stderr
