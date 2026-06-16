"""Tests for raw-data export helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from active_inference.utils.export import (
    chapter_data_dir,
    data_paths_for_figure,
    data_paths_for_extra_figure,
    extra_data_dir,
    save_chapter_data,
    save_extra_data,
)


class TestExportPaths:
    def test_chapter_and_extra_dirs_are_rooted_under_requested_root(self, tmp_path) -> None:
        assert chapter_data_dir("chapter_03", root=tmp_path) == tmp_path / "chapter_03"
        assert extra_data_dir("variational free energy", root=tmp_path) == tmp_path / "extras" / "variational_free_energy"

    def test_data_paths_follow_figure_locations(self, tmp_path) -> None:
        npz, js = data_paths_for_figure("output/figures/chapter_02/example.png", root=tmp_path)
        assert npz == tmp_path / "chapter_02" / "example.npz"
        assert js == tmp_path / "chapter_02" / "example.json"
        npz, js = data_paths_for_extra_figure("output/figures/extras/topic/demo.png", root=tmp_path)
        assert npz == tmp_path / "extras" / "topic" / "demo.npz"
        assert js == tmp_path / "extras" / "topic" / "demo.json"


class TestExportWriters:
    def test_save_chapter_data_writes_npz_and_manifest(self, tmp_path) -> None:
        npz, js = save_chapter_data(2, "demo-stem", {"x value": [1.0, 2.0]}, {"seed": 7}, root=tmp_path)
        assert npz.exists() and js.exists()
        with np.load(npz) as data:
            np.testing.assert_allclose(data["x_value"], [1.0, 2.0])
        manifest = json.loads(js.read_text())
        assert manifest["chapter"] == 2
        assert manifest["seed"] == 7
        assert manifest["arrays"]["x_value"]["shape"] == [2]

    def test_save_extra_data_writes_topic_manifest(self, tmp_path) -> None:
        npz, js = save_extra_data("topic", "demo", {"curve": np.array([0.0, 1.0])}, root=tmp_path)
        assert npz == tmp_path / "extras" / "topic" / "demo.npz"
        manifest = json.loads(js.read_text())
        assert manifest["section"] == "extras"
        assert manifest["topic"] == "topic"

    def test_rejects_empty_or_non_numeric_arrays(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            save_chapter_data(1, "empty", {}, root=tmp_path)
        with pytest.raises(ValueError, match="numeric"):
            save_chapter_data(1, "bad", {"labels": ["a", "b"]}, root=tmp_path)
