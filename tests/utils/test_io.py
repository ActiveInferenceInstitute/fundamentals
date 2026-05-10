"""Tests for ``utils.io`` — path conventions and directory management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from active_inference.utils.io import (
    default_data_dir,
    default_figure_dir,
    ensure_dir,
)


class TestPaths:
    def test_default_figure_dir_under_repo_root(self) -> None:
        fig = default_figure_dir()
        assert fig.name == "figures"
        assert fig.parent.name == "output"

    def test_default_data_dir_under_repo_root(self) -> None:
        data = default_data_dir()
        assert data.name == "data"
        assert data.parent.name == "output"

    def test_default_dirs_share_parent(self) -> None:
        assert default_data_dir().parent == default_figure_dir().parent


class TestEnsureDir:
    def test_creates_missing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        out = ensure_dir(target)
        assert out == target
        assert out.is_dir()

    def test_idempotent_on_existing_dir(self, tmp_path: Path) -> None:
        ensure_dir(tmp_path / "x")
        # Calling again on the same path must not raise.
        out = ensure_dir(tmp_path / "x")
        assert out.is_dir()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        out = ensure_dir(str(tmp_path / "y"))
        assert out.is_dir()
