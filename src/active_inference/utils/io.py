"""Path conventions for figures and serialized data produced by chapter scripts."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_ROOT_ENV = "ACTIVE_INFERENCE_OUTPUT_ROOT"
_FIGURE_DIR_ENV = "ACTIVE_INFERENCE_FIGURE_DIR"
_DATA_DIR_ENV = "ACTIVE_INFERENCE_DATA_DIR"


def _output_root() -> Path:
    """Return the output root, honoring the smoke-test environment override."""
    override = os.environ.get(_OUTPUT_ROOT_ENV)
    if override:
        return Path(override)
    return _REPO_ROOT / "output"


def default_figure_dir() -> Path:
    """Return the directory where chapter and extras figures are rendered."""
    override = os.environ.get(_FIGURE_DIR_ENV)
    if override:
        return Path(override)
    return _output_root() / "figures"


def default_data_dir() -> Path:
    """Return the directory where raw NPZ and JSON data is exported."""
    override = os.environ.get(_DATA_DIR_ENV)
    if override:
        return Path(override)
    return _output_root() / "data"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it unchanged."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
