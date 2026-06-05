"""Tests for the book-grounded extras coverage validator."""

from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path

import numpy as np

from active_inference.extra_topics import EXTRA_TOPICS, build_topic_demo


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    """Import the standalone coverage validator as a module for direct testing."""
    path = REPO_ROOT / "scripts" / "validate_book_topic_coverage.py"
    spec = importlib.util.spec_from_file_location("validate_book_topic_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_book_topic_coverage_matrix_matches_registry() -> None:
    """The coverage matrix must enumerate every registered extras topic."""
    validator = _load_validator()
    errors = validator.validate_coverage(
        REPO_ROOT / "docs" / "reference" / "book_topic_matrix.md",
        REPO_ROOT / "extras",
    )
    assert errors == []


def test_book_topic_coverage_can_require_rendered_artifacts() -> None:
    """Strict coverage mode requires rendered media plus raw-data sidecars."""
    validator = _load_validator()
    errors = validator.validate_coverage(
        REPO_ROOT / "docs" / "reference" / "book_topic_matrix.md",
        REPO_ROOT / "extras",
        require_rendered=True,
        figures_root=REPO_ROOT / "output" / "figures" / "extras",
        data_root=REPO_ROOT / "output" / "data" / "extras",
    )
    assert errors == []


def test_extra_topic_registry_invariants() -> None:
    """Registered topics have unique slugs and complete display metadata."""
    slugs = [spec.slug for spec in EXTRA_TOPICS]
    assert len(slugs) == len(set(slugs))
    for spec in EXTRA_TOPICS:
        assert spec.slug == spec.slug.lower()
        assert spec.title
        assert spec.family
        assert spec.chapters
        assert spec.sections
        assert spec.summary.endswith(".")
        assert spec.source_apis
        for dotted_path in spec.source_apis:
            module_name, _, attr = dotted_path.rpartition(".")
            assert module_name, dotted_path
            assert attr, dotted_path
            module = importlib.import_module(module_name)
            assert hasattr(module, attr), dotted_path


def test_extra_topic_declared_modes_produce_finite_arrays() -> None:
    """Registry-backed visualize and simulate demos produce finite numeric arrays."""
    for spec in EXTRA_TOPICS:
        modes = ["visualize"]
        if spec.has_simulation:
            modes.append("simulate")
        for mode in modes:
            demo = build_topic_demo(spec.slug, mode=mode)
            assert demo.spec is spec
            assert demo.arrays
            assert demo.metadata["source_apis"] == list(spec.source_apis)
            for name, array in demo.arrays.items():
                values = np.asarray(array)
                assert values.size > 0, f"{spec.slug}/{mode}/{name} is empty"
                assert np.all(np.isfinite(values)), f"{spec.slug}/{mode}/{name} is non-finite"
