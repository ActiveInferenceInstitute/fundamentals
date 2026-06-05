"""Tests for the chapter/extras orchestrator provenance validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    """Import the standalone orchestrator validator for direct testing."""
    path = REPO_ROOT / "scripts" / "validate_orchestrator_provenance.py"
    spec = importlib.util.spec_from_file_location("validate_orchestrator_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_orchestrators_satisfy_source_method_contract() -> None:
    """Chapter and extras wrappers stay thin and route through active_inference."""
    validator = _load_validator()
    errors = validator.validate_orchestrators(
        REPO_ROOT / "chapters",
        REPO_ROOT / "extras",
    )
    assert errors == []


def test_validator_rejects_sibling_imports_and_missing_library_route(tmp_path: Path) -> None:
    """The validator catches script-to-script coupling and isolated orchestration."""
    chapters = tmp_path / "chapters" / "chapter_01"
    extras = tmp_path / "extras" / "toy_topic"
    chapters.mkdir(parents=True)
    extras.mkdir(parents=True)
    (chapters / "helper.py").write_text('"""helper."""\nVALUE = 1\n', encoding="utf-8")
    (chapters / "example_bad.py").write_text(
        '"""bad."""\nfrom helper import VALUE\nprint(VALUE)\n',
        encoding="utf-8",
    )
    (extras / "visualize_toy_topic.py").write_text(
        '"""bad extras wrapper."""\nimport numpy as np\nprint(np.arange(3))\n',
        encoding="utf-8",
    )

    validator = _load_validator()
    errors = validator.validate_orchestrators(tmp_path / "chapters", tmp_path / "extras")

    joined = "\n".join(errors)
    assert "sibling script import" in joined
    assert "active_inference API" in joined
