"""Tests for the chapter/extras orchestrator provenance validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
    """Chapter, extras, and demo wrappers stay thin and route through active_inference."""
    validator = _load_validator()
    errors = validator.validate_orchestrators(
        REPO_ROOT / "chapters",
        REPO_ROOT / "extras",
        REPO_ROOT / "demo",
    )
    assert errors == []


def test_validator_rejects_sibling_imports_and_missing_library_route(tmp_path: Path) -> None:
    """The validator catches script-to-script coupling and isolated orchestration."""
    chapters = tmp_path / "chapters" / "chapter_01"
    extras = tmp_path / "extras" / "toy_topic"
    demos = tmp_path / "demo" / "toy_demo"
    chapters.mkdir(parents=True)
    extras.mkdir(parents=True)
    demos.mkdir(parents=True)
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
    errors = validator.validate_orchestrators(
        tmp_path / "chapters",
        tmp_path / "extras",
        tmp_path / "demo",
    )

    joined = "\n".join(errors)
    assert "sibling script import" in joined
    assert "active_inference API" in joined


@pytest.mark.parametrize("stale_import", [
    "from .legacy_helper import prepare",
    "from legacy_helper import prepare",
    "from chapters.chapter_01.example_1_1 import prepare",
    "from extras.entropy.visualize_entropy import prepare",
    "from demo.toy.visualize_toy import prepare",
])
def test_current_library_route_does_not_hide_stale_orchestrator_dependency(
    tmp_path: Path, stale_import: str,
) -> None:
    validator = _load_validator()
    script = tmp_path / "visualize_owned.py"
    current = (
        '"""Owned provenance fixture; never executed."""\n'
        "from active_inference.extra_topics import main_visualize\n"
        "if __name__ == '__main__':\n"
        "    main_visualize('entropy')\n"
    )
    script.write_text(current, encoding="utf-8")
    assert validator._validate_file(script, sibling_modules={"legacy_helper"}) == []
    script.write_text(stale_import + "\n" + current, encoding="utf-8")
    errors = validator._validate_file(script, sibling_modules={"legacy_helper"})
    assert errors
    assert all(str(script) in error for error in errors)
