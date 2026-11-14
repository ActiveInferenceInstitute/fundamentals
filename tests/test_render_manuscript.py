"""Real source fixtures for required manuscript-tool admission."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest


@pytest.fixture
def manuscript(tmp_path, monkeypatch):
    """Relocate the real renderer to a complete, isolated manuscript fixture."""
    source = Path(__file__).resolve().parents[1] / "scripts/render_manuscript.py"
    spec = importlib.util.spec_from_file_location("manuscript_admission_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    manuscript_dir = root / "manuscript"
    manuscript_dir.mkdir(parents=True)
    for name in ("chapters", "extras", "specs", "output/figures"):
        (root / name).mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
        '[project.urls]\nHomepage="https://github.com/example/fixture"\n'
    )
    (root / "specs/gnn-pin.json").write_text(json.dumps({
        "schema": "fixture", "tag": "v1", "commit": "a" * 40,
        "language_version": "GNN v1.1",
    }))
    (manuscript_dir / "config.yaml").write_text(
        "paper:\n  title: Fixture\nrender:\n  formats:\n    html: true\n"
    )
    (manuscript_dir / "variables.yaml").write_text(
        "schema: manuscript-variables/v1\nvariables:\n  PACKAGE_NAME:\n"
        "    value: fixture\n    source: pyproject.toml\n"
    )
    (manuscript_dir / "01_intro.md").write_text(
        "# Introduction {#sec:intro}\n\n{{PACKAGE_NAME}} refers to @sec:intro.\n"
    )
    (manuscript_dir / "references.bib").write_text("")
    registry = root / "output/figures/figure_registry.json"
    registry.write_text("{}")
    build = root / "docs/_build/manuscript"
    build.mkdir(parents=True)
    for name, value in {
        "REPO_ROOT": root, "MANUSCRIPT_DIR": manuscript_dir,
        "CONFIG_PATH": manuscript_dir / "config.yaml",
        "VARIABLES_PATH": manuscript_dir / "variables.yaml",
        "REGISTRY_PATH": registry, "BUILD_DIR": build,
    }.items():
        monkeypatch.setattr(module, name, value)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    marker = tmp_path / "pandoc-invoked"
    pandoc = binary_dir / "pandoc"
    pandoc.write_text(f"#!/bin/sh\nprintf invoked > {shlex.quote(str(marker))}\nexit 99\n")
    pandoc.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary_dir))
    return module, build, marker


class TestRequiredCrossReferenceResolver:
    def test_missing_crossref_preserves_outputs_and_never_invokes_pandoc(
        self, manuscript, capsys,
    ):
        renderer, build, marker = manuscript
        output = build / "fixture.html"
        receipt = build / "render-receipt.json"
        output.write_bytes(b"previous qualified output")
        receipt.write_bytes(b"previous qualified receipt")
        assert renderer.main(["--receipt", str(receipt)]) == 3
        assert "pandoc-crossref" in capsys.readouterr().err
        assert output.read_bytes() == b"previous qualified output"
        assert receipt.read_bytes() == b"previous qualified receipt"
        assert not marker.exists()

    def test_source_check_does_not_claim_a_render(self, manuscript):
        renderer, build, marker = manuscript
        receipt = renderer.build(check=True)
        assert receipt["checked_only"] is True
        assert receipt["outputs"] == []
        assert not (build / "fixture.html").exists()
        assert not marker.exists()
