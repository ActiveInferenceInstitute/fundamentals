"""Conformance tests binding the committed GNN declaration to the implementation.

The weather agent of Chapter 9 §9.1 is declared twice on purpose: as the GNN
document ``specs/weather_pomdp.gnn.md`` and as the literals in
``chapters/chapter_09/example_9_1_state_inference.py``. These tests are the
drift guard between them:

* :func:`test_declared_arrays_equal_the_chapter_literals` fails the moment
  either side is edited alone, comparing the declared ``A`` and ``D`` with the
  chapter script's own assignments read from its source;
* :func:`test_infer_states_reproduces_the_book_posterior` runs the declared
  matrices through the library and checks the book's Eq. 15 numbers;
* :func:`test_pinned_parser_validates_the_committed_spec` parses the document
  with the parser revision recorded in ``specs/gnn-pin.json``; when no pinned
  runtime or checkout is available the test prints an explicit skip, so an
  unparsed document is never reported as a pass.

A green run proves the declaration and the implementation agree and that the
document is well-formed GNN. It does not prove the model is scientifically
adequate or that the book's numbers are correct beyond the Eq. 15 posterior
asserted here.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from active_inference import POMDPModel, infer_states
from active_inference.gnn_spec import (
    GNN_LANGUAGE_VERSION,
    GNN_PIN_PATH,
    GNN_SPEC_PATH,
    REQUIRED_SECTIONS,
    SPEC_MATRIX_KEYS,
    declared_arrays,
    load_pin,
    parse_parameterization,
    parse_sections,
    parse_variables,
    spec_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_MODEL = ROOT / "chapters" / "chapter_09" / "example_9_1_state_inference.py"

#: The book's Eq. 15 posterior for the observation "hot" (row 2 of A).
BOOK_POSTERIOR_HOT = np.array([0.18, 0.40, 0.36, 0.06])

#: The exact posterior for the observation "water" (row 1 of A): A[water, :] D normalized.
BOOK_POSTERIOR_WATER = np.array([0.80, 0.33, 0.05, 0.40]) / np.array([0.80, 0.33, 0.05, 0.40]).sum()


def _chapter_literals() -> dict[str, np.ndarray]:
    """Read the chapter script's ``A`` and ``D`` assignments from its source."""

    tree = ast.parse(CHAPTER_MODEL.read_text(encoding="utf-8"))
    literals: dict[str, np.ndarray] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id in SPEC_MATRIX_KEYS):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        literals[target.id] = np.asarray(ast.literal_eval(value), dtype=float)
    missing = sorted(set(SPEC_MATRIX_KEYS) - set(literals))
    assert not missing, f"{CHAPTER_MODEL.name} no longer assigns {missing}"
    return literals


def test_committed_spec_declares_the_required_sections() -> None:
    sections = list(parse_sections())
    assert sections == list(REQUIRED_SECTIONS)
    summary = spec_summary()
    assert summary["model_name"] == "Fundamentals Weather POMDP (Example 9.1)"
    assert summary["language_version"] == GNN_LANGUAGE_VERSION
    assert summary["variables"]["A"]["dimensions"] == [3, 4]
    assert summary["variables"]["D"]["dimensions"] == [4]


def test_declared_arrays_equal_the_chapter_literals() -> None:
    declared = declared_arrays()
    literals = _chapter_literals()

    assert declared["A"].shape == literals["A"].shape == (3, 4)
    assert declared["D"].shape == literals["D"].shape == (4,)
    assert np.array_equal(declared["A"], literals["A"])
    assert np.array_equal(declared["D"], literals["D"])


def test_declared_arrays_are_valid_probability_arrays() -> None:
    declared = declared_arrays()

    assert (declared["A"] >= 0).all()
    assert np.allclose(declared["A"].sum(axis=0), 1.0)
    assert (declared["D"] >= 0).all()
    assert np.isclose(declared["D"].sum(), 1.0)


def test_declared_shapes_match_the_state_space_block() -> None:
    declared = declared_arrays()
    variables = parse_variables()

    for key in SPEC_MATRIX_KEYS:
        assert declared[key].shape == variables[key].dimensions
        assert int(np.prod(variables[key].dimensions)) == declared[key].size


def test_infer_states_reproduces_the_book_posterior() -> None:
    declared = declared_arrays()
    model = POMDPModel(A=declared["A"], D=declared["D"])

    # Book Eq. 15 rounds to two decimals; the unrounded values are the same
    # posterior, matching the oracle discipline of tests/core/test_pomdp.py.
    np.testing.assert_allclose(infer_states(model, 1), BOOK_POSTERIOR_HOT, atol=0.01)
    np.testing.assert_allclose(infer_states(model, 0), BOOK_POSTERIOR_WATER, atol=0.01)
    np.testing.assert_allclose(
        infer_states(model, 1), [0.1807, 0.3976, 0.3614, 0.0602], atol=1e-3
    )


def test_reader_rejects_malformed_documents() -> None:
    with pytest.raises(ValueError):
        parse_parameterization("## InitialParameterization\n\nA={(0.1, 0.2)")
    text = GNN_SPEC_PATH.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        declared_arrays(text.replace("A[3,4,type=float]", "A[2,4,type=float]"))
    with pytest.raises(ValueError):
        declared_arrays(text.replace("D={(0.25, 0.25, 0.25, 0.25)}", "D={(0.25, 0.25)}"))


def test_pin_record_names_a_pinned_upstream_revision() -> None:
    pin = load_pin()

    assert pin["repository"].endswith("Generalized_Notation_Notation")
    assert re.fullmatch(r"[0-9a-f]{40}", pin["commit"])
    assert re.fullmatch(r"v\d+\.\d+\.\d+", pin["tag"])
    assert pin["language_version"] == GNN_LANGUAGE_VERSION
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pin["recorded"])
    assert pin["parser_files"], "the pin must record the parser files it was reviewed against"


def _pinned_parser_source() -> dict[str, str] | None:
    """Return the validator arguments for a pinned parser, or ``None`` when absent."""

    runtime = os.environ.get("GNN_RUNTIME_PYTHON")
    if runtime and Path(runtime).is_file():
        return {"--gnn-runtime": runtime}
    roots = [
        Path(root)
        for root in (os.environ.get("GNN_ROOT"), str(ROOT.parent / "Generalized_Notation_Notation"))
        if root
    ]
    for root in roots:
        interpreter = Path(os.environ.get("GNN_PYTHON") or root / ".venv" / "bin" / "python")
        if (root / "src" / "gnn").is_dir() and interpreter.is_file():
            return {"--gnn-root": str(root), "--python": str(interpreter)}
    return None


def test_pinned_parser_validates_the_committed_spec(tmp_path: Path) -> None:
    source = _pinned_parser_source()
    if source is None:
        pytest.skip(
            "no pinned GNN parser available: set GNN_RUNTIME_PYTHON to an interpreter with the "
            "pinned distribution, or GNN_ROOT (with GNN_PYTHON) to a checkout — the committed "
            "document was NOT parsed by the pinned revision in this run"
        )
    receipt = tmp_path / "gnn-receipt.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_gnn_spec.py"),
        "--receipt",
        str(receipt),
        # A pass requires the pin's tag to be bound to its commit. The checkout
        # path resolves it in the local object store and ignores this flag; the
        # installed-runtime path resolves it upstream.
        "--verify-tag-remote",
    ]
    for flag, value in source.items():
        command += [flag, value]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["failures"] == []
    assert payload["gnn_version"] == load_pin()["tag"].lstrip("v")
    assert payload["parser_success"] is True
    assert payload["semantic_valid"] is True
    assert payload["consistency_score"] == 1.0


def test_pin_file_and_spec_paths_are_tracked() -> None:
    assert GNN_SPEC_PATH.is_file()
    assert GNN_PIN_PATH.is_file()
    assert GNN_SPEC_PATH.parent == GNN_PIN_PATH.parent == ROOT / "specs"
