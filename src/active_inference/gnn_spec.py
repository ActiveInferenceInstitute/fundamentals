"""Dependency-free reader for the committed GNN declaration of the Chapter 9 model.

The weather agent of *Fundamentals of Active Inference* (Namjoshi), Chapter 9
§9.1, is declared as a Generalized Notation Notation (GNN) document under
``specs/weather_pomdp.gnn.md``. This module reads that document with the Python
standard library and NumPy only, so the declaration is inspectable without the
upstream parser:

* :func:`parse_sections` and :func:`parse_variables` expose the declared
  sections and the ``StateSpaceBlock`` variables with their dimensions;
* :func:`declared_arrays` returns the declared ``A`` and ``D`` matrices,
  reshaped to the declared dimensions;
* :func:`spec_summary` returns the model name, language version, section list,
  declared variables, and ``ModelParameters`` scalars;
* :func:`load_pin` reads ``specs/gnn-pin.json``, the record of the upstream
  parser revision that validates the document.

``scripts/validate_gnn_spec.py`` compares this reader's view with the pinned
upstream parser's view of the same file, and
``tests/test_gnn_spec.py`` binds the declared matrices to the literals in
``chapters/chapter_09/example_9_1_state_inference.py`` and to the runtime
posterior computed by :func:`active_inference.infer_states`. See
``docs/gnn.md`` for the adoption contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "GNN_LANGUAGE_VERSION",
    "GNN_PIN_PATH",
    "GNN_SPEC_PATH",
    "REQUIRED_SECTIONS",
    "SPEC_DIR",
    "SPEC_MATRIX_KEYS",
    "GNNVariable",
    "declared_arrays",
    "load_pin",
    "parse_parameterization",
    "parse_sections",
    "parse_variables",
    "spec_summary",
]

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "specs"
GNN_SPEC_PATH = SPEC_DIR / "weather_pomdp.gnn.md"
GNN_PIN_PATH = SPEC_DIR / "gnn-pin.json"

#: GNN language revision declared by the document header.
GNN_LANGUAGE_VERSION = "GNN v1.1"

#: Declared matrix keys of the canonical model, in declaration order.
SPEC_MATRIX_KEYS = ("A", "D")

#: Sections the committed document must declare, in order.
REQUIRED_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)

_SECTION_RE = re.compile(r"^##\s+([A-Za-z][A-Za-z0-9]*)\s*$", re.MULTILINE)
_VARIABLE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_']*)\s*\[([^\]]*)\]")
_PARAMETER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_']*)\s*=\s*(.*)$", re.MULTILINE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_MODEL_PARAMETER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")


@dataclass(frozen=True)
class GNNVariable:
    """One ``StateSpaceBlock`` variable declaration."""

    name: str
    dimensions: tuple[int, ...]
    data_type: str = "float"


def _read_spec_text(text: str | None) -> str:
    """Return the committed document text, or the caller's override."""

    return GNN_SPEC_PATH.read_text(encoding="utf-8") if text is None else text


def _strip_comments(text: str) -> str:
    """Drop ``#`` comments from GNN blocks so declarations parse as data."""

    return "\n".join(line.split("#", 1)[0].rstrip() for line in text.splitlines())


def parse_sections(text: str | None = None) -> dict[str, str]:
    """Return the document's sections keyed by their level-2 heading name."""

    source = _read_spec_text(text)
    matches = list(_SECTION_RE.finditer(source))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        sections[match.group(1)] = source[match.end() : end].strip("\n")
    return sections


def parse_variables(text: str | None = None) -> dict[str, GNNVariable]:
    """Return the declared ``StateSpaceBlock`` variables."""

    sections = parse_sections(text)
    variables: dict[str, GNNVariable] = {}
    for line in _strip_comments(sections.get("StateSpaceBlock", "")).splitlines():
        match = _VARIABLE_RE.match(line.strip())
        if not match:
            continue
        name, inside = match.group(1), match.group(2)
        dimensions: list[int] = []
        data_type = "float"
        for part in inside.split(","):
            item = part.strip()
            if item.startswith("type="):
                data_type = item.removeprefix("type=").strip()
            elif item:
                dimensions.append(int(item))
        if not dimensions:
            raise ValueError(f"variable {name!r} declares no dimensions")
        variables[name] = GNNVariable(name=name, dimensions=tuple(dimensions), data_type=data_type)
    return variables


def _braced_body(source: str, start: int) -> str:
    """Return the text between the braces starting at ``start`` (a ``{`` index)."""

    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise ValueError("unbalanced braces in the InitialParameterization section")


def parse_parameterization(text: str | None = None) -> dict[str, list[float]]:
    """Return declared parameter values in declaration order, keyed by name."""

    sections = parse_sections(text)
    body = _strip_comments(sections.get("InitialParameterization", ""))
    parameters: dict[str, list[float]] = {}
    for match in _PARAMETER_RE.finditer(body):
        name, remainder = match.group(1), match.group(2).strip()
        if remainder.startswith("{"):
            content = _braced_body(body, body.index("{", match.start(2)))
        elif remainder.startswith("(") and remainder.endswith(")"):
            content = remainder[1:-1]
        else:
            content = remainder
        parameters[name] = [float(value) for value in _NUMBER_RE.findall(content)]
    return parameters


def declared_arrays(text: str | None = None) -> dict[str, np.ndarray]:
    """Return the declared ``A`` and ``D`` arrays reshaped to their dimensions.

    The element count of each parameterization must equal the product of the
    declared ``StateSpaceBlock`` dimensions, which is the GNN dimension rule the
    pinned parser enforces independently.
    """

    source = _read_spec_text(text)
    variables = parse_variables(source)
    parameters = parse_parameterization(source)
    arrays: dict[str, np.ndarray] = {}
    for key in SPEC_MATRIX_KEYS:
        if key not in parameters:
            raise ValueError(f"the document does not declare {key}")
        variable = variables.get(key)
        if variable is None:
            raise ValueError(f"the document does not declare variable {key}")
        values = parameters[key]
        expected = int(np.prod(variable.dimensions))
        if len(values) != expected:
            raise ValueError(
                f"declared {key} has {len(values)} values but "
                f"{variable.dimensions} requires {expected}"
            )
        arrays[key] = np.asarray(values, dtype=float).reshape(variable.dimensions)
    return arrays


def load_pin(path: Path | None = None) -> dict[str, Any]:
    """Return the upstream parser pin recorded for this document."""

    return json.loads((GNN_PIN_PATH if path is None else path).read_text(encoding="utf-8"))


def spec_summary(text: str | None = None) -> dict[str, Any]:
    """Return the declared model name, version, sections, variables, and scalars."""

    source = _read_spec_text(text)
    sections = parse_sections(source)
    variables = parse_variables(source)
    scalars: dict[str, str] = {}
    for line in _strip_comments(sections.get("ModelParameters", "")).splitlines():
        match = _MODEL_PARAMETER_RE.match(line.strip())
        if match:
            scalars[match.group(1)] = match.group(2)
    return {
        "model_name": sections.get("ModelName", "").strip(),
        "language_version": sections.get("GNNVersionAndFlags", "").strip(),
        "sections": list(sections),
        "variables": {
            name: {"dimensions": list(variable.dimensions), "data_type": variable.data_type}
            for name, variable in variables.items()
        },
        "scalars": scalars,
        "arrays": {key: list(array.shape) for key, array in declared_arrays(source).items()},
    }
