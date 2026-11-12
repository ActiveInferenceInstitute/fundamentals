"""Read and evaluate the canonical education-contract document schema.

The schema is a file, not a second copy of a check: :mod:`software.src.education_contract`
loads ``specs/education/schema.json`` at call time and evaluates the document against
exactly what that file says. Only the JSON Schema keyword subset the canonical schema
uses is implemented here (``$ref``, ``type``, ``required``, ``properties``,
``additionalProperties``, ``items``, ``const``, ``enum``, ``pattern``, ``minLength``,
``minItems``, ``uniqueItems``); anything outside that subset is an error rather than a
silent pass, so widening the schema forces a deliberate change here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The repository root: software/src/education_contract/schema.py -> courses/
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = ROOT / "specs" / "education" / "schema.json"

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minItems",
        "uniqueItems",
    }
)

_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
}


@dataclass(frozen=True)
class Issue:
    """One rejected document position, with the rule that rejected it."""

    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.code} {self.location}: {self.message}"


def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Return the canonical schema document, parsed."""
    resolved = DEFAULT_SCHEMA_PATH if path is None else path
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved} must hold a JSON object")
    return data


def schema_digest(path: Path | None = None) -> str:
    """Return the SHA-256 of the canonical schema bytes.

    Members copy this digest into their ``education-contract.json`` pointer, so a
    member that adopted an older schema is reported as diverged instead of being
    silently accepted.
    """
    resolved = DEFAULT_SCHEMA_PATH if path is None else path
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref!r}: only local references are supported")
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"unresolvable $ref {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"$ref {ref!r} does not resolve to a schema object")
    return node


def _unsupported_keywords(schema: dict[str, Any], where: str) -> list[Issue]:
    unknown = sorted(set(schema) - SUPPORTED_KEYWORDS)
    return [
        Issue("EC900", where, f"schema keyword {name!r} is not implemented by this validator")
        for name in unknown
    ]


def _check_type(value: Any, expected: str, where: str) -> list[Issue]:
    checker = _TYPE_CHECKS.get(expected)
    if checker is None:
        return [Issue("EC901", where, f"schema type {expected!r} is not implemented")]
    if not isinstance(value, checker):
        return [Issue("EC001", where, f"expected {expected}, found {type(value).__name__}")]
    return []


def _unique(items: list[Any]) -> bool:
    seen = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
    return len(seen) == len(set(seen))


def _validate(value: Any, schema: dict[str, Any], root: dict[str, Any], where: str) -> list[Issue]:
    if "$ref" in schema:
        return _validate(value, _resolve_ref(schema["$ref"], root), root, where)

    issues = _unsupported_keywords(schema, where)

    if "const" in schema and value != schema["const"]:
        issues.append(Issue("EC002", where, f"must equal {schema['const']!r}, found {value!r}"))
    if "enum" in schema and value not in schema["enum"]:
        issues.append(Issue("EC003", where, f"{value!r} is not one of {schema['enum']}"))
    if "type" in schema:
        type_issues = _check_type(value, schema["type"], where)
        if type_issues:
            # A type mismatch makes every other assertion in this node noise.
            return issues + type_issues

    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            issues.append(Issue("EC004", where, f"{value!r} does not match {schema['pattern']!r}"))
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(Issue("EC005", where, f"needs at least {schema['minLength']} characters"))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(Issue("EC006", where, f"needs at least {schema['minItems']} items"))
        if schema.get("uniqueItems") and not _unique(value):
            issues.append(Issue("EC007", where, "items must be unique"))
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(_validate(item, item_schema, root, f"{where}[{index}]"))

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                issues.append(Issue("EC008", where, f"missing required field {name!r}"))
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                issues.extend(_validate(item, properties[name], root, f"{where}.{name}"))
            elif schema.get("additionalProperties") is False:
                issues.append(Issue("EC009", where, f"unknown field {name!r}"))
    return issues


def validate_document(document: Any, schema_path: Path | None = None) -> list[Issue]:
    """Return every schema violation in ``document`` (empty when it conforms)."""
    root = load_schema(schema_path)
    return _validate(document, root, root, "$")


def validate_pointer(pointer: Any, schema_path: Path | None = None) -> list[Issue]:
    """Return every violation in a member's ``education-contract.json`` pointer."""
    root = load_schema(schema_path)
    defs = root.get("$defs", {})
    pointer_schema = defs.get("pointer")
    if not isinstance(pointer_schema, dict):
        raise ValueError("canonical schema has no $defs.pointer")
    return _validate(pointer, pointer_schema, root, "$")
