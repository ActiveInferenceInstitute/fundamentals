"""The contract rules JSON Schema cannot express.

Every rule here corresponds to one row of the rule table in
``docs/workflows/education.md``. The schema proves a document is *shaped* right;
these rules prove it is *coherent*: the prerequisite graph is a DAG, a unit's
denormalized prerequisites equal its inbound hard edges, and every glossary
reference resolves inside the document that makes it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import Issue


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    """Return the object members of ``value``; schema failures are reported elsewhere."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _cycle_members(edges: list[tuple[str, str]]) -> list[str]:
    """Return the units that remain after a Kahn reduction, i.e. one per cycle."""
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if removed == len(nodes):
        return []
    return sorted(node for node in nodes if indegree[node] > 0)


def run_rules(document: dict[str, Any]) -> list[Issue]:
    """Return every semantic violation in one document (empty when it is coherent)."""
    issues: list[Issue] = []
    units = _as_dicts(document.get("units"))
    edges = _as_dicts(document.get("prerequisite_edges"))
    terms = _as_dicts(document.get("glossary"))

    unit_ids = [str(unit.get("unit_id", "")) for unit in units]
    for repeated in _duplicates(unit_ids):
        issues.append(Issue("EC101", "$.units", f"unit_id {repeated!r} is declared twice"))
    known_units = set(unit_ids)

    for index, unit in enumerate(units):
        where = f"$.units[{index}]"
        objective_ids = [str(o.get("objective_id", "")) for o in _as_dicts(unit.get("objectives"))]
        for repeated in _duplicates(objective_ids):
            issues.append(
                Issue("EC102", where, f"objective_id {repeated!r} is declared twice in this unit")
            )
        statements = [
            str(o.get("statement", "")).strip() for o in _as_dicts(unit.get("objectives"))
        ]
        for repeated in _duplicates(statements):
            issues.append(
                Issue("EC112", where, f"objective statement repeated in this unit: {repeated!r}")
            )

    term_ids = [str(term.get("term_id", "")) for term in terms]
    for repeated in _duplicates(term_ids):
        issues.append(Issue("EC103", "$.glossary", f"term_id {repeated!r} is declared twice"))
    known_terms = set(term_ids)
    folded: dict[str, str] = {}
    for term in terms:
        text = str(term.get("term", "")).strip().casefold()
        term_id = str(term.get("term_id", ""))
        if text in folded and folded[text] != term_id:
            issues.append(
                Issue(
                    "EC111",
                    "$.glossary",
                    f"{text!r} is defined twice under different ids "
                    f"({folded[text]!r} and {term_id!r})",
                )
            )
        folded.setdefault(text, term_id)

    pairs: list[tuple[str, str]] = []
    hard_pairs: list[tuple[str, str]] = []
    soft_pairs: list[tuple[str, str]] = []
    for index, edge in enumerate(edges):
        where = f"$.prerequisite_edges[{index}]"
        source = str(edge.get("from_unit", ""))
        target = str(edge.get("to_unit", ""))
        kind = str(edge.get("kind", ""))
        if source not in known_units:
            issues.append(Issue("EC106", where, f"from_unit {source!r} is not declared in $.units"))
        if target not in known_units:
            issues.append(Issue("EC106", where, f"to_unit {target!r} is not declared in $.units"))
        if source == target:
            issues.append(Issue("EC105", where, f"{source!r} cannot require itself"))
        if (source, target) in pairs:
            issues.append(Issue("EC104", where, f"edge {source!r} -> {target!r} is declared twice"))
        pairs.append((source, target))
        (hard_pairs if kind == "hard" else soft_pairs).append((source, target))

    for pair in sorted(set(hard_pairs) & set(soft_pairs)):
        issues.append(
            Issue(
                "EC109",
                "$.prerequisite_edges",
                f"{pair[0]!r} -> {pair[1]!r} is both hard and soft; pick one",
            )
        )

    for node in _cycle_members([pair for pair in hard_pairs if pair[0] != pair[1]]):
        issues.append(
            Issue(
                "EC107", "$.prerequisite_edges", f"hard prerequisites form a cycle through {node!r}"
            )
        )

    inbound: dict[str, set[str]] = defaultdict(set)
    for source, target in hard_pairs:
        inbound[target].add(source)
    for index, unit in enumerate(units):
        unit_id = str(unit.get("unit_id", ""))
        declared = {str(value) for value in unit.get("prerequisites", [])}
        expected = inbound.get(unit_id, set())
        if declared != expected:
            issues.append(
                Issue(
                    "EC108",
                    f"$.units[{index}].prerequisites",
                    f"{unit_id!r} declares {sorted(declared)} but its inbound hard edges are "
                    f"{sorted(expected)}",
                )
            )

    for index, unit in enumerate(units):
        for term_id in unit.get("glossary_terms", []):
            if str(term_id) not in known_terms:
                issues.append(
                    Issue(
                        "EC110",
                        f"$.units[{index}].glossary_terms",
                        f"{str(term_id)!r} is not declared in $.glossary",
                    )
                )
    return issues


def run_cross_rules(documents: dict[str, dict[str, Any]]) -> list[Issue]:
    """Return violations that only appear when documents are read together.

    ``documents`` maps a repository identity to its parsed bundle. Two repositories
    may not claim the same unit id, and a shared ``term_id`` must mean the same
    thing -- that is the whole reason the group shares a term schema.
    """
    issues: list[Issue] = []
    units_by_id: dict[str, str] = {}
    definitions: dict[str, tuple[str, str]] = {}
    for repository, document in sorted(documents.items()):
        for unit in _as_dicts(document.get("units")):
            unit_id = str(unit.get("unit_id", ""))
            owner = units_by_id.setdefault(unit_id, repository)
            if owner != repository:
                issues.append(
                    Issue(
                        "EC120",
                        "$.units",
                        f"unit_id {unit_id!r} is claimed by both {owner!r} and {repository!r}",
                    )
                )
        for term in _as_dicts(document.get("glossary")):
            term_id = str(term.get("term_id", ""))
            definition = str(term.get("definition", "")).strip()
            previous = definitions.setdefault(term_id, (definition, repository))
            if previous[0] != definition and previous[1] != repository:
                issues.append(
                    Issue(
                        "EC121",
                        "$.glossary",
                        f"term_id {term_id!r} means different things in "
                        f"{previous[1]!r} and {repository!r}",
                    )
                )
    return issues
