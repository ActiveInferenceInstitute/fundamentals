"""The member registry and the copy-with-integrity-check drift test.

The canonical schema lives once, in this repository. Every member that adopts the
contract vendors an identical copy at ``specs/education/schema.json`` and commits an
``education-contract.json`` pointer naming the digest it adopted. This module reads a
member checkout and answers one question per member: has it adopted the current
contract (``adopted``), not yet (``pending``), or adopted something else
(``diverged``)? A diverged member fails the gate; a pending member is reported and
only fails under ``--strict``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rules import run_rules
from .schema import Issue, schema_digest, validate_document, validate_pointer

REGISTRY_PATH = Path("specs") / "education" / "contract.json"
REGISTRY_SCHEMA = "education-contract-registry/v1"
POINTER_NAME = "education-contract.json"
MEMBER_SCHEMA_PATH = Path("specs") / "education" / "schema.json"

PENDING = "pending"
ADOPTED = "adopted"
DIVERGED = "diverged"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MemberSpec:
    """One member repository as named by the registry."""

    id: str
    repository: str
    pointer: str = POINTER_NAME
    schema_path: str = str(MEMBER_SCHEMA_PATH)


@dataclass(frozen=True)
class Registry:
    """The registry file, plus the canonical identities it asserts."""

    path: Path
    contract: str
    schema_sha256: str
    contract_doc: str
    members: tuple[MemberSpec, ...]


@dataclass(frozen=True)
class MemberState:
    """What one member checkout says about its adoption of the contract."""

    id: str
    repository: str
    state: str
    detail: str
    issues: tuple[Issue, ...] = ()
    bundles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractReport:
    """The whole-group verdict, plus the registry's own integrity problems."""

    registry_issues: tuple[Issue, ...] = ()
    members: tuple[MemberState, ...] = field(default_factory=tuple)

    def by_state(self, state: str) -> tuple[MemberState, ...]:
        return tuple(member for member in self.members if member.state == state)

    @property
    def failures(self) -> tuple[MemberState, ...]:
        return self.by_state(DIVERGED)

    @property
    def ok(self) -> bool:
        return not self.registry_issues and not self.failures


def load_registry(path: Path) -> Registry:
    """Parse the registry file and reject a shape this module cannot trust."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    if data.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"{path}: schema must be {REGISTRY_SCHEMA!r}, found {data.get('schema')!r}"
        )
    raw_members = data.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(f"{path}: members must be a non-empty array")
    members: list[MemberSpec] = []
    for index, entry in enumerate(raw_members):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: members[{index}] must be an object")
        member_id = entry.get("id")
        repository = entry.get("repository")
        if not isinstance(member_id, str) or not isinstance(repository, str):
            raise ValueError(f"{path}: members[{index}] needs string id and repository")
        members.append(
            MemberSpec(
                id=member_id,
                repository=repository,
                pointer=str(entry.get("pointer", POINTER_NAME)),
                schema_path=str(entry.get("schema_path", MEMBER_SCHEMA_PATH)),
            )
        )
    return Registry(
        path=path,
        contract=str(data.get("contract", "")),
        schema_sha256=str(data.get("schema_sha256", "")),
        contract_doc=str(data.get("contract_doc", "")),
        members=tuple(members),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_member(member_root: Path, spec: MemberSpec, *, canonical_schema: Path) -> MemberState:
    """Return this member's adoption state, with the exact divergence when there is one."""
    repository_dir = member_root / spec.id
    if not repository_dir.is_dir():
        return MemberState(
            spec.id,
            spec.repository,
            UNAVAILABLE,
            f"no checkout at {repository_dir}",
        )

    pointer_path = repository_dir / spec.pointer
    vendored_schema = repository_dir / spec.schema_path
    expected_digest = schema_digest(canonical_schema)

    if not pointer_path.is_file():
        if vendored_schema.is_file():
            return MemberState(
                spec.id,
                spec.repository,
                DIVERGED,
                f"{spec.schema_path} is vendored but {spec.pointer} is missing",
                (Issue("EC201", spec.pointer, "adoption pointer is missing"),),
            )
        return MemberState(
            spec.id,
            spec.repository,
            PENDING,
            f"no {spec.pointer} in {spec.id}",
        )

    issues: list[Issue] = []
    try:
        pointer = _read_json(pointer_path)
    except (OSError, ValueError) as error:
        return MemberState(
            spec.id,
            spec.repository,
            DIVERGED,
            f"{spec.pointer} is unreadable: {error}",
            (Issue("EC202", spec.pointer, str(error)),),
        )

    issues.extend(validate_pointer(pointer, canonical_schema))
    if isinstance(pointer, dict) and pointer.get("schema_sha256") != expected_digest:
        issues.append(
            Issue(
                "EC203",
                spec.pointer,
                f"adopted schema digest {pointer.get('schema_sha256')!r} is not the current "
                f"{expected_digest}",
            )
        )

    if not vendored_schema.is_file():
        issues.append(Issue("EC204", spec.schema_path, "vendored schema copy is missing"))
    elif vendored_schema.read_bytes() != canonical_schema.read_bytes():
        issues.append(
            Issue(
                "EC205",
                spec.schema_path,
                f"vendored schema digest {schema_digest(vendored_schema)} is not the current "
                f"{expected_digest}",
            )
        )

    bundles: list[str] = []
    declared = pointer.get("bundles") if isinstance(pointer, dict) else None
    for entry in declared if isinstance(declared, list) else []:
        relative = str(entry)
        bundles.append(relative)
        bundle_path = repository_dir / relative
        if not bundle_path.is_file():
            issues.append(Issue("EC206", relative, "declared bundle is missing"))
            continue
        try:
            document = _read_json(bundle_path)
        except (OSError, ValueError) as error:
            issues.append(Issue("EC207", relative, f"bundle is unreadable: {error}"))
            continue
        if not isinstance(document, dict):
            issues.append(Issue("EC208", relative, "bundle must hold a JSON object"))
            continue
        issues.extend(validate_document(document, canonical_schema))
        issues.extend(run_rules(document))

    state = ADOPTED if not issues else DIVERGED
    detail = (
        f"adopted {len(bundles)} bundle(s)"
        if state == ADOPTED
        else f"{len(issues)} contract violation(s)"
    )
    return MemberState(spec.id, spec.repository, state, detail, tuple(issues), tuple(bundles))


def check_contract(
    registry: Registry,
    *,
    member_root: Path,
    canonical_schema: Path,
    repo_root: Path,
) -> ContractReport:
    """Check the registry's own integrity, then every member it names."""
    registry_issues: list[Issue] = []
    if registry.contract != "education-contract/v1":
        registry_issues.append(
            Issue(
                "EC210",
                REGISTRY_PATH.name,
                f"contract must be 'education-contract/v1', found {registry.contract!r}",
            )
        )
    expected_digest = schema_digest(canonical_schema)
    if registry.schema_sha256 != expected_digest:
        registry_issues.append(
            Issue(
                "EC212",
                REGISTRY_PATH.name,
                f"schema_sha256 is {registry.schema_sha256!r} but "
                f"{canonical_schema} hashes to {expected_digest}",
            )
        )
    if not (repo_root / registry.contract_doc).is_file():
        registry_issues.append(
            Issue("EC213", REGISTRY_PATH.name, f"contract_doc {registry.contract_doc!r} is missing")
        )

    members = tuple(
        check_member(member_root, spec, canonical_schema=canonical_schema)
        for spec in registry.members
    )
    return ContractReport(tuple(registry_issues), members)
