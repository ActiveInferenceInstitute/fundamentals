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

from .rules import run_cross_rules, run_rules
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
    documents: tuple[dict[str, Any], ...] = field(default=(), repr=False)


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
        if (
            not isinstance(member_id, str)
            or not member_id
            or Path(member_id).name != member_id
            or member_id in (".", "..")
            or not isinstance(repository, str)
            or not repository
        ):
            raise ValueError(f"{path}: members[{index}] needs a safe id and repository")
        if any(member.id == member_id or member.repository == repository for member in members):
            raise ValueError(f"{path}: duplicate member identity {member_id!r}")
        for name in ("pointer", "schema_path"):
            if name in entry and not isinstance(entry[name], str):
                raise ValueError(f"{path}: members[{index}].{name} must be a string")
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


def contained_member_path(repository_dir: Path, relative: str) -> Path:
    """Resolve a declared member path without admitting absolute or escaped paths."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("member path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"member path is not repository-relative: {relative!r}")
    root = repository_dir.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"member path escapes supplied repository: {relative!r}")
    return resolved


def check_member(
    member_root: Path | None,
    spec: MemberSpec,
    *,
    canonical_schema: Path,
    repo_root: Path | None = None,
) -> MemberState:
    """Check one actual checkout.

    ``repo_root`` names the exact member checkout, including an isolated worktree.
    When absent, existing group callers retain ``member_root / spec.id`` discovery.
    No sibling or canonical path is consulted when ``repo_root`` is supplied.
    All pointer, schema and bundle paths must resolve inside this checkout.
    """
    if repo_root is None:
        if member_root is None:
            raise ValueError("supply repo_root or the member checkout parent")
        try:
            repository_dir = contained_member_path(Path(member_root), spec.id)
        except (OSError, ValueError, RuntimeError) as error:
            return MemberState(
                spec.id,
                spec.repository,
                DIVERGED,
                str(error),
                (Issue("EC209", spec.id, str(error)),),
            )
    else:
        repository_dir = Path(repo_root).resolve()
    if not repository_dir.is_dir():
        return MemberState(
            spec.id, spec.repository, UNAVAILABLE, f"no checkout at {repository_dir}"
        )

    issues: list[Issue] = []
    try:
        pointer_path = contained_member_path(repository_dir, spec.pointer)
        vendored_schema = contained_member_path(repository_dir, spec.schema_path)
    except (OSError, ValueError, RuntimeError) as error:
        return MemberState(
            spec.id,
            spec.repository,
            DIVERGED,
            str(error),
            (Issue("EC209", spec.id, str(error)),),
        )
    expected_digest = schema_digest(canonical_schema)
    if not pointer_path.exists():
        if vendored_schema.exists():
            return MemberState(
                spec.id,
                spec.repository,
                DIVERGED,
                "adoption pointer is missing",
                (Issue("EC201", spec.pointer, "adoption pointer is missing"),),
            )
        return MemberState(spec.id, spec.repository, PENDING, f"no {spec.pointer} in {spec.id}")
    try:
        pointer = _read_json(pointer_path)
    except (OSError, ValueError) as error:
        return MemberState(
            spec.id,
            spec.repository,
            DIVERGED,
            f"{spec.pointer} is unreadable",
            (Issue("EC202", spec.pointer, str(error)),),
        )
    pointer_issues = validate_pointer(pointer, canonical_schema)
    issues.extend(pointer_issues)
    if isinstance(pointer, dict) and pointer.get("schema_sha256") != expected_digest:
        issues.append(Issue("EC203", spec.pointer, "adopted schema digest is not current"))
    try:
        if not vendored_schema.is_file():
            issues.append(Issue("EC204", spec.schema_path, "vendored schema copy is missing"))
        elif vendored_schema.read_bytes() != canonical_schema.read_bytes():
            issues.append(Issue("EC205", spec.schema_path, "vendored schema bytes differ"))
    except OSError as error:
        issues.append(Issue("EC205", spec.schema_path, str(error)))

    bundles: list[str] = []
    documents: list[dict[str, Any]] = []
    declared = pointer.get("bundles") if isinstance(pointer, dict) else None
    for entry in declared if isinstance(declared, list) else []:
        if not isinstance(entry, str):
            continue  # Pointer schema already reports this malformed declaration.
        bundles.append(entry)
        try:
            bundle_path = contained_member_path(repository_dir, entry)
            if not bundle_path.is_file():
                issues.append(Issue("EC206", entry, "declared bundle is missing"))
                continue
            document = _read_json(bundle_path)
        except (OSError, ValueError, RuntimeError) as error:
            issues.append(Issue("EC207", entry, str(error)))
            continue
        structural = validate_document(document, canonical_schema)
        issues.extend(
            Issue(issue.code, f"{entry}:{issue.location}", issue.message) for issue in structural
        )
        if structural:
            continue
        documents.append(document)
        issues.extend(
            Issue(issue.code, f"{entry}:{issue.location}", issue.message)
            for issue in run_rules(document)
        )
    state = ADOPTED if not issues else DIVERGED
    detail = (
        f"adopted {len(bundles)} bundle(s)"
        if state == ADOPTED
        else f"{len(issues)} contract violation(s)"
    )
    return MemberState(
        spec.id,
        spec.repository,
        state,
        detail,
        tuple(issues),
        tuple(bundles),
        tuple(documents),
    )


def check_contract(
    registry: Registry,
    *,
    member_root: Path,
    canonical_schema: Path,
    repo_root: Path,
    member_paths: dict[str, Path] | None = None,
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
    try:
        contract_doc = contained_member_path(repo_root, registry.contract_doc)
        if not contract_doc.is_file():
            raise ValueError(f"contract_doc {registry.contract_doc!r} is missing")
    except (OSError, ValueError, RuntimeError) as error:
        registry_issues.append(Issue("EC213", REGISTRY_PATH.name, str(error)))

    overrides = member_paths or {}
    unknown = set(overrides) - {spec.id for spec in registry.members}
    if unknown:
        registry_issues.append(
            Issue("EC214", REGISTRY_PATH.name, f"unknown member roots: {sorted(unknown)}")
        )
    members = tuple(
        check_member(
            member_root,
            spec,
            canonical_schema=canonical_schema,
            repo_root=overrides.get(spec.id),
        )
        for spec in registry.members
    )
    registry_issues.extend(
        run_cross_rules({member.repository: list(member.documents) for member in members})
    )
    return ContractReport(tuple(registry_issues), members)
