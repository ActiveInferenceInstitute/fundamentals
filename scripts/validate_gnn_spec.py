"""Validate the committed GNN specification with the pinned upstream parser.

The canonical GNN pin is recorded in ``specs/gnn-pin.json`` (repository, tag,
commit, and SHA-256 digests of the parser files). This script supports two
verified sources for that pin:

* a GNN git checkout (``--gnn-root`` or ``GNN_ROOT``): the pinned commit is
  materialized with ``git archive`` and every recorded digest is verified, so
  the live working tree is never imported;
* an installed pinned runtime (``--gnn-runtime`` or ``GNN_RUNTIME_PYTHON``): an
  interpreter whose ``gnn`` package is installed at the pinned revision; the
  installed parser file digests are verified *before* the document is parsed,
  and the distribution revision recorded by the installer is compared with the
  pin.

Both sources also bind the pin's ``tag`` to its ``commit`` where an object store
is available: a GNN checkout resolves the tag locally (``--gnn-root``),
``--verify-tag-remote`` resolves it upstream with ``git ls-remote``. A binding
that could not be verified — no object store, no remote answer, a tag that
resolves elsewhere, or an install with no recorded VCS commit — is reported in
the receipt and ends the run as *unverified* (``ok: false``, exit 2), never a
pass. The same holds for a ``--gnn-root`` directory without git metadata: its
parser files must match the pin's digests, but it is attested as a
digest-verified tree, not as the pinned Git revision.

Either way the document is parsed by ``scripts/gnn_parser_probe.py`` in a
subprocess and checked for conformance: the pinned parser's view of the
declared sections, variables, model name, and ``A``/``D`` parameters must agree
with the dependency-free reader in ``active_inference.gnn_spec``, and the
pinned semantic validator and consistency checker must accept the document.
The binding of the declaration to the implementation lives in
``tests/test_gnn_spec.py``; this script proves the document is well-formed GNN
and that two independent readers agree on it.

Exit codes: ``0`` pass with the tag and revision verified, ``1`` conformance
failure, ``2`` pinned runtime or checkout unavailable or the tag/revision
binding unverified (the run is then explicitly *not a pass*).

Usage:

    # Local checkout (strongest: archive of the pinned commit)
    GNN_ROOT=/path/to/Generalized_Notation_Notation \\
    GNN_PYTHON=/path/to/parser-runtime/bin/python \\
    uv run python scripts/validate_gnn_spec.py --json

    # Pinned installed runtime (used by CI); --verify-tag-remote binds the tag
    uv run python scripts/validate_gnn_spec.py \\
        --gnn-runtime /path/to/venv/bin/python --verify-tag-remote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.gnn_spec import (  # noqa: E402
    GNN_PIN_PATH,
    GNN_SPEC_PATH,
    REQUIRED_SECTIONS,
    SPEC_MATRIX_KEYS,
    declared_arrays,
    load_pin,
    parse_sections,
    parse_variables,
    spec_summary,
)

PROBE = REPO_ROOT / "scripts" / "gnn_parser_probe.py"
DEFAULT_RECEIPT = REPO_ROOT / "docs" / "_build" / "gnn" / "validation-receipt.json"
PACKAGE_PREFIX = "src/gnn/"


class Unavailable(RuntimeError):
    """Raised when the pinned parser runtime or checkout is not available."""


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of one file, streamed in blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_digests(package_dir: Path, pin: dict[str, Any], *, origin: str) -> dict[str, str]:
    """Verify every recorded parser digest under an installed ``gnn`` package."""

    observed: dict[str, str] = {}
    for relative, expected in sorted(pin["parser_files"].items()):
        suffix = relative[len(PACKAGE_PREFIX) :] if relative.startswith(PACKAGE_PREFIX) else relative
        path = package_dir / suffix
        if not path.is_file():
            raise Unavailable(f"pinned parser file is missing from {origin}: {suffix}")
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            raise Unavailable(
                f"pinned parser file digest mismatch for {suffix} in {origin}: "
                f"expected {expected}, observed {observed[relative]}"
            )
    return observed


#: A named conformance failure: the pin's ``tag`` and ``commit`` name one
#: revision, so a tag that resolves elsewhere is reported by name.
TAG_COMMIT_MISMATCH = "tag-commit-mismatch"

#: A named provenance result: the parsed source was not proven to be the pin's
#: revision. Recorded in the receipt and never surfaced as ``ok: true``.
PROVENANCE_UNVERIFIED = "provenance-unverified"

# Locating the installed package and reading its installer metadata never parses
# a document, so the digest and revision checks can run before the probe.
LOCATE_EXPRESSION = (
    "import importlib.util, json; "
    "spec = importlib.util.find_spec('gnn'); "
    "print(json.dumps({'module': spec.origin if spec else None, "
    "'locations': [str(path) for path in (spec.submodule_search_locations or [])] "
    "if spec else []}))"
)
REVISION_EXPRESSION = """
import importlib.metadata as md
import json

records = []
for name in md.packages_distributions().get("gnn", []):
    record = {"name": name, "version": None, "commit": None, "url": None}
    try:
        distribution = md.distribution(name)
    except md.PackageNotFoundError:
        continue
    record["version"] = distribution.version
    raw = distribution.read_text("direct_url.json")
    if raw:
        try:
            direct = json.loads(raw)
        except ValueError:
            direct = {}
        vcs = direct.get("vcs_info") or {}
        record["commit"] = vcs.get("commit_id")
        record["url"] = direct.get("url")
    records.append(record)
print(json.dumps(records))
"""


def _locate_package(python: str, *, origin: str) -> tuple[Path, str]:
    """Return the ``gnn`` package directory and module file an interpreter imports.

    This asks the interpreter's import system where the package is; it never
    imports the parser or parses a document, so digest verification can run
    before any input is parsed.
    """

    completed = subprocess.run(
        [python, "-c", LOCATE_EXPRESSION],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(f"cannot locate the pinned gnn package in {origin}: {tail}")
    try:
        located = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Unavailable(f"cannot locate the pinned gnn package in {origin}: {exc}") from exc
    module = located.get("module")
    if not module:
        raise Unavailable(
            f"the pinned gnn package is not installed in {origin}; install it from the "
            "pinned revision or pass --gnn-root"
        )
    locations = located.get("locations") or []
    package_dir = Path(locations[0]).resolve() if locations else Path(module).resolve().parent
    return package_dir, str(module)


def _installed_revision(python: str, *, origin: str) -> dict[str, Any]:
    """Return the VCS revision the installer recorded for the ``gnn`` distribution.

    ``direct_url.json`` records ``vcs_info.commit_id`` only when the runtime was
    installed from a git URL. An editable or index install has no commit to
    compare, so the result is reported as unresolvable rather than assumed.
    """

    completed = subprocess.run(
        [python, "-c", REVISION_EXPRESSION],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(f"cannot read the installed gnn metadata in {origin}: {tail}")
    try:
        records = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Unavailable(f"cannot read the installed gnn metadata in {origin}: {exc}") from exc
    commits = sorted({record["commit"] for record in records if record.get("commit")})
    versions = sorted({str(record["version"]) for record in records if record.get("version")})
    if not commits:
        return {
            "status": "unresolvable-offline",
            "commit": None,
            "version": versions[0] if len(versions) == 1 else None,
            "distributions": [record["name"] for record in records],
            "detail": "the installer recorded no VCS commit (not installed from a git URL)",
        }
    if len(commits) > 1:
        return {
            "status": "conflicting",
            "commit": None,
            "version": versions[0] if len(versions) == 1 else None,
            "distributions": [record["name"] for record in records],
            "detail": f"the installer recorded several commits {commits}",
        }
    return {
        "status": "verified",
        "commit": commits[0],
        "version": versions[0] if len(versions) == 1 else None,
        "distributions": [record["name"] for record in records],
        "detail": "direct_url.json vcs_info.commit_id",
    }


def tag_binding(
    gnn_root: Path | None, pin: dict[str, Any], *, remote: bool = False
) -> dict[str, Any]:
    """Bind the pin's ``tag`` to its ``commit`` in an available object store.

    A GNN checkout resolves the tag locally; ``remote`` resolves it upstream with
    ``git ls-remote``. When neither is available the result says so explicitly
    instead of assuming the binding holds.
    """

    tag = pin["tag"]
    record: dict[str, Any] = {
        "status": "unresolvable-offline",
        "tag": tag,
        "commit": pin["commit"],
        "resolved": None,
        "detail": "no GNN object store or remote was supplied",
    }
    if gnn_root is not None and (gnn_root / ".git").exists():
        resolved = subprocess.run(
            ["git", "-C", str(gnn_root), "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if resolved.returncode == 0 and resolved.stdout.strip():
            record["resolved"] = resolved.stdout.strip()
            record["status"] = "verified"
            record["detail"] = f"local object store {gnn_root}"
        else:
            record["detail"] = f"tag {tag} is not present in {gnn_root}"
        return record
    if remote:
        # ``--tags`` lists an annotated tag and its peeled ``^{}`` object only
        # when the pattern matches both, so match the whole tag family and then
        # select the peeled commit.
        resolved = subprocess.run(
            ["git", "ls-remote", "--tags", pin["repository"], f"refs/tags/{tag}*"],
            capture_output=True,
            text=True,
            check=False,
        )
        if resolved.returncode != 0:
            record["detail"] = f"git ls-remote failed: {resolved.stderr.strip()}"
            return record
        peeled = f"refs/tags/{tag}^{{}}"
        pairs = [line.split() for line in resolved.stdout.splitlines() if line.strip()]
        for wanted in (peeled, f"refs/tags/{tag}"):
            for sha, name in pairs:
                if name == wanted:
                    record["resolved"] = sha
                    record["status"] = "verified"
                    record["detail"] = "git ls-remote"
                    return record
        record["detail"] = f"tag {tag} is not listed by {pin['repository']}"
        return record
    return record


def tag_failures(tag: dict[str, Any] | None, pin: dict[str, Any]) -> list[str]:
    """Return the named failure when the pin's tag resolves to a different commit."""

    if tag is not None and tag.get("status") == "verified" and tag.get("resolved") != pin["commit"]:
        return [
            f"{TAG_COMMIT_MISMATCH}: tag {pin['tag']} resolves to {tag.get('resolved')} "
            f"but the pin records {pin['commit']}"
        ]
    return []


def revision_failures(revision: dict[str, Any], pin: dict[str, Any]) -> list[str]:
    """Return failures when the installed runtime's recorded commit is not the pin."""

    if revision.get("status") == "verified" and revision.get("commit") != pin["commit"]:
        return [
            f"{TAG_COMMIT_MISMATCH}: the installed runtime records commit "
            f"{revision.get('commit')} but the pin records {pin['commit']}"
        ]
    if revision.get("status") == "conflicting":
        return [
            f"{TAG_COMMIT_MISMATCH}: the installed runtime metadata is ambiguous: "
            f"{revision.get('detail')}"
        ]
    return []


def provenance_failures(
    tag: dict[str, Any] | None, revision: dict[str, Any] | None
) -> list[str]:
    """Return one named failure for each part of the pin binding left unverified.

    Only ``status == "verified"`` is evidence for the pin. Anything else — no
    object store, no remote answer, no installer-recorded commit, or a source
    tree with no git metadata — leaves the run unverified, so it cannot end with
    ``ok: true`` even when the document itself conforms.
    """

    failures: list[str] = []
    for label, record in (("tag binding", tag), ("revision", revision)):
        record = record or {}
        if record.get("status") != "verified":
            detail = record.get("detail") or "no detail recorded"
            failures.append(
                f"{PROVENANCE_UNVERIFIED}: {label} is "
                f"{record.get('status', 'absent')}: {detail}"
            )
    return failures


def materialize_pin(
    gnn_root: Path, pin: dict[str, Any]
) -> tuple[Path, str, dict[str, Any]]:
    """Return the pinned source tree, how it was obtained, and its revision record.

    A directory with git metadata is materialized from the pinned commit, so its
    revision is verified. A directory without git metadata is only digest-verified:
    its parser files must match the pin, but nothing proves which revision it is,
    so the revision record says so instead of claiming a git archive.
    """

    if not gnn_root.is_dir():
        raise Unavailable(f"GNN checkout is not a directory: {gnn_root}")
    if not (gnn_root / ".git").exists():
        _verify_digests(gnn_root / "src" / "gnn", pin, origin=str(gnn_root))
        return (
            gnn_root,
            f"digest-verified source tree {gnn_root} (no git metadata)",
            {
                "status": "unverified",
                "commit": None,
                "version": None,
                "detail": "the source tree has no git metadata; its parser files match "
                "the pin's digests but no revision was resolved",
            },
        )

    commit = pin["commit"]
    resolved = subprocess.run(
        ["git", "-C", str(gnn_root), "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise Unavailable(f"pinned commit {commit} is not present in {gnn_root}")
    if resolved.stdout.strip() != commit:
        raise Unavailable(
            f"resolved commit {resolved.stdout.strip()} differs from the pin {commit}"
        )

    target = Path(tempfile.mkdtemp(prefix="fundamentals-gnn-pin-"))
    archive = target / "pinned.tar"
    completed = subprocess.run(
        ["git", "-C", str(gnn_root), "archive", "--format=tar", f"--output={archive}", commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Unavailable(f"git archive failed for {commit}: {completed.stderr.strip()}")
    with tarfile.open(archive) as handle:
        try:
            handle.extractall(target, filter="data")
        except TypeError:  # pragma: no cover - Python < 3.12 fallback
            handle.extractall(target)
    archive.unlink()
    _verify_digests(target / "src" / "gnn", pin, origin=f"archive of {commit}")
    return (
        target,
        f"git archive of {commit}",
        {
            "status": "verified",
            "commit": commit,
            "version": None,
            "detail": f"git archive of {commit}",
        },
    )


def run_probe(python: str, spec: Path, output: Path, *, source: Path | None) -> dict[str, Any]:
    """Run the pinned parser probe in a subprocess and return its payload."""

    environment = dict(os.environ)
    if source is not None:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source / "src") + (
            os.pathsep + existing if existing else ""
        )
    completed = subprocess.run(
        [python, str(PROBE), "--spec", str(spec), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(
            f"pinned parser probe failed with {python}: {tail}. Provide an interpreter that "
            "has the GNN parser dependencies (the GNN checkout's own .venv, or a runtime "
            "installed from the pinned revision) via --python/--gnn-runtime."
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _variable_dimensions(parser: dict[str, Any]) -> dict[str, list[int]]:
    """Return the pinned parser's declared dimensions keyed by variable name."""

    return {variable["name"]: list(variable["dimensions"]) for variable in parser["variables"]}


def conformance_failures(
    payload: dict[str, Any],
    *,
    spec_text: str,
    pin: dict[str, Any],
    source: Path | None,
    package_dir: Path | None = None,
    tag: dict[str, Any] | None = None,
) -> list[str]:
    """Return every conformance failure between the document and its readers."""

    failures: list[str] = []
    parser = payload["parser"]
    expected_version = pin["tag"].lstrip("v")
    summary = spec_summary(spec_text)
    arrays = declared_arrays(spec_text)
    variables = parse_variables(spec_text)

    if payload["gnn_version"] != expected_version:
        failures.append(
            f"parser version {payload['gnn_version']} differs from the pinned {expected_version}"
        )
    # C1: the pin's tag and commit are two spellings of one revision; a tag that
    # resolves elsewhere in the object store is a named failure.
    failures.extend(tag_failures(tag, pin))
    if source is not None:
        module = Path(str(payload["gnn_module"])).resolve()
        if not module.is_relative_to((source / "src").resolve()):
            failures.append(f"parser was imported from {module}, outside the pinned tree {source}")
    elif package_dir is not None:
        module = Path(str(payload["gnn_module"])).resolve()
        if not module.is_relative_to(package_dir):
            failures.append(
                f"parser was imported from {module}, outside the verified package {package_dir}"
            )
    if not parser["success"]:
        failures.append("pinned parser reported failure")
    for message in parser["errors"]:
        failures.append(f"parser error: {message}")
    for message in parser["warnings"]:
        failures.append(f"parser warning: {message}")

    parser_sections = set(parser["sections"])
    declared_sections = list(parse_sections(spec_text))
    if parser_sections != set(declared_sections):
        failures.append(
            f"the pinned parser sees sections {sorted(parser_sections)}, the document text "
            f"declares {sorted(declared_sections)}"
        )
    if declared_sections != list(REQUIRED_SECTIONS):
        failures.append(
            f"declared sections {declared_sections} differ from the required order "
            f"{list(REQUIRED_SECTIONS)}"
        )
    if parser["model_name"] != summary["model_name"]:
        failures.append(
            f"parser model name {parser['model_name']!r} differs from the declared "
            f"{summary['model_name']!r}"
        )
    language_version = str(pin["language_version"]).removeprefix("GNN v")
    if str(parser["version"]) != language_version:
        failures.append(
            f"declared language version {parser['version']!r} differs from the pinned "
            f"{language_version!r}"
        )

    dimensions = _variable_dimensions(parser)
    for name, variable in variables.items():
        if dimensions.get(name) != list(variable.dimensions):
            failures.append(
                f"parser dimensions for {name} are {dimensions.get(name)}, the document "
                f"declares {list(variable.dimensions)}"
            )

    for key in SPEC_MATRIX_KEYS:
        value = parser["parameters"].get(key)
        if value is None:
            failures.append(f"the pinned parser did not read parameter {key}")
            continue
        declared = arrays[key]
        try:
            observed = np.asarray(value, dtype=float).reshape(declared.shape)
        except ValueError as exc:
            failures.append(
                f"parser parameter {key} does not reshape to {declared.shape}: {exc}"
            )
            continue
        if not np.array_equal(observed, declared):
            failures.append(f"parser parameter {key} differs from the document's declared {key}")

    semantic = payload["semantic"]
    if not semantic.get("is_valid"):
        failures.append(
            f"pinned semantic validator rejected the document: {semantic.get('errors')}"
        )
    for message in semantic.get("warnings", []):
        failures.append(f"semantic warning: {message}")

    orientation = payload["orientation"]
    for message in orientation.get("warnings", []):
        failures.append(f"B-orientation warning: {message}")
    for tensor in orientation.get("tensors", []):
        if tensor.get("orientation") != "canonical":
            failures.append(
                f"B tensor {tensor.get('variable')} orientation is {tensor.get('orientation')}, "
                "not canonical"
            )

    consistency = payload["consistency"]
    if not consistency.get("consistent"):
        failures.append(f"pinned consistency checker reported: {consistency.get('warnings')}")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Validate the committed specification against the pinned parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gnn-root",
        type=Path,
        default=Path(os.environ["GNN_ROOT"]) if os.environ.get("GNN_ROOT") else None,
        help="Generalized_Notation_Notation checkout (default: $GNN_ROOT)",
    )
    parser.add_argument(
        "--gnn-runtime",
        default=os.environ.get("GNN_RUNTIME_PYTHON"),
        help="interpreter with the pinned GNN distribution installed",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("GNN_PYTHON"),
        help="parser interpreter used with --gnn-root (default: $GNN_PYTHON or this interpreter)",
    )
    parser.add_argument(
        "--gnn-tag-checkout",
        type=Path,
        default=os.environ.get("GNN_TAG_CHECKOUT"),
        help="local Git tag store for the installed parser (default: $GNN_TAG_CHECKOUT)",
    )
    parser.add_argument(
        "--verify-tag-remote",
        action="store_true",
        help="require a verified tag and use git ls-remote when no checkout is supplied",
    )
    parser.add_argument("--spec", type=Path, default=GNN_SPEC_PATH)
    parser.add_argument("--pin", type=Path, default=GNN_PIN_PATH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--json", action="store_true", help="print the receipt to stdout")
    args = parser.parse_args(argv)

    pin = load_pin(args.pin)
    if args.gnn_root is None and not args.gnn_runtime:
        print(
            "gnn validation not run: supply --gnn-root/GNN_ROOT or --gnn-runtime/"
            "GNN_RUNTIME_PYTHON",
            file=sys.stderr,
        )
        return 2

    if args.gnn_tag_checkout is not None and (args.gnn_root is not None or not args.gnn_runtime):
        parser.error("--gnn-tag-checkout requires the installed GNN runtime without --gnn-root")
    tag = tag_binding(
        args.gnn_tag_checkout or args.gnn_root,
        pin,
        remote=args.verify_tag_remote and args.gnn_tag_checkout is None,
    )
    if args.verify_tag_remote and tag["status"] != "verified":
        # A required tag check must fail when the selected source cannot resolve it.
        print(f"gnn validation not run: --verify-tag-remote: {tag['detail']}", file=sys.stderr)
        return 2
    try:
        spec_bytes = args.spec.read_bytes()
        spec_text = spec_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"gnn specification unreadable: {args.spec}: {exc}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="fundamentals-gnn-") as directory:
        output = Path(directory) / "probe.json"
        selected_spec = Path(directory) / "selected.gnn.md"
        selected_spec.write_bytes(spec_bytes)
        try:
            if args.gnn_root is not None:
                source, origin, revision = materialize_pin(args.gnn_root, pin)
                payload = run_probe(
                    args.python or sys.executable, selected_spec, output, source=source
                )
                package_dir = None
            else:
                source = None
                # The digest and revision checks come before the parser runs, so
                # a tampered parser cannot be reached by parsing first.
                package_dir, _module = _locate_package(args.gnn_runtime, origin=args.gnn_runtime)
                _verify_digests(package_dir, pin, origin=args.gnn_runtime)
                revision = _installed_revision(args.gnn_runtime, origin=args.gnn_runtime)
                payload = run_probe(args.gnn_runtime, selected_spec, output, source=None)
                origin = f"installed runtime {args.gnn_runtime} ({package_dir})"
        except Unavailable as exc:
            print(f"gnn validation not run: {exc}", file=sys.stderr)
            return 2

        failures = conformance_failures(
            payload, spec_text=spec_text, pin=pin, source=source,
            package_dir=package_dir, tag=tag,
        )
        failures.extend(revision_failures(revision, pin))

    provenance = provenance_failures(tag, revision)
    receipt = {
        "ok": not failures and not provenance,
        "spec": str(args.spec),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "pin": pin,
        "pin_source": origin,
        "gnn_version": payload["gnn_version"],
        "gnn_module": payload["gnn_module"],
        "language_version": pin["language_version"],
        "parser_language_version": payload["parser"].get("version"),
        "tag_binding": tag,
        "revision": revision,
        "provenance": {"verified": not provenance, "failures": provenance},
        "parser_success": payload["parser"]["success"],
        "semantic_valid": payload["semantic"].get("is_valid"),
        "orientation": [
            tensor.get("orientation") for tensor in payload["orientation"].get("tensors", [])
        ],
        "consistency_score": payload["consistency"].get("consistency_score"),
        "failures": failures,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    if failures:
        print("gnn conformance failures:", file=sys.stderr)
        for message in failures:
            print(f"  {message}", file=sys.stderr)
        return 1
    if provenance:
        # The document conforms, but the source it was parsed by was not proven
        # to be the pin's revision; that is not a pass.
        print("gnn provenance not verified:", file=sys.stderr)
        for message in provenance:
            print(f"  {message}", file=sys.stderr)
        print(f"gnn validation not verified; receipt {args.receipt}", file=sys.stderr)
        return 2
    print(
        f"gnn validation passed: {args.spec} parsed by gnn {payload['gnn_version']} "
        f"({origin}); language version {payload['parser'].get('version')}; "
        f"tag {tag['tag']} -> {tag['status']}; revision {revision['status']} "
        f"({revision.get('detail')}); receipt {args.receipt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
