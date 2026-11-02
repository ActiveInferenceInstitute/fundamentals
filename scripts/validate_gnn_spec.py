"""Validate the committed GNN specification with the pinned upstream parser.

The canonical GNN pin is recorded in ``specs/gnn-pin.json`` (repository, tag,
commit, and SHA-256 digests of the parser files). This script supports two
verified sources for that pin:

* a GNN git checkout (``--gnn-root`` or ``GNN_ROOT``): the pinned commit is
  materialized with ``git archive`` and every recorded digest is verified, so
  the live working tree is never imported;
* an installed pinned runtime (``--gnn-runtime`` or ``GNN_RUNTIME_PYTHON``): an
  interpreter whose ``gnn`` package is installed at the pinned revision; the
  distribution version and the installed parser file digests are verified.

Either way the document is parsed by ``scripts/gnn_parser_probe.py`` in a
subprocess and checked for conformance: the pinned parser's view of the
declared sections, variables, model name, and ``A``/``D`` parameters must agree
with the dependency-free reader in ``active_inference.gnn_spec``, and the
pinned semantic validator and consistency checker must accept the document.
The binding of the declaration to the implementation lives in
``tests/test_gnn_spec.py``; this script proves the document is well-formed GNN
and that two independent readers agree on it.

Exit codes: ``0`` pass, ``1`` conformance failure, ``2`` pinned runtime or
checkout unavailable (the run is then explicitly *not run*, never a pass).

Usage:

    # Local checkout (strongest: archive of the pinned commit)
    GNN_ROOT=/path/to/Generalized_Notation_Notation \\
    GNN_PYTHON=/path/to/parser-runtime/bin/python \\
    uv run python scripts/validate_gnn_spec.py --json

    # Pinned installed runtime (used by CI)
    uv run python scripts/validate_gnn_spec.py --gnn-runtime /path/to/venv/bin/python
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


def materialize_pin(gnn_root: Path, pin: dict[str, Any]) -> tuple[Path, str]:
    """Return a directory holding the pinned source tree and how it was obtained."""

    if not gnn_root.is_dir():
        raise Unavailable(f"GNN checkout is not a directory: {gnn_root}")
    if not (gnn_root / ".git").exists():
        _verify_digests(gnn_root / "src" / "gnn", pin, origin=str(gnn_root))
        return gnn_root, f"digest-verified source tree {gnn_root} (no git metadata)"

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
    return target, f"git archive of {commit}"


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
    pin: dict[str, Any],
    source: Path | None,
) -> list[str]:
    """Return every conformance failure between the document and its readers."""

    failures: list[str] = []
    parser = payload["parser"]
    expected_version = pin["tag"].lstrip("v")
    summary = spec_summary()
    arrays = declared_arrays()
    variables = parse_variables()

    if payload["gnn_version"] != expected_version:
        failures.append(
            f"parser version {payload['gnn_version']} differs from the pinned {expected_version}"
        )
    if source is not None:
        module = Path(str(payload["gnn_module"])).resolve()
        if not module.is_relative_to((source / "src").resolve()):
            failures.append(f"parser was imported from {module}, outside the pinned tree {source}")
    if not parser["success"]:
        failures.append("pinned parser reported failure")
    for message in parser["errors"]:
        failures.append(f"parser error: {message}")
    for message in parser["warnings"]:
        failures.append(f"parser warning: {message}")

    parser_sections = set(parser["sections"])
    declared_sections = list(parse_sections())
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

    with tempfile.TemporaryDirectory(prefix="fundamentals-gnn-") as directory:
        output = Path(directory) / "probe.json"
        try:
            if args.gnn_root is not None:
                source, origin = materialize_pin(args.gnn_root, pin)
                payload = run_probe(
                    args.python or sys.executable, args.spec, output, source=source
                )
            else:
                source = None
                payload = run_probe(args.gnn_runtime, args.spec, output, source=None)
                package_dir = Path(str(payload["gnn_module"])).resolve().parent
                _verify_digests(package_dir, pin, origin=args.gnn_runtime)
                origin = f"installed runtime {args.gnn_runtime} ({package_dir})"
        except Unavailable as exc:
            print(f"gnn validation not run: {exc}", file=sys.stderr)
            return 2

        failures = conformance_failures(payload, pin=pin, source=source)

    receipt = {
        "ok": not failures,
        "spec": str(args.spec),
        "pin": pin,
        "pin_source": origin,
        "gnn_version": payload["gnn_version"],
        "gnn_module": payload["gnn_module"],
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
    print(
        f"gnn validation passed: {args.spec} parsed by gnn {payload['gnn_version']} "
        f"({origin}); receipt {args.receipt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
