"""Offline tests for the pin-binding checks in ``scripts/validate_gnn_spec.py``.

No test here touches the network or an installed GNN runtime. Remote tag
resolution is exercised against a stubbed ``git ls-remote``; the checkout path
uses a throwaway local git repository.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
OTHER = "b" * 40


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_gnn_spec_under_test", ROOT / "scripts" / "validate_gnn_spec.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vgs = _load()


def _pin(**overrides: object) -> dict:
    pin = {
        "repository": "https://example.invalid/Generalized_Notation_Notation",
        "tag": "v9.9.9",
        "commit": COMMIT,
        "language_version": "GNN v1.1",
        "parser_files": {},
    }
    pin.update(overrides)
    return pin


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "gnn"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x", encoding="utf-8")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "c")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_local_tag_resolves_to_the_pinned_commit(checkout: tuple[Path, str]) -> None:
    repo, head = checkout
    _git(repo, "tag", "-a", "-m", "annotated", "v9.9.9")
    binding = vgs.tag_binding(repo, _pin(commit=head))

    assert binding["status"] == "verified"
    assert binding["resolved"] == head  # the peeled commit, not the tag object
    assert vgs.tag_failures(binding, _pin(commit=head)) == []


def test_local_tag_pointing_elsewhere_is_a_named_failure(checkout: tuple[Path, str]) -> None:
    repo, head = checkout
    _git(repo, "tag", "v9.9.9")
    pin = _pin(commit=OTHER)
    binding = vgs.tag_binding(repo, pin)

    assert binding["status"] == "verified"
    failures = vgs.tag_failures(binding, pin)
    assert len(failures) == 1
    assert failures[0].startswith(vgs.TAG_COMMIT_MISMATCH)
    assert head in failures[0] and OTHER in failures[0]


def test_missing_local_tag_is_unresolved_not_verified(checkout: tuple[Path, str]) -> None:
    repo, _head = checkout
    binding = vgs.tag_binding(repo, _pin())

    assert binding["status"] == "unresolvable-offline"
    assert vgs.tag_failures(binding, _pin()) == []


def test_without_a_store_or_remote_the_binding_is_explicitly_unresolved() -> None:
    binding = vgs.tag_binding(None, _pin())

    assert binding["status"] == "unresolvable-offline"
    assert binding["resolved"] is None


def _stub_ls_remote(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> list:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="boom")

    monkeypatch.setattr(vgs.subprocess, "run", fake_run)
    return calls


def test_remote_prefers_the_peeled_commit_of_an_annotated_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_ls_remote(
        monkeypatch, f"{OTHER}\trefs/tags/v9.9.9\n{COMMIT}\trefs/tags/v9.9.9^{{}}\n"
    )
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert calls and calls[0][:3] == ["git", "ls-remote", "--tags"]
    assert binding["status"] == "verified"
    assert binding["resolved"] == COMMIT
    assert vgs.tag_failures(binding, _pin()) == []


def test_remote_tag_resolving_elsewhere_fails_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, f"{OTHER}\trefs/tags/v9.9.9\n")
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert [f.split(":")[0] for f in vgs.tag_failures(binding, _pin())] == [vgs.TAG_COMMIT_MISMATCH]


def test_remote_ignores_look_alike_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, f"{OTHER}\trefs/tags/v9.9.9-rc1\n")
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert binding["status"] == "unresolvable-offline"
    assert "not listed" in binding["detail"]


def test_remote_failure_is_unresolved_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, "", returncode=128)
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert binding["status"] == "unresolvable-offline"
    assert "ls-remote failed" in binding["detail"]


def test_installed_revision_mismatch_and_conflict_fail() -> None:
    pin = _pin()
    good = {"status": "verified", "commit": COMMIT}
    bad = {"status": "verified", "commit": OTHER}
    conflict = {"status": "conflicting", "commit": None, "detail": "two commits"}
    unknown = {"status": "unresolvable-offline", "commit": None}

    assert vgs.revision_failures(good, pin) == []
    assert vgs.revision_failures(unknown, pin) == []
    assert vgs.revision_failures(bad, pin)[0].startswith(vgs.TAG_COMMIT_MISMATCH)
    assert vgs.revision_failures(conflict, pin)[0].startswith(vgs.TAG_COMMIT_MISMATCH)


def _run_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pin: dict, *extra: str
) -> tuple[int, list[str]]:
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    probed: list[str] = []

    def fake_probe(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        probed.append("probe")
        raise AssertionError("the parser must not run before the digests verify")

    monkeypatch.setattr(vgs, "run_probe", fake_probe)
    code = vgs.main(
        [
            "--gnn-runtime",
            sys.executable,
            "--pin",
            str(pin_path),
            "--receipt",
            str(tmp_path / "receipt.json"),
            *extra,
        ]
    )
    return code, probed


def test_digests_are_verified_before_the_document_is_parsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "gnn"
    package.mkdir()
    (package / "__init__.py").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(vgs, "_locate_package", lambda *_a, **_k: (package, str(package / "__init__.py")))
    pin = _pin(parser_files={"src/gnn/__init__.py": "0" * 64})

    code, probed = _run_main(monkeypatch, tmp_path, pin)

    assert code == 2
    assert probed == []
    assert "digest mismatch" in capsys.readouterr().err


def test_a_missing_pinned_package_is_not_run_before_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def absent(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise vgs.Unavailable("the pinned gnn package is not installed")

    monkeypatch.setattr(vgs, "_locate_package", absent)
    code, probed = _run_main(monkeypatch, tmp_path, _pin())

    assert code == 2
    assert probed == []


def test_verify_tag_remote_that_cannot_answer_is_not_run_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_ls_remote(monkeypatch, "", returncode=128)
    code, probed = _run_main(monkeypatch, tmp_path, _pin(), "--verify-tag-remote")

    assert code == 2
    assert probed == []
    assert "--verify-tag-remote" in capsys.readouterr().err


def _verified_tag(pin: dict) -> dict:
    return {
        "status": "verified",
        "tag": pin["tag"],
        "commit": pin["commit"],
        "resolved": pin["commit"],
        "detail": "stub remote",
    }


def _probe_payload() -> dict:
    """Return the payload keys ``main`` reads once conformance is stubbed out."""
    return {
        "gnn_version": "9.9.9",
        "gnn_module": "/stub/gnn/__init__.py",
        "parser": {"success": True, "version": "1.1"},
        "semantic": {"is_valid": True},
        "orientation": {"tensors": []},
        "consistency": {"consistency_score": 1.0},
    }


def test_provenance_requires_a_verified_tag_and_revision() -> None:
    pin = _pin()
    tag = _verified_tag(pin)
    verified = {"status": "verified", "commit": pin["commit"]}
    unknown = {"status": "unresolvable-offline", "commit": None}

    assert vgs.provenance_failures(tag, verified) == []
    (unverified_tag,) = vgs.provenance_failures(None, verified)
    assert unverified_tag.startswith(vgs.PROVENANCE_UNVERIFIED)
    (unverified_revision,) = vgs.provenance_failures(tag, unknown)
    assert "revision" in unverified_revision


def test_unresolved_runtime_revision_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pin = _pin()
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    package = tmp_path / "gnn"
    package.mkdir()

    monkeypatch.setattr(
        vgs, "_locate_package", lambda *_a, **_k: (package, str(package / "__init__.py"))
    )
    monkeypatch.setattr(vgs, "_verify_digests", lambda *_a, **_k: {})
    monkeypatch.setattr(
        vgs,
        "_installed_revision",
        lambda *_a, **_k: {
            "status": "unresolvable-offline",
            "commit": None,
            "version": "9.9.9",
            "detail": "the installer recorded no VCS commit",
        },
    )
    monkeypatch.setattr(vgs, "tag_binding", lambda *_a, **_k: _verified_tag(pin))
    monkeypatch.setattr(vgs, "run_probe", lambda *_a, **_k: _probe_payload())
    monkeypatch.setattr(vgs, "conformance_failures", lambda *_a, **_k: [])

    code = vgs.main(
        [
            "--gnn-runtime",
            sys.executable,
            "--pin",
            str(pin_path),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ]
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))

    assert code == 2
    assert receipt["ok"] is False
    assert receipt["revision"]["status"] == "unresolvable-offline"
    assert receipt["provenance"]["verified"] is False
    assert any("revision" in f for f in receipt["provenance"]["failures"])
    assert "gnn provenance not verified" in capsys.readouterr().err


def test_non_git_source_tree_is_unverified_not_a_git_revision(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    parser = tree / "src" / "gnn" / "__init__.py"
    parser.parent.mkdir(parents=True)
    parser.write_text("parser", encoding="utf-8")
    digest = hashlib.sha256(parser.read_bytes()).hexdigest()
    pin = _pin(parser_files={"src/gnn/__init__.py": digest})

    source, origin, revision = vgs.materialize_pin(tree, pin)

    assert source == tree
    assert "no git metadata" in origin
    assert revision["status"] == "unverified"
    assert revision["commit"] is None
    assert "git archive" not in revision["detail"]
    assert vgs.provenance_failures(_verified_tag(pin), revision)


def test_git_checkout_is_attested_as_the_pinned_revision(checkout: tuple[Path, str]) -> None:
    repo, head = checkout
    pin = _pin(commit=head, parser_files={})

    source, origin, revision = vgs.materialize_pin(repo, pin)

    assert source != repo
    assert origin == f"git archive of {head}"
    assert revision["status"] == "verified"
    assert revision["commit"] == head
    assert vgs.provenance_failures(_verified_tag(pin), revision) == []


def test_non_git_root_ends_unverified_not_as_a_git_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pin = _pin(parser_files={})
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    tree = tmp_path / "tree"
    (tree / "src" / "gnn").mkdir(parents=True)
    (tree / "src" / "gnn" / "__init__.py").write_text("parser", encoding="utf-8")
    monkeypatch.setattr(vgs, "run_probe", lambda *_a, **_k: _probe_payload())
    monkeypatch.setattr(vgs, "conformance_failures", lambda *_a, **_k: [])

    code = vgs.main(
        [
            "--gnn-root",
            str(tree),
            "--python",
            sys.executable,
            "--pin",
            str(pin_path),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ]
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))

    assert code == 2
    assert receipt["ok"] is False
    assert "no git metadata" in receipt["pin_source"]
    assert receipt["revision"]["status"] == "unverified"
    assert "git archive" not in receipt["revision"]["detail"]
    assert receipt["tag_binding"]["status"] == "unresolvable-offline"
    assert receipt["provenance"]["verified"] is False


def test_verified_tag_does_not_make_a_non_git_tree_a_git_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pin = _pin(parser_files={})
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    tree = tmp_path / "tree"
    (tree / "src" / "gnn").mkdir(parents=True)
    (tree / "src" / "gnn" / "__init__.py").write_text("parser", encoding="utf-8")
    monkeypatch.setattr(vgs, "tag_binding", lambda *_a, **_k: _verified_tag(pin))
    monkeypatch.setattr(vgs, "run_probe", lambda *_a, **_k: _probe_payload())
    monkeypatch.setattr(vgs, "conformance_failures", lambda *_a, **_k: [])

    code = vgs.main(
        [
            "--gnn-root",
            str(tree),
            "--python",
            sys.executable,
            "--verify-tag-remote",
            "--pin",
            str(pin_path),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ]
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))

    # The upstream tag is proven, but the parsed source tree is not a git
    # revision; the two facts must not be conflated.
    assert receipt["tag_binding"]["status"] == "verified"
    assert receipt["revision"]["status"] == "unverified"
    assert code == 2
    assert receipt["ok"] is False
    assert receipt["provenance"]["verified"] is False


class TestSelectedDocument:
    @staticmethod
    def document():
        sections = {
            "GNNVersionAndFlags": "GNN v1.1",
            "ModelName": "SelectedAlternate",
            "StateSpaceBlock": "A[2,2,type=float]\nD[2,type=float]",
            "InitialParameterization": "A = {0.6, 0.3, 0.4, 0.7}\nD = {0.2, 0.8}",
        }
        return "\n".join(
            f"## {name}\n{sections.get(name, '')}\n" for name in vgs.REQUIRED_SECTIONS
        )

    @staticmethod
    def payload():
        return {
            "gnn_version": "9.9.9",
            "gnn_module": "/synthetic/gnn/__init__.py",
            "parser": {
                "success": True, "errors": [], "warnings": [],
                "sections": list(vgs.REQUIRED_SECTIONS),
                "model_name": "SelectedAlternate", "version": "1.1",
                "variables": [
                    {"name": "A", "dimensions": [2, 2]},
                    {"name": "D", "dimensions": [2]},
                ],
                "parameters": {"A": [0.6, 0.3, 0.4, 0.7], "D": [0.2, 0.8]},
            },
            "semantic": {"is_valid": True},
            "orientation": {"tensors": []},
            "consistency": {"consistent": True, "consistency_score": 1.0},
        }

    def test_alternate_model_and_parameters_are_compared_to_selected_text(self):
        payload = self.payload()
        assert vgs.conformance_failures(
            payload, spec_text=self.document(), pin=_pin(), source=None,
        ) == []
        payload["parser"]["parameters"]["D"] = [0.8, 0.2]
        failures = vgs.conformance_failures(
            payload, spec_text=self.document(), pin=_pin(), source=None,
        )
        assert any("parameter D differs" in failure for failure in failures)

    def test_cli_binds_captured_bytes_despite_selected_file_replacement(
        self, tmp_path, monkeypatch,
    ):
        # Inject unavailable provenance and a concurrent input replacement;
        # this tests document custody, not authenticity of an upstream parser.
        selected = tmp_path / "alternate.gnn.md"
        original = self.document().encode()
        selected.write_bytes(original)
        pin_path = tmp_path / "pin.json"
        pin_path.write_text(json.dumps(_pin()))
        receipt_path = tmp_path / "receipt.json"
        package = tmp_path / "gnn"
        package.mkdir()
        for name in ("GNN_ROOT", "GNN_TAG_CHECKOUT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(vgs, "_locate_package", lambda *a, **k: (package, package / "__init__.py"))
        monkeypatch.setattr(
            vgs, "_installed_revision",
            lambda *a, **k: {
                "status": "unresolvable-offline", "commit": None, "detail": "test unavailable",
            },
        )

        def replace_selected(_python, spec, _output, *, source):
            selected.write_text("replaced after capture")
            assert spec.read_bytes() == original
            result = self.payload()
            result["gnn_module"] = str(package / "__init__.py")
            return result

        monkeypatch.setattr(vgs, "run_probe", replace_selected)
        assert vgs.main([
            "--gnn-runtime", sys.executable, "--spec", str(selected),
            "--pin", str(pin_path), "--receipt", str(receipt_path),
        ]) == 2
        receipt = json.loads(receipt_path.read_text())
        assert receipt["spec"] == str(selected)
        assert receipt["spec_sha256"] == hashlib.sha256(original).hexdigest()
        assert receipt["failures"] == []
        assert receipt["ok"] is False
