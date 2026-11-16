"""Owned-section generation must not rewrite authored extras documentation."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generator():
    path = REPO_ROOT / "scripts/generate_extra_readme_sections.py"
    spec = importlib.util.spec_from_file_location("extra_readme_sections", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def extras_copy(tmp_path):
    return Path(shutil.copytree(REPO_ROOT / "extras", tmp_path / "extras"))


class TestOwnedReadmeGeneration:
    def test_complete_owned_set_and_bespoke_bytes_survive_round_trip(self, generator, extras_copy):
        all_readmes = sorted(extras_copy.glob("*/README.md"))
        before = {path: path.read_bytes() for path in all_readmes}
        owned = {path for path, raw in before.items() if b"## Scripts\n" in raw and b"## Outputs\n" in raw}
        assert len(owned) == 58
        assert {extras_copy / slug / "README.md" for slug in generator.OWNED_TOPICS} == owned
        assert len(set(before) - owned) == 13
        suffix = b"\n## Authored future notes\r\nRetain these exact bytes: \xc3\xa9.\r\n"
        for path in owned:
            path.write_bytes(before[path] + suffix)
        admitted = {path: path.read_bytes() for path in all_readmes}
        assert generator.main(["--extras-root", str(extras_copy), "--check"]) == 1
        assert {path: path.read_bytes() for path in all_readmes} == admitted
        assert generator.main(["--extras-root", str(extras_copy), "--write"]) == 0
        for path in all_readmes:
            after = path.read_bytes()
            if path in owned:
                assert after.startswith(before[path].split(b"## Scripts\n", 1)[0])
                assert after.endswith(suffix.lstrip(b"\n"))
                assert f"`visualize_{path.parent.name}.npz`".encode() in after
                assert f"`visualize_{path.parent.name}.json`".encode() in after
            else:
                assert after == before[path]
        generated = {path: path.read_bytes() for path in all_readmes}
        assert generator.main(["--extras-root", str(extras_copy), "--check"]) == 0
        assert generator.main(["--extras-root", str(extras_copy), "--write"]) == 0
        assert {path: path.read_bytes() for path in all_readmes} == generated

    def test_modes_have_truthful_artifact_and_command_contracts(self, generator):
        all_modes = generator.render_sections("active_generalized_filtering")
        outputs = all_modes[b"Outputs"].decode()
        assert "`animation_active_generalized_filtering.gif`" in outputs
        assert "`animation_active_generalized_filtering.npz`" in outputs
        assert "`animation_active_generalized_filtering.json`" in outputs
        assert "interactive_active_generalized_filtering.py\n" in outputs
        assert "interactive_active_generalized_filtering.py --save" not in outputs
        assert "interactive_active_generalized_filtering.png" not in outputs
        for slug in ("model_representation", "entropy"):
            text = b"".join(generator.render_sections(slug).values()).decode()
            assert f"`visualize_{slug}.png`" in text
            assert f"`visualize_{slug}.npz`" in text
            assert f"`visualize_{slug}.json`" in text
            assert f"animation_{slug}" not in text
        static = b"".join(generator.render_sections("model_representation").values())
        assert b"simulate_" not in static and b"interactive_" not in static

    def test_authored_sections_fences_and_crlf_are_preserved(self, generator):
        prefix = b"# Authored\r\n```md\r\n## Scripts\r\nexample only\r\n```\r\n\xff\r\n"
        middle = b"## Authored analysis\r\nExact prose and spaces.  \r\n"
        suffix = b"## References\r\nDo not rewrite.\r\n"
        raw = (
            prefix + b"## Scripts\r\nstale scripts\r\n" + middle
            + b"## Outputs\r\nstale outputs\r\n" + suffix
        )
        sections = generator.render_sections("model_representation")
        actual = generator.replace_sections(raw, sections)
        assert actual == (
            prefix + sections[b"Scripts"].replace(b"\n", b"\r\n") + middle
            + sections[b"Outputs"].replace(b"\n", b"\r\n") + suffix
        )

    @pytest.mark.parametrize("malformation", ["missing", "duplicate", "missing_wrapper"])
    def test_late_invalid_topic_prevents_all_writes(self, generator, extras_copy, malformation):
        directory = extras_copy / "variational_message_passing"
        readme = directory / "README.md"
        if malformation == "missing":
            readme.write_bytes(readme.read_bytes().replace(b"## Outputs", b"## Authored outputs"))
        elif malformation == "duplicate":
            readme.write_bytes(readme.read_bytes() + b"\n## Scripts\nDuplicate\n")
        else:
            (directory / "visualize_variational_message_passing.py").unlink()
        before = {path: path.read_bytes() for path in extras_copy.glob("*/README.md")}
        assert generator.main(["--extras-root", str(extras_copy), "--write"]) == 2
        assert {path: path.read_bytes() for path in before} == before

    def test_symlinked_owned_readme_is_not_overwritten(self, generator, extras_copy, tmp_path):
        readme = extras_copy / "entropy/README.md"
        outside = tmp_path / "authored.md"
        outside.write_bytes(readme.read_bytes())
        original = outside.read_bytes()
        readme.unlink()
        readme.symlink_to(outside)
        assert generator.main(["--extras-root", str(extras_copy), "--write"]) == 2
        assert readme.is_symlink() and outside.read_bytes() == original
