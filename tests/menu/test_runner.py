"""Unit tests for the chapter-runner discovery layer.

These tests do *not* invoke chapter scripts — that's covered by
``tests/chapters/test_smoke.py``. They only verify the discovery /
classification logic that the text menu and ``run.sh`` rely on.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from active_inference.menu import runner, tui
from active_inference.extra_topics import extra_topic_slugs


class TestDiscovery:
    def test_repo_root_resolves(self) -> None:
        assert runner.REPO_ROOT.is_dir()
        assert (runner.REPO_ROOT / "chapters").is_dir()
        assert (runner.REPO_ROOT / "extras").is_dir()
        assert (runner.REPO_ROOT / "src" / "active_inference").is_dir()

    def test_discover_chapters_returns_entries_in_order(self) -> None:
        chapters = runner.discover_chapters()
        numbers = [c.number for c in chapters]
        assert numbers == sorted(numbers)
        assert numbers[:3] == [1, 2, 3]

    def test_chapter_1_has_five_concept_scripts(self) -> None:
        scripts = runner.discover_scripts(1)
        names = [s.name for s in scripts]
        assert names == [
            "01_box_scenario.py",
            "02_three_perspectives.py",
            "03_bayes_intuition.py",
            "04_inverse_problem.py",
            "05_belief_from_stream.py",
        ]
        for s in scripts:
            assert s.kind == "concept"

    def test_chapter_2_excludes_interactive_explorer(self) -> None:
        names = [s.name for s in runner.discover_scripts(2)]
        assert "interactive_explorer.py" not in names
        assert "example_2_2_linear_probabilistic.py" in names
        assert "visualize_generative_model.py" in names

    def test_chapter_3_includes_animations_and_visualizations(self) -> None:
        scripts = runner.discover_scripts(3)
        kinds = {s.kind for s in scripts}
        assert kinds == {"example", "animation", "visualize"}
        assert len(scripts) == 18

    def test_no_animations_flag_drops_gifs(self) -> None:
        scripts = runner.discover_scripts(3, include_animations=False)
        assert all(s.kind != "animation" for s in scripts)
        assert any(s.kind == "example" for s in scripts)

    def test_unknown_chapter_raises(self) -> None:
        with pytest.raises(KeyError):
            runner.discover_scripts(99)

    def test_discover_extras_returns_topic_entries(self) -> None:
        extras = runner.discover_extras()
        slugs = [entry.slug for entry in extras]
        assert slugs == list(extra_topic_slugs())
        assert {
            "bayes_equation",
            "kl_divergence",
            "entropy",
            "enthalpy",
            "temperature",
            "variational_free_energy",
        }.issubset(slugs)
        assert slugs[:6] == list(extra_topic_slugs())[:6]

    def test_discover_demos_returns_topic_entries(self) -> None:
        from active_inference.demo_topics import demo_topic_slugs

        demos = runner.discover_demos()
        slugs = [entry.slug for entry in demos]
        assert slugs == list(demo_topic_slugs())
        assert slugs == ["eye_saccades", "bicycle", "drone_flight"]

    def test_extra_topic_scripts_include_declared_kinds(self) -> None:
        scripts = runner.discover_extra_scripts("entropy")
        names = [script.name for script in scripts]
        kinds = {script.kind for script in scripts}
        assert "visualize_entropy.py" in names
        assert "simulate_entropy.py" in names
        assert {"visualize", "simulate"}.issubset(kinds)
        assert all(script.topic == "entropy" for script in scripts)

    def test_extra_topic_interactive_scripts_are_discovered_on_request(self) -> None:
        scripts = runner.discover_extra_scripts("entropy", include_interactive=True)
        names = [script.name for script in scripts]
        interactive = next(script for script in scripts if script.name == "interactive_entropy.py")
        assert "interactive_entropy.py" in names
        assert interactive.kind == "interactive"
        assert interactive.topic == "entropy"

    def test_unknown_extra_topic_raises(self) -> None:
        with pytest.raises(KeyError):
            runner.discover_extra_scripts("not_a_topic")


class TestClassification:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("example_2_1_linear_deterministic.py", "example"),
            ("animation_em_steps.py", "animation"),
            ("simulate_entropy.py", "simulate"),
            ("visualize_calibration.py", "visualize"),
            ("01_box_scenario.py", "concept"),
            ("interactive_explorer.py", "interactive"),
            ("misc_helper.py", "other"),
        ],
    )
    def test_classify(self, name: str, expected: str) -> None:
        assert runner._classify(Path(name)) == expected


class TestExecutionHelpers:
    def test_build_env_prepends_src_extra_and_existing_pythonpath(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", "existing")
        env = runner._build_env([tmp_path / "extrasrc"])
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(runner.REPO_ROOT / "src")
        assert parts[1] == str(tmp_path / "extrasrc")
        assert parts[-1] == "existing"
        assert env["MPLBACKEND"] == "Agg"

    def test_run_script_path_with_args_and_capture(self, tmp_path: Path) -> None:
        script = tmp_path / "hello.py"
        script.write_text(
            "import os, sys\nprint(os.environ['MPLBACKEND'])\nprint(sys.argv[-1])\n",
            encoding="utf-8",
        )
        completed = runner.run_script(
            script,
            save=False,
            extra_args=["--flag"],
            capture_output=True,
            quiet=True,
        )
        assert completed.returncode == 0
        assert completed.stdout.splitlines() == ["Agg", "--flag"]

    def test_run_chapter_fail_fast_and_keep_going(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        scripts = [
            runner.ScriptEntry(tmp_path / "first.py", 1, "example"),
            runner.ScriptEntry(tmp_path / "second.py", 1, "example"),
        ]
        monkeypatch.setattr(runner, "discover_scripts", lambda *args, **kwargs: scripts)
        calls: list[str] = []

        def fake_run(script, **kwargs):
            calls.append(script.name)
            return subprocess.CompletedProcess([script.name], 1 if script.name == "first.py" else 0)

        monkeypatch.setattr(runner, "run_script", fake_run)
        stopped = runner.run_chapter(1, quiet=True)
        assert [script.name for script, _ in stopped] == ["first.py"]
        assert calls == ["first.py"]

        calls.clear()
        continued = runner.run_chapter(1, keep_going=True, quiet=True)
        assert [script.name for script, _ in continued] == ["first.py", "second.py"]
        assert calls == ["first.py", "second.py"]

    def test_run_extra_topic_fail_fast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        scripts = [
            runner.ScriptEntry(tmp_path / "visualize_demo.py", None, "visualize", topic="demo"),
            runner.ScriptEntry(tmp_path / "simulate_demo.py", None, "simulate", topic="demo"),
        ]
        monkeypatch.setattr(runner, "discover_extra_scripts", lambda *args, **kwargs: scripts)
        monkeypatch.setattr(
            runner,
            "run_script",
            lambda script, **kwargs: subprocess.CompletedProcess([script.name], 2),
        )
        stopped = runner.run_extra_topic("demo", quiet=True)
        assert len(stopped) == 1
        assert stopped[0][1] == 2

    def test_run_all_helpers_stop_on_first_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            runner,
            "discover_chapters",
            lambda **kwargs: [
                runner.ChapterEntry(1, tmp_path / "chapter_01"),
                runner.ChapterEntry(2, tmp_path / "chapter_02"),
            ],
        )
        monkeypatch.setattr(
            runner,
            "run_chapter",
            lambda number, **kwargs: [(runner.ScriptEntry(tmp_path / f"{number}.py", number, "example"), number)],
        )
        chapters = runner.run_all_chapters(quiet=True)
        assert list(chapters) == [1]

        monkeypatch.setattr(
            runner,
            "discover_extras",
            lambda **kwargs: [
                runner.ExtraTopicEntry("a", tmp_path / "a"),
                runner.ExtraTopicEntry("b", tmp_path / "b"),
            ],
        )
        monkeypatch.setattr(
            runner,
            "run_extra_topic",
            lambda slug, **kwargs: [(runner.ScriptEntry(tmp_path / f"{slug}.py", None, "visualize", topic=slug), 1)],
        )
        extras = runner.run_all_extras(quiet=True)
        assert list(extras) == ["a"]


class TestMenuRendering:
    def test_render_menu_includes_each_chapter(self) -> None:
        text = tui.render_menu(runner.discover_chapters())
        assert "Chapter 01" in text
        assert "Chapter 02" in text
        assert "Chapter 03" in text
        assert "Extras" in text
        assert "entropy" in text
        assert "[a]" in text
        assert "[q]" in text

    def test_main_list_flag_does_not_invoke_subprocess(self) -> None:
        rc = tui.main(["--list"])
        assert rc == 0


class TestScriptResolution:
    def test_filename_fragment_matches(self) -> None:
        path = tui._resolve_script_path("example_2_2_linear")
        assert path is not None
        assert path.name == "example_2_2_linear_probabilistic.py"

    def test_extra_filename_fragment_matches(self) -> None:
        path = tui._resolve_script_path("visualize_entropy")
        assert path is not None
        assert path.name == "visualize_entropy.py"

    def test_full_relative_path_resolves(self) -> None:
        path = tui._resolve_script_path(
            "chapters/chapter_01/03_bayes_intuition.py"
        )
        assert path is not None
        assert path.name == "03_bayes_intuition.py"

    def test_unknown_fragment_returns_none(self) -> None:
        assert tui._resolve_script_path("definitely_not_a_script") is None


@pytest.fixture
def discovery_tree(tmp_path, monkeypatch):
    roots = {}
    for attribute, relative in (
        ("CHAPTERS_ROOT", "chapters"), ("EXTRAS_ROOT", "extras"), ("DEMO_ROOT", "demo"),
    ):
        root = tmp_path / relative
        root.mkdir()
        roots[relative] = root
        monkeypatch.setattr(runner, attribute, root)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tui, "REPO_ROOT", tmp_path)
    for relative in ("chapters/chapter_01", "chapters/chapter_02", "extras/toy", "demo/toy"):
        directory = tmp_path / relative
        directory.mkdir()
        (directory / "visualize_probe.py").write_text("pass\n")
        (directory / "interactive_probe.py").write_text("pass\n")
        (directory / "_helper.py").write_text("pass\n")
        (directory / "README.md").write_text("Authored fixture\n")
    return tmp_path


class TestOperationSnapshots:
    def test_snapshot_is_stable_and_later_operations_see_changes(self, discovery_tree):
        snapshot = runner.discover_snapshot()
        chapter = discovery_tree / "chapters/chapter_01"
        (chapter / "example_new.py").write_text("pass\n")
        old_names = [script.name for script in runner.discover_scripts(1, snapshot=snapshot)]
        assert old_names == ["visualize_probe.py"]
        assert [script.name for script in runner.discover_scripts(
            1, snapshot=snapshot, include_interactive=True,
        )] == ["interactive_probe.py", "visualize_probe.py"]
        assert [script.name for script in runner.discover_scripts(1)] == [
            "example_new.py", "visualize_probe.py",
        ]
        assert tui._resolve_script_path("example_new", snapshot=snapshot) is None
        assert tui._resolve_script_path("example_new") == chapter / "example_new.py"
        with pytest.raises(TypeError):
            snapshot.chapters[3] = snapshot.chapters[1]

    def test_one_capture_does_not_rescan_each_topic(self, discovery_tree, monkeypatch):
        calls = []
        original = runner._scan_scripts

        def observe(base, **kwargs):
            calls.append(base)
            return original(base, **kwargs)

        monkeypatch.setattr(runner, "_scan_scripts", observe)
        snapshot = runner.discover_snapshot()
        for number in snapshot.chapters:
            assert runner.discover_scripts(number, snapshot=snapshot)[0].name == "visualize_probe.py"
        assert runner.discover_extra_scripts("toy", snapshot=snapshot)[0].topic == "toy"
        assert runner.discover_demo_scripts("toy", snapshot=snapshot)[0].demo == "toy"
        assert len(calls) == len(set(calls)) == 4

    def test_concurrent_captures_cannot_clear_another_operation(self, discovery_tree, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        barrier = Barrier(2)
        original = runner._scan_scripts

        def interleave(base, **kwargs):
            if base.name == "chapter_01":
                barrier.wait(timeout=5)
            return original(base, **kwargs)

        monkeypatch.setattr(runner, "_scan_scripts", interleave)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(runner.discover_snapshot) for _ in range(2)]
            for future in futures:
                snapshot = future.result(timeout=10)
                assert tuple(snapshot.chapters) == (1, 2)
                assert tuple(snapshot.extras) == ("toy",)
                assert tuple(snapshot.demos) == ("toy",)
                assert all(len(entry.scripts) == 2 for entry in snapshot.chapters.values())

    def test_batch_execution_uses_the_admitted_snapshot(self, discovery_tree):
        marker = discovery_tree / "executed.txt"
        first = discovery_tree / "chapters/chapter_01/visualize_probe.py"
        second = discovery_tree / "chapters/chapter_02/visualize_probe.py"
        first.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('first\\n')\n"
        )
        second.write_text(
            f"from pathlib import Path\np=Path({str(marker)!r})\n"
            "p.write_text(p.read_text() + 'second\\n')\n"
        )
        snapshot = runner.discover_snapshot()
        (first.parent / "example_late.py").write_text("raise RuntimeError('not admitted')\n")
        results = runner.run_all_chapters(snapshot=snapshot, quiet=True)
        assert [code for values in results.values() for _, code in values] == [0, 0]
        assert marker.read_text() == "first\nsecond\n"
