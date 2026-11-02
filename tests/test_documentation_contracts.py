"""Repository documentation drift checks."""

from __future__ import annotations

import importlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _chapter_numbers() -> list[int]:
    return sorted(
        int(path.name.removeprefix("chapter_"))
        for path in (REPO_ROOT / "chapters").glob("chapter_*")
        if path.is_dir()
    )


def _chapter_scripts(chapter: int) -> list[Path]:
    chapter_dir = REPO_ROOT / "chapters" / f"chapter_{chapter:02d}"
    return sorted(path for path in chapter_dir.glob("*.py") if path.name != "__init__.py")


def _extra_topics() -> list[Path]:
    extras_root = REPO_ROOT / "extras"
    if not extras_root.exists():
        return []
    return sorted(path for path in extras_root.iterdir() if path.is_dir())


def _extra_scripts(topic_dir: Path) -> list[Path]:
    return sorted(path for path in topic_dir.glob("*.py") if path.name != "__init__.py")


def _is_interactive_script(path: Path) -> bool:
    return "interactive" in path.name


def _strip_fenced_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _markdown_files() -> list[Path]:
    ignored = {".git", ".venv", ".pytest_cache", "__pycache__"}
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not ignored.intersection(path.parts)
    )


def test_live_chapters_have_docs_and_folder_contracts() -> None:
    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        assert (REPO_ROOT / "chapters" / f"chapter_{suffix}" / "README.md").exists()
        assert (REPO_ROOT / "chapters" / f"chapter_{suffix}" / "AGENTS.md").exists()
        assert (REPO_ROOT / "docs" / "chapters" / f"chapter_{suffix}.md").exists()
        assert (REPO_ROOT / "output" / "figures" / f"chapter_{suffix}" / "README.md").exists()
        assert (REPO_ROOT / "output" / "data" / f"chapter_{suffix}" / "README.md").exists()
        figure_readme = (
            REPO_ROOT / "output" / "figures" / f"chapter_{suffix}" / "README.md"
        ).read_text(encoding="utf-8")
        assert f"output/data/chapter_{suffix}" in figure_readme
        assert "NPZ" in figure_readme and "JSON" in figure_readme


def test_logical_folders_have_readme_and_agents_guides() -> None:
    required = [
        "chapters",
        "docs",
        "docs/chapters",
        "docs/reference",
        "docs/statistics",
        "docs/topics",
        "extras",
        "output",
        "output/data",
        "output/data/extras",
        "output/figures",
        "output/figures/extras",
        "scripts",
        "src",
        "src/active_inference",
        "src/active_inference/core",
        "src/active_inference/estimators",
        "src/active_inference/menu",
        "src/active_inference/utils",
        "src/active_inference/visualizations",
        "src/active_inference/web",
        "tests",
        "tests/chapters",
        "tests/core",
        "tests/estimators",
        "tests/extras",
        "tests/menu",
        "tests/utils",
        "tests/visualizations",
        "tests/web",
    ]
    missing: list[str] = []
    for rel in required:
        folder = REPO_ROOT / rel
        for name in ("README.md", "AGENTS.md"):
            if not (folder / name).exists():
                missing.append(f"{rel}/{name}")
    assert not missing, "Missing folder docs:\n" + "\n".join(missing)


def test_reading_order_enumerates_all_live_chapters() -> None:
    text = (REPO_ROOT / "docs" / "reading_order.md").read_text(encoding="utf-8")
    for chapter in _chapter_numbers():
        assert f"chapters/chapter_{chapter:02d}.md" in text


def test_docs_indices_enumerate_all_topic_pages() -> None:
    topic_files = sorted(
        path.name
        for path in (REPO_ROOT / "docs" / "topics").glob("*.md")
        if path.name not in {"AGENTS.md", "README.md"}
    )
    docs_hub = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    topics_index = (REPO_ROOT / "docs" / "topics" / "README.md").read_text(encoding="utf-8")
    for filename in topic_files:
        assert filename in docs_hub, f"{filename} missing from docs/README.md"
        assert filename in topics_index, f"{filename} missing from docs/topics/README.md"


def test_chapter_readme_coverage_counts_match_live_scripts() -> None:
    text = (REPO_ROOT / "chapters" / "README.md").read_text(encoding="utf-8")
    for chapter in _chapter_numbers():
        match = re.search(rf"^\| Chapter {chapter} \| (?P<count>\d+)\b", text, re.MULTILINE)
        assert match, f"Chapter {chapter} missing from chapters/README.md coverage table"
        assert int(match.group("count")) == len(_chapter_scripts(chapter))


def test_chapter_docs_mention_every_live_script() -> None:
    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        docs = [
            REPO_ROOT / "chapters" / f"chapter_{suffix}" / "README.md",
            REPO_ROOT / "docs" / "chapters" / f"chapter_{suffix}.md",
        ]
        scripts = [path.name for path in _chapter_scripts(chapter)]
        for doc in docs:
            text = doc.read_text(encoding="utf-8")
            missing = [script for script in scripts if script not in text]
            assert not missing, (
                f"{doc.relative_to(REPO_ROOT)} does not mention live scripts: "
                + ", ".join(missing)
            )


def test_extras_docs_mention_every_live_script_and_artifact_path() -> None:
    overview = (REPO_ROOT / "extras" / "README.md").read_text(encoding="utf-8")
    figures_overview = (
        REPO_ROOT / "output" / "figures" / "extras" / "README.md"
    ).read_text(encoding="utf-8")
    data_overview = (
        REPO_ROOT / "output" / "data" / "extras" / "README.md"
    ).read_text(encoding="utf-8")
    for topic_dir in _extra_topics():
        topic = topic_dir.name
        readme = topic_dir / "README.md"
        assert readme.exists(), f"{readme.relative_to(REPO_ROOT)} missing"
        text = readme.read_text(encoding="utf-8")
        assert f"output/figures/extras/{topic}" in text
        assert f"output/data/extras/{topic}" in text
        assert f"[`{topic}/`]({topic}/)" in overview

        scripts = [path.name for path in _extra_scripts(topic_dir)]
        missing = [script for script in scripts if script not in text]
        assert not missing, (
            f"{readme.relative_to(REPO_ROOT)} does not mention live scripts: "
            + ", ".join(missing)
        )

    for output_overview in (figures_overview, data_overview):
        assert "../../../extras/README.md" in output_overview
        assert "output/figures/extras/<topic>" in output_overview
        assert "output/data/extras/<topic>" in output_overview


def _figure_manifest(chapter: int) -> tuple[str, str]:
    """Return the (Expected files, Flag variants) sections for one chapter.

    These README sections are the recorded manifest of a chapter render. The
    contract test compares documents to each other, never to whichever
    artifacts happen to be on disk, because `output/figures/` is gitignored
    and regenerable and a checkout may hold a partial or absent render.
    """

    readme = REPO_ROOT / "output" / "figures" / f"chapter_{chapter:02d}" / "README.md"
    text = readme.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for heading in ("Expected files", "Flag variants"):
        body = text.partition(f"## {heading}\n")[2].partition("\n## ")[0]
        sections[heading] = body
    assert sections["Expected files"].strip(), (
        f"{readme.relative_to(REPO_ROOT)} has no Expected files section"
    )
    return sections["Expected files"], sections["Flag variants"]


def _recorded_media(body: str) -> list[str]:
    return re.findall(r"`([^`]+\.(?:png|gif))`", body)


def test_output_figures_overview_counts_match_the_recorded_manifest() -> None:
    text = (REPO_ROOT / "output" / "figures" / "README.md").read_text(encoding="utf-8")

    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        expected, _ = _figure_manifest(chapter)
        recorded = _recorded_media(expected)
        png_count = sum(name.endswith(".png") for name in recorded)
        gif_count = sum(name.endswith(".gif") for name in recorded)

        line = next(
            line for line in text.splitlines() if f"[`chapter_{suffix}/`]" in line
        )
        documented_png = re.search(r"(\d+)\s+PNGs?", line)
        documented_gif = re.search(r"(\d+)\s+GIFs?", line)
        assert (int(documented_png.group(1)) if documented_png else 0) == png_count, line
        assert (int(documented_gif.group(1)) if documented_gif else 0) == gif_count, line


def test_rendered_chapter_media_is_recorded_in_the_manifest() -> None:
    """Rendered media must be recorded; nothing is required to be present.

    A checkout can hold no media at all, the tracked sample for Chapters 1-3,
    or one freshly rendered chapter. Any artifact that is present has to be
    listed in that chapter's README, either in the Expected files table (the
    default render) or among the Flag variants.
    """

    for chapter in _chapter_numbers():
        suffix = f"{chapter:02d}"
        expected, variants = _figure_manifest(chapter)
        allowed = set(_recorded_media(expected))
        allowed.update(name for name in _recorded_media(variants) if "<" not in name)

        media_dir = REPO_ROOT / "output" / "figures" / f"chapter_{suffix}"
        present = sorted(
            path.name
            for pattern in ("*.png", "*.gif")
            for path in media_dir.glob(pattern)
        )
        unrecorded = [name for name in present if name not in allowed]
        assert not unrecorded, (
            f"output/figures/chapter_{suffix}/ holds media missing from its README: "
            + ", ".join(unrecorded)
        )


def test_chapter_smoke_tests_cover_every_noninteractive_chapter_script() -> None:
    smoke = importlib.import_module("tests.chapters.test_smoke")
    covered: set[Path] = set()
    for name, value in vars(smoke).items():
        if not name.startswith("CHAPTER_") or not isinstance(value, list):
            continue
        if all(isinstance(item, Path) for item in value):
            covered.update(path.resolve() for path in value)

    live = {
        script.resolve()
        for chapter in _chapter_numbers()
        for script in _chapter_scripts(chapter)
        if not _is_interactive_script(script)
    }
    missing = sorted(path.relative_to(REPO_ROOT).as_posix() for path in live - covered)
    assert not missing, "Chapter smoke tests do not cover:\n" + "\n".join(missing)


def test_root_readme_keeps_required_community_links() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://activeinference.institute/" in text
    assert "https://textbook-group.activeinference.institute/" in text


def test_gitignore_ignores_nested_generated_media_but_preserves_docs() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "output/figures/**/*.png" in text
    assert "output/figures/**/*.gif" in text
    assert "!output/figures/**/README.md" in text
    assert "!output/data/README.md" in text
    assert "!output/data/**/README.md" in text


def test_raw_data_contract_is_documented_for_agents_and_users() -> None:
    required = (
        "NPZ",
        "JSON",
        "save_chapter_data",
        "save_extra_data",
        "validate_raw_data_exports.py",
    )
    docs = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "output" / "data" / "README.md",
        REPO_ROOT / "output" / "data" / "AGENTS.md",
        REPO_ROOT / "scripts" / "README.md",
        REPO_ROOT / "docs" / "reference" / "utils.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for term in required:
            assert term in text, f"{term} missing from {path.relative_to(REPO_ROOT)}"


def test_reference_pages_cover_public_subpackage_exports() -> None:
    docs_by_module = {
        "active_inference.core": REPO_ROOT / "docs" / "reference" / "core.md",
        "active_inference.estimators": REPO_ROOT / "docs" / "reference" / "estimators.md",
        "active_inference.utils": REPO_ROOT / "docs" / "reference" / "utils.md",
        "active_inference.visualizations": REPO_ROOT / "docs" / "reference" / "visualizations.md",
    }
    for module_name, doc_path in docs_by_module.items():
        module = importlib.import_module(module_name)
        text = doc_path.read_text(encoding="utf-8")
        missing = [symbol for symbol in getattr(module, "__all__", ()) if symbol not in text]
        assert not missing, (
            f"{doc_path.relative_to(REPO_ROOT)} missing exports from {module_name}: "
            + ", ".join(missing)
        )


def test_script_docs_enumerate_top_level_utility_scripts() -> None:
    scripts = sorted(
        path.name for path in (REPO_ROOT / "scripts").iterdir() if path.suffix in {".py", ".sh"}
    )
    docs = [REPO_ROOT / "scripts" / "README.md", REPO_ROOT / "scripts" / "AGENTS.md"]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        missing = [script for script in scripts if script not in text]
        assert not missing, f"{doc.relative_to(REPO_ROOT)} missing scripts: {missing}"


def test_active_inference_topic_matches_implemented_control_layers() -> None:
    docs = [
        REPO_ROOT / "docs" / "topics" / "active_inference.md",
        REPO_ROOT / "docs" / "topics" / "bayesian_mechanics.md",
    ]
    forbidden = (
        "does not yet ship a control layer",
        "action as a planned extension",
        "planned Chapter 7+ work",
        "not a runtime concept yet",
    )
    required = ("ActiveInferenceAgent", "policy_posterior")
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, f"stale active-inference wording in {path}: {phrase}"
    active_text = docs[0].read_text(encoding="utf-8")
    for term in required:
        assert term in active_text


def test_markdown_local_links_resolve() -> None:
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
    broken: list[str] = []
    for path in _markdown_files():
        text = _strip_fenced_code(path.read_text(encoding="utf-8"))
        for match in pattern.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith("#"):
                continue
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (
                REPO_ROOT / target_path.lstrip("/")
                if target_path.startswith("/")
                else path.parent / target_path
            )
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)
