"""Jupyter notebook export for chapter, extras, and demo orchestrators."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from active_inference.menu.runner import ScriptEntry

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from active_inference.utils.io import default_figure_dir, ensure_dir

_TEXTBOOK_GROUP_URL = "https://textbook-group.activeinference.institute/"
_NOTEBOOK_FORMAT = 4
_NOTEBOOK_MINOR = 5


def _repo_root() -> Path:
    """Locate the repository root (same walk-up policy as menu.runner)."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[3], here.parents[2], here.parents[1]):
        if (candidate / "chapters").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here.parents[3]


def _output_root() -> Path:
    """Return the output root, honoring the smoke-test environment override."""
    override = os.environ.get("ACTIVE_INFERENCE_OUTPUT_ROOT")
    if override:
        return Path(override)
    return _repo_root() / "output"


def default_notebook_dir() -> Path:
    """Return the notebook output directory under ``output/notebooks/``.

    Honors ``ACTIVE_INFERENCE_OUTPUT_ROOT`` when that environment variable is set.
    """
    return _output_root() / "notebooks"


def _runner():
    """Import discovery helpers lazily to avoid circular imports at package init."""
    from active_inference.menu import runner

    return runner


def chapter_notebook_path(chapter: int, *, root: Path | None = None) -> Path:
    """Resolve the on-disk path for one chapter notebook file."""
    base = root if root is not None else default_notebook_dir()
    return base / f"chapter_{chapter:02d}.ipynb"


def extra_notebook_path(topic: str, *, root: Path | None = None) -> Path:
    """Resolve the on-disk path for one extras-topic notebook file."""
    base = root if root is not None else default_notebook_dir()
    return base / "extras" / f"{topic}.ipynb"


def demo_notebook_path(slug: str, *, root: Path | None = None) -> Path:
    """Resolve the on-disk path for one application-demo notebook file."""
    base = root if root is not None else default_notebook_dir()
    return base / "demo" / f"{slug}.ipynb"


def read_module_docstring(path: Path) -> str:
    """Return the module docstring from ``path`` without executing it."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    doc = ast.get_docstring(tree) or ""
    paragraph = doc.strip().split("\n\n")[0].strip()
    return " ".join(line.strip() for line in paragraph.splitlines())


def relative_repo_path(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible."""
    root = _repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _kind_badge(kind: str) -> str:
    """Map a script classification label to a short display badge."""
    labels = {
        "example": "Example",
        "animation": "Animation",
        "visualize": "Visualization",
        "concept": "Concept",
        "interactive": "Interactive",
        "simulate": "Simulation",
        "other": "Script",
    }
    return labels.get(kind, kind.title())


def _figure_gif_path(entry: ScriptEntry) -> Path:
    """Return the expected GIF path for an animation orchestrator."""
    stem = entry.path.stem
    figures = default_figure_dir()
    if entry.chapter is not None:
        return figures / f"chapter_{entry.chapter:02d}" / f"{stem}.gif"
    if entry.topic is not None:
        return figures / "extras" / entry.topic / f"{stem}.gif"
    if entry.demo is not None:
        return figures / "demo" / entry.demo / f"{stem}.gif"
    raise ValueError(f"Cannot infer figure directory for {entry.path}")


def _data_sidecar_hint(entry: ScriptEntry) -> str:
    """Return a markdown hint pointing at NPZ/JSON sidecars when present."""
    stem = entry.path.stem
    if entry.chapter is not None:
        rel = f"output/data/chapter_{entry.chapter:02d}/{stem}"
    elif entry.topic is not None:
        rel = f"output/data/extras/{entry.topic}/{stem}"
    elif entry.demo is not None:
        rel = f"output/data/demo/{entry.demo}/{stem}"
    else:
        return ""
    return f"\n\nRaw data (after `--save`): `{rel}.npz`, `{rel}.json`."


def _setup_cell_source() -> str:
    """Return the shared notebook setup cell that configures paths and matplotlib."""
    return (
        "from pathlib import Path\n"
        "import runpy\n"
        "import sys\n"
        "import warnings\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Locate the repository root at runtime (not the export machine's path):\n"
        "# walk up from the notebook's working directory to the folder containing\n"
        "# both `src/active_inference` and `chapters/`. This keeps a clean clone of\n"
        "# the repo runnable regardless of who generated the notebook.\n"
        "def _find_repo_root(start=None):\n"
        "    here = Path(start or Path.cwd()).resolve()\n"
        "    for candidate in [here, *here.parents]:\n"
        "        if (candidate / 'src' / 'active_inference').is_dir() and (candidate / 'chapters').is_dir():\n"
        "            return candidate\n"
        "    return here\n"
        "\n"
        "REPO_ROOT = _find_repo_root()\n"
        "if str(REPO_ROOT / 'src') not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
        "\n"
        "%matplotlib inline\n"
        "warnings.filterwarnings('default')\n"
        "plt.close('all')\n"
    )


def _inline_script_cell(entry: ScriptEntry) -> str:
    """Return code that executes one static orchestrator via ``runpy``."""
    rel = relative_repo_path(entry.path)
    return (
        f"plt.close('all')\n"
        f"runpy.run_path(REPO_ROOT / {rel!r}, run_name='__main__')\n"
    )


def _animation_cells(entry: ScriptEntry) -> list[nbformat.NotebookNode]:
    """Return code cells that render an animation GIF and embed it inline."""
    rel = relative_repo_path(entry.path)
    gif = _figure_gif_path(entry)
    gif_rel = relative_repo_path(gif)
    run_code = (
        "import subprocess\n"
        "from IPython.display import Image, display\n"
        "\n"
        f"plt.close('all')\n"
        f"subprocess.run(\n"
        f"    [sys.executable, str(REPO_ROOT / {rel!r}), '--save'],\n"
        f"    check=True,\n"
        f"    cwd=str(REPO_ROOT),\n"
        f")\n"
        f"display(Image(filename=str(REPO_ROOT / {gif_rel!r})))\n"
    )
    return [new_code_cell(run_code)]


def _interactive_markdown(entry: ScriptEntry) -> str:
    """Return markdown instructions for running a GUI-only interactive script."""
    rel = relative_repo_path(entry.path)
    return (
        f"This script opens a matplotlib GUI with sliders and blocks until the "
        f"window is closed. Run it from a terminal or launch it via "
        f"`./run.sh --web`:\n\n"
        f"```bash\n"
        f"python {rel}\n"
        f"```\n\n"
        f"Slider-driven exploration is not executed inside this notebook kernel."
    )


def _section_markdown(entry: ScriptEntry) -> str:
    """Return the markdown header and summary for one orchestrator section."""
    doc = read_module_docstring(entry.path)
    badge = _kind_badge(entry.kind)
    body = f"## `{entry.path.stem}`\n\n**{badge}** — {doc}" if doc else f"## `{entry.path.stem}`\n\n**{badge}**"
    if entry.kind == "interactive":
        body += f"\n\n{_interactive_markdown(entry)}"
    elif entry.kind == "animation":
        body += _data_sidecar_hint(entry)
    return body


def build_notebook(
    scripts: Sequence[ScriptEntry],
    *,
    title: str,
    source_folder: str,
    include_animations: bool = True,
) -> nbformat.NotebookNode:
    """Assemble a Jupyter notebook from discovered chapter orchestrator scripts.

    Each script becomes a markdown section plus an execution or instruction cell.
    """
    preamble = (
        f"# {title}\n\n"
        f"Source orchestrators: `{source_folder}/`\n\n"
        f"Textbook reading group: [{_TEXTBOOK_GROUP_URL}]({_TEXTBOOK_GROUP_URL})\n\n"
        f"Regenerate with `uv run python scripts/export_notebooks.py`."
    )
    cells: list[nbformat.NotebookNode] = [
        new_markdown_cell(preamble),
        new_code_cell(_setup_cell_source()),
    ]

    for entry in scripts:
        if entry.kind == "animation" and not include_animations:
            cells.append(
                new_markdown_cell(
                    _section_markdown(entry)
                    + "\n\n*Animation section skipped (`--no-animations`). "
                    "Re-export without that flag or run the script with `--save`.*"
                )
            )
            continue

        cells.append(new_markdown_cell(_section_markdown(entry)))

        if entry.kind == "interactive":
            continue
        if entry.kind == "animation":
            cells.extend(_animation_cells(entry))
        else:
            cells.append(new_code_cell(_inline_script_cell(entry)))

    notebook = new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    return notebook


def write_notebook(notebook: nbformat.NotebookNode, path: Path) -> Path:
    """Validate a notebook structure and write it to the given output path."""
    nbformat.validate(notebook)
    ensure_dir(path.parent)
    nbformat.write(notebook, path)
    return path


@dataclass
class ExportResult:
    """Summary of a batch notebook export across chapters, extras, and demos."""

    chapter_paths: list[Path] = field(default_factory=list)
    extra_paths: list[Path] = field(default_factory=list)
    demo_paths: list[Path] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Return the number of notebooks written."""
        return len(self.chapter_paths) + len(self.extra_paths) + len(self.demo_paths)


def export_chapter_notebook(
    chapter: int,
    *,
    root: Path | None = None,
    include_animations: bool = True,
    include_interactive: bool = True,
) -> Path:
    """Export one chapter notebook using the live orchestrator discovery inventory."""
    runner = _runner()
    scripts = runner.discover_scripts(
        chapter,
        include_animations=include_animations,
        include_interactive=include_interactive,
    )
    notebook = build_notebook(
        scripts,
        title=f"Chapter {chapter:02d}",
        source_folder=f"chapters/chapter_{chapter:02d}",
        include_animations=include_animations,
    )
    return write_notebook(notebook, chapter_notebook_path(chapter, root=root))


def export_extra_notebook(
    topic: str,
    *,
    root: Path | None = None,
    include_animations: bool = True,
    include_interactive: bool = True,
) -> Path:
    """Export one extras-topic notebook using the live orchestrator discovery inventory."""
    runner = _runner()
    scripts = runner.discover_extra_scripts(
        topic,
        include_animations=include_animations,
        include_interactive=include_interactive,
    )
    notebook = build_notebook(
        scripts,
        title=topic.replace("_", " ").title(),
        source_folder=f"extras/{topic}",
        include_animations=include_animations,
    )
    return write_notebook(notebook, extra_notebook_path(topic, root=root))


def export_demo_notebook(
    slug: str,
    *,
    root: Path | None = None,
    include_animations: bool = True,
    include_interactive: bool = True,
) -> Path:
    """Export one application-demo notebook using the live orchestrator discovery inventory."""
    runner = _runner()
    scripts = runner.discover_demo_scripts(
        slug,
        include_animations=include_animations,
        include_interactive=include_interactive,
    )
    notebook = build_notebook(
        scripts,
        title=slug.replace("_", " ").title(),
        source_folder=f"demo/{slug}",
        include_animations=include_animations,
    )
    return write_notebook(notebook, demo_notebook_path(slug, root=root))


def export_all_notebooks(
    *,
    chapters: Sequence[int] | None = None,
    extras: Sequence[str] | None = None,
    demos: Sequence[str] | None = None,
    include_chapters: bool = True,
    include_extras: bool = True,
    include_demos: bool = True,
    root: Path | None = None,
    include_animations: bool = True,
    include_interactive: bool = True,
) -> ExportResult:
    """Export notebooks for the selected chapters, extras topics, and demos."""
    runner = _runner()
    result = ExportResult()

    if include_chapters:
        chapter_numbers = (
            list(chapters)
            if chapters is not None
            else [entry.number for entry in runner.discover_chapters()]
        )
        for number in chapter_numbers:
            result.chapter_paths.append(
                export_chapter_notebook(
                    number,
                    root=root,
                    include_animations=include_animations,
                    include_interactive=include_interactive,
                )
            )

    if include_extras:
        topics = (
            list(extras)
            if extras is not None
            else [entry.slug for entry in runner.discover_extras()]
        )
        for topic in topics:
            result.extra_paths.append(
                export_extra_notebook(
                    topic,
                    root=root,
                    include_animations=include_animations,
                    include_interactive=include_interactive,
                )
            )

    if include_demos:
        slugs = (
            list(demos)
            if demos is not None
            else [entry.slug for entry in runner.discover_demos()]
        )
        for slug in slugs:
            result.demo_paths.append(
                export_demo_notebook(
                    slug,
                    root=root,
                    include_animations=include_animations,
                    include_interactive=include_interactive,
                )
            )

    return result
