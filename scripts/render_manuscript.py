#!/usr/bin/env python3
"""Render the companion manuscript from its committed sources.

This is the authoritative render path for ``manuscript/``. Before any format is
produced it validates the surfaces the prose depends on:

* every ``{{TOKEN}}`` used in a section is declared in ``manuscript/variables.yaml``
  and its declared value still equals the value recomputed from the repository;
* every section has exactly one labelled H1, and labels are unique;
* every ``::: {#fig:...}`` block names a figure registered in
  ``output/figures/figure_registry.json``, the registered file exists, and the
  registry records no figure the prose never uses; the block carries only its
  image, because the registry supplies the alternative text and caption;
* every ``[@key]`` citation resolves in ``references.bib`` when
  ``bibliography.fail_on_missing`` is set.

``--check`` runs those validations and stops, which is what CI runs because it
needs no TeX installation and no Pandoc binary. The default mode then assembles
the admitted sources in manuscript order and renders each format enabled in
``manuscript/config.yaml``.

Exit codes: 0 success, 1 contract violation, 2 unusable configuration input,
3 missing external tool (named on stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - reported, never a silent pass
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
CONFIG_PATH = MANUSCRIPT_DIR / "config.yaml"
VARIABLES_PATH = MANUSCRIPT_DIR / "variables.yaml"
REGISTRY_PATH = REPO_ROOT / "output" / "figures" / "figure_registry.json"
BUILD_DIR = REPO_ROOT / "docs" / "_build" / "manuscript"

#: Files in ``manuscript/`` that are guidance or configuration, not sections.
NON_SECTION_FILES = {"README.md", "AGENTS.md", "SYNTAX.md", "preamble.md"}

#: Namespaces owned by pandoc-crossref and the formalism filter, not BibTeX keys.
CROSSREF_NAMESPACES = (
    "fig:",
    "tbl:",
    "eq:",
    "sec:",
    "def:",
    "prop:",
    "thm:",
    "lem:",
    "cor:",
    "rem:",
    "axm:",
    "clm:",
    "exm:",
)

SUPPORTED_FORMATS = {"html", "pdf"}
FORMAT_SUFFIX = {"html": ".html", "pdf": ".pdf"}

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
SECTION_LABEL_RE = re.compile(r"\{#sec:([A-Za-z0-9_]+)\}\s*$")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)\)(?:\{(?P<attrs>[^}]*)\})?")
FIGURE_BLOCK_RE = re.compile(
    r"^:::[ \t]*\{#fig:(?P<label>[A-Za-z0-9_]+)[^}]*\}[ \t]*\n(?P<body>.*?)^:::[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
CITATION_RE = re.compile(r"@([A-Za-z0-9_][A-Za-z0-9_:.-]*)")
BIB_ENTRY_RE = re.compile(r"@[A-Za-z]+\{([^,\s]+)\s*,")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


class ContractError(RuntimeError):
    """A manuscript surface violates its declared contract."""


def _load_yaml(path: Path) -> dict[str, Any]:
    """Return one YAML mapping, refusing a missing file or an unreadable document."""

    if yaml is None:
        raise ContractError(
            "PyYAML is not installed: run `uv sync` so the locked environment "
            "provides it for the manuscript renderer"
        )
    if not path.is_file():
        raise ContractError(f"missing configuration input: {path.relative_to(REPO_ROOT)}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"{path.relative_to(REPO_ROOT)} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{path.relative_to(REPO_ROOT)} must be a mapping")
    return payload


def _pyproject() -> dict[str, Any]:
    """Return the parsed ``pyproject.toml``."""

    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def compute_variables() -> dict[str, str]:
    """Return every manuscript variable recomputed from tracked repository state."""

    project = _pyproject()["project"]
    homepage = project["urls"]["Homepage"]
    slug = homepage.split("github.com/", 1)[-1].strip("/")
    pin = json.loads((REPO_ROOT / "specs" / "gnn-pin.json").read_text(encoding="utf-8"))
    chapters = sorted(
        path
        for path in (REPO_ROOT / "chapters").iterdir()
        if path.is_dir() and path.name.startswith("chapter_")
    )
    chapter_scripts = [
        path
        for chapter in chapters
        for path in chapter.glob("*.py")
        if path.name != "__init__.py"
    ]
    extras_topics = sorted(path for path in (REPO_ROOT / "extras").iterdir() if path.is_dir())

    return {
        "PACKAGE_NAME": str(project["name"]),
        "PACKAGE_VERSION": str(project["version"]),
        "REPOSITORY_SLUG": slug,
        "CHAPTER_COUNT": str(len(chapters)),
        "CHAPTER_SCRIPT_COUNT": str(len(chapter_scripts)),
        "EXTRAS_TOPIC_COUNT": str(len(extras_topics)),
        "GNN_PIN_SCHEMA": str(pin["schema"]),
        "GNN_PIN_TAG": str(pin["tag"]),
        "GNN_PIN_COMMIT": str(pin["commit"]),
        "GNN_PIN_LANGUAGE_VERSION": str(pin["language_version"]),
    }


def load_variables() -> dict[str, dict[str, str]]:
    """Return the declared variables, keyed by token, with their declared source."""

    payload = _load_yaml(VARIABLES_PATH)
    if payload.get("schema") != "manuscript-variables/v1":
        raise ContractError(
            "manuscript/variables.yaml must declare schema: manuscript-variables/v1"
        )
    declared = payload.get("variables")
    if not isinstance(declared, dict) or not declared:
        raise ContractError("manuscript/variables.yaml declares no variables")
    resolved: dict[str, dict[str, str]] = {}
    for token, record in declared.items():
        if not re.fullmatch(r"[A-Z0-9_]+", str(token)):
            raise ContractError(f"variable {token!r} is not an uppercase token name")
        if not isinstance(record, dict) or "value" not in record or "source" not in record:
            raise ContractError(f"variable {token} must declare a value and a source")
        resolved[str(token)] = {
            "value": str(record["value"]),
            "source": str(record["source"]),
        }
    return resolved


def verify_variables(
    declared: dict[str, dict[str, str]], used: set[str]
) -> dict[str, str]:
    """Return the resolved values, or raise when declaration and repository disagree."""

    computed = compute_variables()
    failures: list[str] = []
    for token, record in sorted(declared.items()):
        if token not in computed:
            failures.append(f"{token} has no repository producer")
            continue
        if record["value"] != computed[token]:
            failures.append(
                f"{token} is declared as {record['value']!r} but {record['source']} "
                f"now yields {computed[token]!r}: update manuscript/variables.yaml"
            )
    for token in sorted(used - set(declared)):
        failures.append(f"{{{{{token}}}}} is used in prose but not declared in variables.yaml")
    for token in sorted(set(declared) - used):
        failures.append(f"{token} is declared but no section uses it")
    if failures:
        raise ContractError("manuscript variables:\n  " + "\n  ".join(failures))
    return {token: record["value"] for token, record in declared.items()}


@dataclass
class Source:
    """One admitted manuscript section."""

    path: Path
    text: str
    label: str
    heading: str


@dataclass
class Manuscript:
    """The assembled manuscript and the observations made while assembling it."""

    config: dict[str, Any]
    sources: list[Source] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    figures: dict[str, dict[str, Any]] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    unused_bib_keys: list[str] = field(default_factory=list)


def _section_sort_key(path: Path) -> tuple[int, str]:
    """Order main sections, then supplements, then back matter."""

    match = re.match(r"^(\d{2})_", path.name)
    if match:
        number = int(match.group(1))
        return (2 if number >= 90 else 0, path.name)
    if re.match(r"^S\d{2}_", path.name):
        return (1, path.name)
    raise ContractError(f"{path.relative_to(REPO_ROOT)} is not a numbered manuscript section")


def admitted_sources() -> list[Path]:
    """Return the manuscript section files in admitted order."""

    if not MANUSCRIPT_DIR.is_dir():
        raise ContractError("manuscript/ does not exist")
    candidates = [
        path
        for path in MANUSCRIPT_DIR.glob("*.md")
        if path.name not in NON_SECTION_FILES
    ]
    if not candidates:
        raise ContractError("manuscript/ contains no section files")
    return sorted(candidates, key=_section_sort_key)


def load_sections(paths: list[Path]) -> list[Source]:
    """Return the sections with their single labelled H1 validated."""

    sources: list[Source] = []
    seen: dict[str, Path] = {}
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        headings = H1_RE.findall(FENCE_RE.sub("", raw))
        if len(headings) != 1:
            raise ContractError(
                f"{path.relative_to(REPO_ROOT)} must contain exactly one H1, found {len(headings)}"
            )
        label_match = SECTION_LABEL_RE.search(headings[0])
        if label_match is None:
            raise ContractError(
                f"{path.relative_to(REPO_ROOT)} H1 must end with a {{#sec:...}} label"
            )
        label = label_match.group(1)
        if label in seen:
            raise ContractError(
                f"section label sec:{label} is declared by both "
                f"{seen[label].relative_to(REPO_ROOT)} and {path.relative_to(REPO_ROOT)}"
            )
        seen[label] = path
        sources.append(
            Source(
                path=path,
                text=raw,
                label=label,
                heading=SECTION_LABEL_RE.sub("", headings[0]).strip(),
            )
        )
    return sources


def _used_tokens(sources: list[Source]) -> set[str]:
    """Return every ``{{TOKEN}}`` name used anywhere in the sources."""

    used: set[str] = set()
    for source in sources:
        used.update(TOKEN_RE.findall(source.text))
    return used


def validate_figures(sources: list[Source]) -> dict[str, dict[str, Any]]:
    """Return the registry after binding it to the manuscript's figure blocks."""

    if not REGISTRY_PATH.is_file():
        raise ContractError(
            f"missing figure registry: {REGISTRY_PATH.relative_to(REPO_ROOT)}"
        )
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ContractError("figure_registry.json must map labels to entries")

    referenced: dict[str, str] = {}
    failures: list[str] = []
    for source in sources:
        where = source.path.relative_to(REPO_ROOT)
        for block in FIGURE_BLOCK_RE.finditer(source.text):
            label = f"fig:{block.group('label')}"
            image = IMAGE_RE.search(block.group("body"))
            if image is None:
                failures.append(f"{where}: {label} block holds no image")
                continue
            if IMAGE_RE.sub("", block.group("body")).strip():
                failures.append(
                    f"{where}: {label} block may hold only its image; the caption lives in "
                    "the registry"
                )
            if image.group("alt").strip():
                failures.append(
                    f"{where}: {label} authors inline alt text; leave the brackets empty so "
                    "the registry alternative stays the only copy"
                )
            if label in referenced:
                failures.append(f"{where}: {label} is declared twice")
            referenced[label] = image.group("path")
        for match in IMAGE_RE.finditer(FIGURE_BLOCK_RE.sub("", source.text)):
            failures.append(
                f"{where}: image {match.group('path')} is not inside a ::: {{#fig:...}} block"
            )

    for label, entry in sorted(registry.items()):
        if not isinstance(entry, dict):
            failures.append(f"{label}: registry entry must be a mapping")
            continue
        if entry.get("label") != label:
            failures.append(f"{label}: entry label reads {entry.get('label')!r}")
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            failures.append(f"{label}: entry has no filename")
            continue
        if not (REGISTRY_PATH.parent / filename).is_file():
            failures.append(f"{label}: generated file output/figures/{filename} is missing")
        generator = entry.get("generated_by")
        if not isinstance(generator, str) or not generator:
            failures.append(f"{label}: entry has no generated_by producer")
        elif "/" in generator or generator.endswith(".py"):
            if not (REPO_ROOT / generator.split("::", 1)[0]).is_file():
                failures.append(f"{label}: producer {generator} does not exist")
        caption = entry.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            failures.append(f"{label}: entry has no caption")
        alt = entry.get("alt_text")
        if alt is None and isinstance(entry.get("metadata"), dict):
            alt = entry["metadata"].get("alt_text")
        if not isinstance(alt, str) or not alt.strip():
            failures.append(f"{label}: entry declares no alt_text")
        elif isinstance(caption, str) and alt.strip() == caption.strip():
            failures.append(f"{label}: alt_text repeats the caption instead of describing it")
        elif "[" in alt or "]" in alt:
            failures.append(f"{label}: alt_text must not contain square brackets")

    for label, path in sorted(referenced.items()):
        entry = registry.get(label)
        if entry is None:
            failures.append(f"{label} is referenced in prose but not registered")
            continue
        expected = f"../output/figures/{entry.get('filename')}"
        if path != expected:
            failures.append(f"{label} is referenced as {path} but registers as {expected}")
    for label in sorted(set(registry) - set(referenced)):
        failures.append(f"{label} is registered but no section references it")

    if failures:
        raise ContractError("figure registry:\n  " + "\n  ".join(failures))
    return registry


def bind_figures(text: str, registry: dict[str, dict[str, Any]]) -> str:
    """Return ``text`` with each figure block carrying its registry alt and caption.

    The authored block names the label and image; the alternative text and the
    caption come from the registry so neither is a second hand-maintained copy.
    A trailing full stop is dropped from the caption because pandoc-crossref
    appends one.
    """

    def replace(match: re.Match[str]) -> str:
        """Return one figure block carrying its registry alt and caption."""
        label = f"fig:{match.group('label')}"
        entry = registry[label]
        image = IMAGE_RE.search(match.group("body"))
        assert image is not None  # validated before assembly
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        alt = str(entry.get("alt_text") or metadata.get("alt_text", ""))
        caption = str(entry["caption"]).strip().rstrip(".")
        attrs = image.group("attrs") or ""
        image_line = f"![{alt}]({image.group('path')})"
        if attrs:
            image_line += f"{{{attrs}}}"
        return f"::: {{#{label}}}\n{image_line}\n\n{caption}\n:::"

    return FIGURE_BLOCK_RE.sub(replace, text)


def validate_bibliography(
    config: dict[str, Any], sources: list[Source]
) -> tuple[list[str], list[str]]:
    """Return the cited keys and the bibliography keys left unused."""

    bibliography = config.get("bibliography") or {}
    bib_path = REPO_ROOT / str(bibliography.get("references_path", "manuscript/references.bib"))
    if not bib_path.is_file():
        raise ContractError(f"missing bibliography: {bib_path.relative_to(REPO_ROOT)}")
    defined = set(BIB_ENTRY_RE.findall(bib_path.read_text(encoding="utf-8")))

    cited: list[str] = []
    for source in sources:
        body = FENCE_RE.sub("", source.text)
        for key in CITATION_RE.findall(body):
            if key.startswith(CROSSREF_NAMESPACES) or key in cited:
                continue
            cited.append(key)

    if bibliography.get("fail_on_missing", True):
        missing = [key for key in cited if key not in defined]
        if missing:
            raise ContractError(
                "unresolved citations: " + ", ".join(missing) + f" (checked {bib_path.name})"
            )
    unused = sorted(defined - set(cited)) if bibliography.get("fail_on_unused") else []
    return cited, unused


def _read_preamble() -> str:
    """Return the fenced LaTeX blocks of ``manuscript/preamble.md``."""

    path = MANUSCRIPT_DIR / "preamble.md"
    if not path.is_file():
        return ""
    blocks = re.findall(r"```latex\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    return "\n".join(block.strip("\n") for block in blocks)


def _pandoc_version() -> str:
    """Return pandoc's first version line."""

    result = subprocess.run(
        ["pandoc", "--version"], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()[0].strip()


def _tool(name: str, purpose: str) -> str:
    """Return the absolute path of ``name``, or raise when it is absent."""

    found = shutil.which(name)
    if found is None:
        raise MissingTool(f"{name} is required to {purpose} and is not on PATH")
    return found


class MissingTool(RuntimeError):
    """An external render tool is absent; the run is refused, never faked."""


def _format_settings(config: dict[str, Any]) -> list[str]:
    """Return the enabled formats in render order, refusing unsupported ones."""

    declared = (config.get("render") or {}).get("formats") or {}
    enabled = [name for name, value in declared.items() if value]
    unsupported = sorted(set(enabled) - SUPPORTED_FORMATS)
    if unsupported:
        raise ContractError(
            "manuscript/config.yaml enables formats this renderer does not produce: "
            + ", ".join(unsupported)
        )
    if not enabled:
        raise ContractError("manuscript/config.yaml enables no render format")
    return [name for name in ("html", "pdf") if name in enabled]


def _pandoc_command(
    *,
    combined: Path,
    output: Path,
    fmt: str,
    config: dict[str, Any],
    header: Path | None,
) -> list[str]:
    """Return the pandoc argv for one format."""
    paper = config.get("paper") or {}
    metadata = config.get("metadata") or {}
    bibliography = config.get("bibliography") or {}

    command = [
        "pandoc",
        str(combined),
        "--from",
        "markdown+tex_math_dollars+pipe_tables",
        "--standalone",
        "--number-sections",
        "--toc",
        "--toc-depth=2",
        f"--resource-path={MANUSCRIPT_DIR}:{REPO_ROOT}",
        "--metadata",
        f"title={paper.get('title', 'Manuscript')}",
        "--metadata",
        f"lang={metadata.get('language', 'en')}",
    ]
    subtitle = paper.get("subtitle")
    if subtitle:
        command += ["--metadata", f"subtitle={subtitle}"]
    version = paper.get("version")
    if version:
        command += ["--metadata", f"version={version}"]
    date = paper.get("date")
    if date:
        command += ["--metadata", f"date={date}"]
    for author in config.get("authors") or []:
        name = author.get("name")
        if name:
            command += ["--metadata", f"author={name}"]
    keywords = config.get("keywords") or []
    if keywords:
        command += ["--metadata", f"keywords={', '.join(str(word) for word in keywords)}"]

    bib_path = REPO_ROOT / str(bibliography.get("references_path", "manuscript/references.bib"))
    command += ["--bibliography", str(bib_path)]

    if shutil.which("pandoc-crossref"):
        command += ["--filter", "pandoc-crossref"]
    if header is not None:
        command += ["--include-in-header", str(header)]

    if fmt == "pdf":
        command += ["--pdf-engine=xelatex", "--natbib"]
    else:
        command += ["--citeproc", "--embed-resources"]
    command += ["--output", str(output)]
    return command


def _slug(config: dict[str, Any]) -> str:
    """Return the filesystem stem derived from the configured title."""

    title = str((config.get("paper") or {}).get("title", "manuscript"))
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "manuscript"


def build(check: bool) -> dict[str, Any]:
    """Validate the manuscript surfaces and, unless ``check``, render them."""

    config = _load_yaml(CONFIG_PATH)
    formats = _format_settings(config)
    sources = load_sections(admitted_sources())

    declared = load_variables()
    variables = verify_variables(declared, _used_tokens(sources))
    figures = validate_figures(sources)
    citations, unused_bib_keys = validate_bibliography(config, sources)

    receipt: dict[str, Any] = {
        "ok": True,
        "checked_only": check,
        "sections": [
            {
                "path": source.path.relative_to(REPO_ROOT).as_posix(),
                "label": f"sec:{source.label}",
                "heading": source.heading,
            }
            for source in sources
        ],
        "variables": variables,
        "figures": sorted(figures),
        "citations": citations,
        "unused_bibliography_keys": unused_bib_keys,
        "formats": formats,
        "outputs": [],
    }
    if check:
        return receipt

    _tool("pandoc", "render the manuscript")
    receipt["pandoc"] = _pandoc_version()
    if "pdf" in formats:
        _tool("xelatex", "render the manuscript PDF")
    receipt["pandoc_crossref"] = bool(shutil.which("pandoc-crossref"))

    build_dir = BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    # Substitute every declared token, bind each figure's registry alternative,
    # then join the sections in admitted order.
    combined_text = "\n\n".join(
        bind_figures(
            TOKEN_RE.sub(lambda match: variables[match.group(1)], source.text), figures
        ).rstrip()
        + "\n"
        for source in sources
    )
    slug = _slug(config)

    with tempfile.TemporaryDirectory(prefix="fundamentals-manuscript-") as directory:
        temp = Path(directory)
        combined = temp / f"{slug}.md"
        combined.write_text(combined_text, encoding="utf-8")
        header: Path | None = None
        preamble = _read_preamble()
        if preamble:
            header = temp / "preamble.tex"
            header.write_text(preamble + "\n", encoding="utf-8")
        for fmt in formats:
            output = build_dir / f"{slug}{FORMAT_SUFFIX[fmt]}"
            command = _pandoc_command(
                combined=combined, output=output, fmt=fmt, config=config, header=header
            )
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise ContractError(
                    f"pandoc {fmt} render failed (exit {completed.returncode}):\n"
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            if completed.stderr.strip():
                print(completed.stderr.strip(), file=sys.stderr)
            receipt["outputs"].append(
                {
                    "format": fmt,
                    "path": output.relative_to(REPO_ROOT).as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Validate and render the manuscript; return a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the manuscript surfaces without invoking pandoc",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=BUILD_DIR / "render-receipt.json",
        help="where to write the render receipt (default: docs/_build/manuscript/)",
    )
    parser.add_argument("--json", action="store_true", help="print the receipt to stdout")
    args = parser.parse_args(argv)

    try:
        receipt = build(args.check)
    except MissingTool as exc:
        print(f"manuscript render not run: {exc}", file=sys.stderr)
        return 3
    except ContractError as exc:
        print(f"manuscript contract violation: {exc}", file=sys.stderr)
        return 1

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    if args.check:
        print(
            f"manuscript check passed: {len(receipt['sections'])} sections, "
            f"{len(receipt['variables'])} variables, {len(receipt['figures'])} figures, "
            f"{len(receipt['citations'])} citations; receipt {args.receipt}"
        )
    else:
        rendered = ", ".join(item["path"] for item in receipt["outputs"])
        print(f"manuscript rendered: {rendered}; receipt {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
