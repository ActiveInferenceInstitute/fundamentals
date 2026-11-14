"""Stdlib-only HTTP server for the chapter-runner web UI.

Routes
------

GET  /                          → SPA shell (HTML)
GET  /favicon.svg               → inline SVG favicon (mono with red dot)
GET  /static/app.css            → CSS bundle (inline string)
GET  /static/app.js             → JS bundle (inline string)
GET  /api/index                 → list of chapters + doc pages (JSON)
GET  /api/chapter/<N>           → scripts + figures + docs for chapter N (JSON)
GET  /api/doc/<id>              → rendered-to-HTML markdown for one doc page
GET  /figures/<chapter>/<file>  → serve a figure file from output/figures/
GET  /docs-raw/<path>           → serve a raw file from docs/ (for assets)
POST /api/run                   → run a chapter script with ``--save``
POST /api/launch-interactive    → launch an interactive (GUI) script

Everything stays in the repository sandbox via path-canonicalization in
:func:`_safe_subpath`. Subprocess invocations reuse the headless runner in
:mod:`active_inference.menu.runner`.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from ..demo_topics import demo_topic_spec
from ..menu.runner import (
    OUTPUT_DIR,
    REPO_ROOT,
    ScriptEntry,
    _build_env,
    discover_chapters,
    discover_snapshot,
    discover_demos,
    discover_extras,
    run_script,
)
from ..extra_topics import extra_topic_spec
from .templates import CSS, FAVICON_SVG, JS, render_index_html

LOG = logging.getLogger("active_inference.web")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

DOCS_DIR = REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _chapter_figure_dir(chapter: int) -> Path:
    """Return chapter-specific metadata used for discovery and display."""
    return OUTPUT_DIR / f"chapter_{chapter:02d}"


def _extra_figure_dir(topic: str) -> Path:
    """Return extras-topic figure directory used for discovery and display."""
    return OUTPUT_DIR / "extras" / topic


def _demo_figure_dir(slug: str) -> Path:
    """Return demo-topic figure directory used for discovery and display."""
    return OUTPUT_DIR / "demo" / slug


_DOC_SUMMARY_LIMIT = 220


def _list_figures(chapter: int, scripts: Sequence[ScriptEntry]) -> list[dict[str, Any]]:
    """List repository resources needed by the local browser UI."""
    base = _chapter_figure_dir(chapter)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".gif", ".jpg", ".jpeg", ".webp", ".svg"}:
            continue
        stat = p.stat()
        width, height = _image_dimensions(p)
        out.append({
            "name": p.name,
            "url": f"/figures/{chapter:02d}/{p.name}",
            "kind": "animation" if p.suffix.lower() == ".gif" else "static",
            "extension": p.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "size_human": _human_bytes(stat.st_size),
            "mtime": stat.st_mtime,
            "mtime_human": _human_time(stat.st_mtime),
            "width": width,
            "height": height,
            "generated_by": _guess_generator(p, scripts),
        })
    return out


def _list_extra_figures(topic: str, scripts: Sequence[ScriptEntry]) -> list[dict[str, Any]]:
    """List rendered figure files for one extras topic."""
    base = _extra_figure_dir(topic)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".gif", ".jpg", ".jpeg", ".webp", ".svg"}:
            continue
        stat = p.stat()
        width, height = _image_dimensions(p)
        out.append({
            "name": p.name,
            "url": f"/figures/extras/{topic}/{p.name}",
            "kind": "animation" if p.suffix.lower() == ".gif" else "static",
            "extension": p.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "size_human": _human_bytes(stat.st_size),
            "mtime": stat.st_mtime,
            "mtime_human": _human_time(stat.st_mtime),
            "width": width,
            "height": height,
            "generated_by": _guess_generator(p, scripts),
        })
    return out


def _list_demo_figures(slug: str, scripts: Sequence[ScriptEntry]) -> list[dict[str, Any]]:
    """List rendered figure files for one demo topic."""
    base = _demo_figure_dir(slug)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".gif", ".jpg", ".jpeg", ".webp", ".svg"}:
            continue
        stat = p.stat()
        width, height = _image_dimensions(p)
        out.append({
            "name": p.name,
            "url": f"/figures/demo/{slug}/{p.name}",
            "kind": "animation" if p.suffix.lower() == ".gif" else "static",
            "extension": p.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "size_human": _human_bytes(stat.st_size),
            "mtime": stat.st_mtime,
            "mtime_human": _human_time(stat.st_mtime),
            "width": width,
            "height": height,
            "generated_by": _guess_generator(p, scripts),
        })
    return out


def _demo_registry_payload(slug: str) -> dict[str, Any]:
    """Return registry metadata for one demo topic."""
    try:
        spec = demo_topic_spec(slug)
    except KeyError:
        return {
            "title": slug.replace("_", " ").title(),
            "summary": "Application demo",
            "chapters": [],
            "source_apis": [],
        }
    return {
        "title": spec.title,
        "summary": spec.summary,
        "chapters": list(spec.chapters),
        "source_apis": list(spec.source_apis),
    }


def _extra_registry_payload(topic: str) -> dict[str, Any]:
    """Return registry metadata for one extras topic."""
    try:
        spec = extra_topic_spec(topic)
    except KeyError:
        return {
            "title": topic.replace("_", " ").title(),
            "family": "Extras",
            "summary": "Extras topic",
            "chapters": [],
            "sections": [],
        }
    return {
        "title": spec.title,
        "family": spec.family,
        "summary": spec.summary,
        "chapters": list(spec.chapters),
        "sections": list(spec.sections),
    }


def _script_meta(entry: ScriptEntry) -> dict[str, Any]:
    """Return UI-friendly metadata for one script."""
    info: dict[str, Any] = {
        "name": entry.name,
        "kind": entry.kind,
        "relative": entry.relative,
    }
    try:
        stat = entry.path.stat()
        info["size"] = stat.st_size
        info["size_human"] = _human_bytes(stat.st_size)
        info["mtime"] = stat.st_mtime
        info["mtime_human"] = _human_time(stat.st_mtime)
    except OSError:
        pass
    info["docstring"] = _extract_docstring(entry.path)
    info["example_number"] = _extract_example_number(entry.name)
    return info


def _extract_docstring(path: Path) -> str | None:
    """Read the module docstring (first triple-quoted block) from a script."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Find the first triple-quoted string. Handle both ''' and \"\"\".
    for quote in ('"""', "'''"):
        start = text.find(quote)
        if start == -1:
            continue
        end = text.find(quote, start + 3)
        if end == -1:
            continue
        # Strip the leading docline only — the rest is the summary.
        body = text[start + 3:end].strip()
        if not body:
            continue
        first_line = body.splitlines()[0].strip()
        if not first_line:
            continue
        if len(first_line) > _DOC_SUMMARY_LIMIT:
            first_line = first_line[: _DOC_SUMMARY_LIMIT - 1] + "…"
        return first_line
    return None


def _extract_example_number(name: str) -> str | None:
    """Pull '2.5' out of 'example_2_5_nonlinear_probabilistic.py'."""
    m = re.match(r"^example_(\d+)_(\d+)_", name)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.match(r"^(\d{2})_", name)
    if m:
        return m.group(1)
    return None


def _guess_generator(figure: Path, inventory: Sequence[ScriptEntry]) -> str | None:
    """Match a figure against the operation's captured scripts without rescanning."""
    stem = figure.stem
    scripts = {script.path.stem for script in inventory}
    if stem in scripts:
        return f"{stem}.py"
    # Strip trailing suffixes like "_posterior", "_curve" until we hit a script stem.
    parts = stem.split("_")
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in scripts:
            return f"{candidate}.py"
    return None




def _human_bytes(n: int) -> str:
    """Format a machine value for compact human-readable display."""
    units = ("B", "KB", "MB", "GB")
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{n} B"


def _human_time(epoch: float) -> str:
    """Format a machine value for compact human-readable display."""
    delta = max(0.0, time.time() - epoch)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)} min ago"
    if delta < 86400:
        return f"{int(delta / 3600)} h ago"
    days = int(delta / 86400)
    return f"{days} day ago" if days == 1 else f"{days} days ago"


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` without loading the full image.

    Reads only the file header (PNG / GIF / JPEG). Returns ``(None, None)``
    for any format we can't parse cheaply.
    """
    try:
        with path.open("rb") as fp:
            head = fp.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                fp.seek(16)
                width = int.from_bytes(fp.read(4), "big")
                height = int.from_bytes(fp.read(4), "big")
                return width, height
            if head[:6] in (b"GIF87a", b"GIF89a"):
                width = int.from_bytes(head[6:8], "little")
                height = int.from_bytes(head[8:10], "little")
                return width, height
            if head[:2] == b"\xff\xd8":
                # JPEG: walk markers until SOF0/SOF2.
                fp.seek(2)
                while True:
                    while fp.read(1) != b"\xff":
                        pass
                    marker = fp.read(1)
                    if marker in (b"\xc0", b"\xc2"):
                        fp.read(3)
                        height = int.from_bytes(fp.read(2), "big")
                        width = int.from_bytes(fp.read(2), "big")
                        return width, height
                    size = int.from_bytes(fp.read(2), "big")
                    fp.read(size - 2)
    except (OSError, ValueError):
        return None, None
    return None, None


# Friendly chapter subtitles surfaced in the sidebar and header. Falls back
# to the chapter README's H1 if absent.
_CHAPTER_SUBTITLES: dict[int, str] = {
    1: "The Hypothesis-Testing Brain",
    2: "Hidden State Estimation",
    3: "Combining Learning and Inference",
}


def _chapter_subtitle(chapter: int, directory: Path) -> str:
    """Return chapter-specific metadata used for discovery and display."""
    if chapter in _CHAPTER_SUBTITLES:
        return _CHAPTER_SUBTITLES[chapter]
    readme = directory / "README.md"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                # Drop the leading "Chapter N — " if present.
                stripped = re.sub(r"^Chapter\s+\d+\s*[—\-:]\s*", "", title)
                return stripped or title
    return ""


# Map chapter numbers to relevant documentation pages.
_CHAPTER_DOC_MAP: dict[int, list[str]] = {
    1: ["chapters/chapter_01.md", "topics/bayesian_inference.md", "topics/inverse_problem.md"],
    2: ["chapters/chapter_02.md", "topics/generative_models.md", "topics/gradient_descent.md"],
    3: [
        "chapters/chapter_03.md",
        "topics/learning_and_inference.md",
        "topics/multivariate_gaussians.md",
        "statistics/calibration.md",
        "statistics/posterior_predictive.md",
    ],
}


def _list_chapter_docs(chapter: int) -> list[dict[str, str]]:
    """List repository resources needed by the local browser UI."""
    out: list[dict[str, str]] = []
    for rel in _CHAPTER_DOC_MAP.get(chapter, []):
        path = DOCS_DIR / rel
        if path.is_file():
            out.append({
                "id": rel.replace("/", "__"),
                "title": _doc_title(path),
                "path": str(path.relative_to(REPO_ROOT)),
            })
    return out


def _doc_title(path: Path) -> str:
    """Return documentation metadata used by the local browser UI."""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem.replace("_", " ").title()


def _list_doc_pages() -> list[dict[str, str]]:
    """Return all top-level docs pages for the sidebar."""
    if not DOCS_DIR.is_dir():
        return []
    entries: list[dict[str, str]] = []
    priority = [
        ("README.md", "Docs hub"),
        ("reading_order.md", "Reading order"),
        ("architecture.md", "Architecture"),
        ("notation.md", "Notation"),
        ("cookbook.md", "Cookbook"),
        ("uv.md", "uv workflow"),
    ]
    for rel, label in priority:
        path = DOCS_DIR / rel
        if path.is_file():
            entries.append({
                "id": rel.replace("/", "__"),
                "title": label,
                "subtitle": rel,
                "path": str(path.relative_to(REPO_ROOT)),
            })
    return entries


# ---------------------------------------------------------------------------
# Tiny Markdown → HTML converter (no third-party deps)
# ---------------------------------------------------------------------------


def _md_to_html(md: str) -> str:
    """Minimal Markdown converter sufficient for the project's docs.

    Supports: ATX headings, fenced code blocks, inline code, links, images,
    bold / italic, unordered + ordered lists, blockquotes, GitHub-style
    tables, horizontal rules. Everything else is rendered as paragraphs.
    """

    def esc(s: str) -> str:
        """Support request handling for the local chapter browser UI."""
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def inline(line: str) -> str:
        # protect code spans first
        """Support request handling for the local chapter browser UI."""
        out: list[str] = []
        i = 0
        while i < len(line):
            if line[i] == "`":
                end = line.find("`", i + 1)
                if end != -1:
                    out.append("<code>" + esc(line[i + 1:end]) + "</code>")
                    i = end + 1
                    continue
            out.append(esc(line[i]))
            i += 1
        text = "".join(out)
        # images: ![alt](src)
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                      r'<img alt="\1" src="\2">', text)
        # links: [text](href)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                      r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
        # bold / italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<em>\1</em>", text)
        return text

    lines = md.splitlines()
    html: list[str] = []
    in_code = False
    code_lang = ""
    code_buf: list[str] = []
    list_stack: list[str] = []  # 'ul' or 'ol' frames
    in_table = False
    in_blockquote = False

    def close_lists() -> None:
        """Support request handling for the local chapter browser UI."""
        while list_stack:
            html.append(f"</{list_stack.pop()}>")

    def close_blockquote() -> None:
        """Support request handling for the local chapter browser UI."""
        nonlocal in_blockquote
        if in_blockquote:
            html.append("</blockquote>")
            in_blockquote = False

    def close_table() -> None:
        """Support request handling for the local chapter browser UI."""
        nonlocal in_table
        if in_table:
            html.append("</tbody></table>")
            in_table = False

    def flush_blocks() -> None:
        """Support request handling for the local chapter browser UI."""
        close_lists()
        close_blockquote()
        close_table()

    i = 0
    while i < len(lines):
        raw = lines[i]
        # Fenced code blocks ```lang ... ```
        m = re.match(r"^```(\S*)\s*$", raw)
        if m and not in_code:
            close_lists()
            close_blockquote()
            close_table()
            in_code = True
            code_lang = m.group(1)
            code_buf = []
            i += 1
            continue
        if in_code:
            if raw.strip() == "```":
                cls = f' class="lang-{code_lang}"' if code_lang else ""
                html.append(f"<pre><code{cls}>" + esc("\n".join(code_buf)) + "</code></pre>")
                in_code = False
                code_lang = ""
                code_buf = []
            else:
                code_buf.append(raw)
            i += 1
            continue

        stripped = raw.strip()
        # Horizontal rule
        if re.match(r"^(---+|\*\*\*+|___+)$", stripped):
            flush_blocks()
            html.append("<hr>")
            i += 1
            continue
        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_blocks()
            level = len(m.group(1))
            html.append(f"<h{level}>{inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue
        # Blockquote
        if stripped.startswith(">"):
            close_lists()
            close_table()
            if not in_blockquote:
                html.append("<blockquote>")
                in_blockquote = True
            html.append(f"<p>{inline(stripped.lstrip('>').strip())}</p>")
            i += 1
            continue
        # Tables (GitHub flavor) — detect header + separator
        if "|" in raw and i + 1 < len(lines) and re.match(r"^\s*\|?\s*[:\-]+", lines[i + 1]):
            close_lists()
            close_blockquote()
            header_cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            html.append("<table><thead><tr>")
            for cell in header_cells:
                html.append(f"<th>{inline(cell)}</th>")
            html.append("</tr></thead><tbody>")
            in_table = True
            i += 2  # skip the header + separator rows
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                html.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
                i += 1
            close_table()
            continue
        # Lists
        m_ul = re.match(r"^(\s*)[-*+]\s+(.*)$", raw)
        m_ol = re.match(r"^(\s*)\d+\.\s+(.*)$", raw)
        if m_ul or m_ol:
            close_blockquote()
            close_table()
            tag = "ul" if m_ul else "ol"
            text = (m_ul or m_ol).group(2)
            if not list_stack or list_stack[-1] != tag:
                close_lists()
                html.append(f"<{tag}>")
                list_stack.append(tag)
            html.append(f"<li>{inline(text)}</li>")
            i += 1
            continue
        # Blank line: flush blockquote / paragraph runs
        if not stripped:
            flush_blocks()
            i += 1
            continue
        # Default: paragraph
        flush_blocks()
        html.append(f"<p>{inline(stripped)}</p>")
        i += 1

    if in_code:
        cls = f' class="lang-{code_lang}"' if code_lang else ""
        html.append(f"<pre><code{cls}>" + esc("\n".join(code_buf)) + "</code></pre>")
    close_lists()
    close_blockquote()
    close_table()
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _safe_subpath(base: Path, rel: str) -> Path | None:
    """Resolve ``rel`` against ``base`` and reject directory traversal."""
    try:
        target = (base / rel).resolve()
    except (OSError, ValueError):
        return None
    base_resolved = base.resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError:
        return None
    if not target.exists() or not target.is_file():
        return None
    return target


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _RequestError(Exception):
    """An explicitly classified client request failure, not a server exception."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _reject_json_constant(value: str) -> None:
    """Reject non-JSON NaN/infinity tokens before execution admission."""
    raise ValueError(f"invalid JSON constant: {value}")


class _Handler(BaseHTTPRequestHandler):
    """Handle local web-server state for the chapter browser UI."""
    server_version = "ActiveInferenceWeb/0.1"

    def _admit_host(self) -> tuple[str, int]:
        """Admit only an explicitly bound hostname or this socket's local address."""
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or any(character.isspace() for character in hosts[0]):
            raise _RequestError(HTTPStatus.FORBIDDEN, "A single local Host is required")
        try:
            authority = urlparse("//" + hosts[0])
            host = authority.hostname
            port = authority.port if authority.port is not None else 80
        except ValueError:
            raise _RequestError(HTTPStatus.FORBIDDEN, "Invalid Host") from None
        local_address = self.connection.getsockname()[0]
        allowed = {local_address.lower(), self.server.requested_host}
        if ipaddress.ip_address(local_address).is_loopback:
            allowed.add("localhost")
        if (
            host not in allowed or port != self.server.server_port
            or authority.username is not None or authority.password is not None
            or authority.path or authority.params or authority.query or authority.fragment
        ):
            raise _RequestError(HTTPStatus.FORBIDDEN, "Host is not bound by this server")
        return host, port

    def _execution_json(self) -> dict[str, Any]:
        """Check browser provenance and a bounded JSON envelope before dispatch."""
        host, port = self._admit_host()
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1:
            raise _RequestError(HTTPStatus.FORBIDDEN, "Multiple origins are not admitted")
        if origins:
            try:
                origin = urlparse(origins[0])
                same_origin = (
                    origin.scheme == "http" and origin.hostname == host
                    and (origin.port if origin.port is not None else 80) == port
                    and origin.username is None and origin.password is None
                    and not any((origin.path, origin.params, origin.query, origin.fragment))
                )
            except ValueError:
                same_origin = False
            if not same_origin:
                raise _RequestError(HTTPStatus.FORBIDDEN, "A same-origin request is required")
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise _RequestError(HTTPStatus.FORBIDDEN, "Cross-site execution is not admitted")
        if (
            len(self.headers.get_all("Content-Type", [])) != 1
            or self.headers.get_content_type() != "application/json"
            or self.headers.get_content_charset("utf-8").lower() not in ("utf-8", "utf8")
        ):
            raise _RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "UTF-8 application/json is required")
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) > 1:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Ambiguous request framing")
        if not lengths:
            raise _RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        if not lengths[0].isascii() or not lengths[0].isdecimal():
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
        length_text = lengths[0].lstrip("0") or "0"
        if len(length_text) > 7:
            raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "JSON body exceeds 1 MiB")
        length = int(length_text)
        if length > 1024 * 1024:
            raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "JSON body exceeds 1 MiB")
        try:
            data = json.loads(
                self.rfile.read(length).decode("utf-8"), parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Malformed UTF-8 JSON") from None
        if not isinstance(data, dict):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return data

    def _request_path(self) -> str:
        try:
            return urlparse(self.path).path
        except ValueError:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Malformed request target") from None

    # Quieter access logs.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        """Support request handling for the local chapter browser UI."""
        LOG.info("%s - " + fmt, self.address_string(), *args)

    # ---------- routing ----------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        """Support request handling for the local chapter browser UI."""
        path = self.path
        try:
            self._admit_host()
            path = self._request_path()
            if path == "/" or path == "/index.html":
                return self._send_html(render_index_html(
                    host=self.server.server_address[0],
                    port=self.server.server_address[1],
                    version=self.server.app_version,
                ))
            if path == "/static/app.css":
                return self._send_bytes(CSS.encode("utf-8"), "text/css; charset=utf-8")
            if path == "/static/app.js":
                return self._send_bytes(JS.encode("utf-8"),
                                        "application/javascript; charset=utf-8")
            if path in ("/favicon.svg", "/favicon.ico"):
                return self._send_bytes(FAVICON_SVG.encode("utf-8"),
                                        "image/svg+xml; charset=utf-8")
            if path == "/api/index":
                return self._send_json(self._index_payload())
            m = re.match(r"^/api/chapter/(\d+)$", path)
            if m:
                return self._send_json(self._chapter_payload(int(m.group(1))))
            m = re.match(r"^/api/extra/([a-z0-9_]+)$", path)
            if m:
                return self._send_json(self._extra_payload(m.group(1)))
            m = re.match(r"^/api/demo/([a-z0-9_]+)$", path)
            if m:
                return self._send_json(self._demo_payload(m.group(1)))
            m = re.match(r"^/api/doc/(.+)$", path)
            if m:
                return self._send_json(self._doc_payload(unquote(m.group(1))))
            m = re.match(r"^/figures/extras/([a-z0-9_]+)/(.+)$", path)
            if m:
                topic = m.group(1)
                name = unquote(m.group(2))
                return self._serve_static_file(_extra_figure_dir(topic), name)
            m = re.match(r"^/figures/demo/([a-z0-9_]+)/(.+)$", path)
            if m:
                slug = m.group(1)
                name = unquote(m.group(2))
                return self._serve_static_file(_demo_figure_dir(slug), name)
            m = re.match(r"^/figures/(\d{2})/(.+)$", path)
            if m:
                chapter = int(m.group(1))
                name = unquote(m.group(2))
                return self._serve_static_file(_chapter_figure_dir(chapter), name)
            m = re.match(r"^/docs-raw/(.+)$", path)
            if m:
                return self._serve_static_file(DOCS_DIR, unquote(m.group(1)))
            return self._send_error(HTTPStatus.NOT_FOUND, f"No route for {path!r}")
        except _RequestError as exc:
            return self._send_error(exc.status, str(exc))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("GET %s failed", path)
            return self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        """Support request handling for the local chapter browser UI."""
        path = self.path
        try:
            path = self._request_path()
            if path not in ("/api/run", "/api/launch-interactive"):
                self._admit_host()
                return self._send_error(HTTPStatus.NOT_FOUND, f"No POST route for {path!r}")
            data = self._execution_json()
            if path == "/api/run":
                return self._send_json(self._run_payload(data))
            return self._send_json(self._launch_interactive_payload(data))
        except _RequestError as exc:
            return self._send_error(exc.status, str(exc))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("POST %s failed", path)
            return self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # ---------- payloads --------------------------------------------------
    def _index_payload(self) -> dict[str, Any]:
        """Support request handling for the local chapter browser UI."""
        snapshot = discover_snapshot()
        chapters = []
        for entry in discover_chapters(snapshot=snapshot, include_interactive=True):
            scripts = entry.scripts
            kind_counts: dict[str, int] = {}
            for s in scripts:
                kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
            figures = _list_figures(entry.number, entry.scripts)
            chapters.append({
                "number": entry.number,
                "title": entry.title,
                "relative": entry.relative,
                "scripts": [_script_meta(s) for s in scripts],
                "kind_counts": kind_counts,
                "figure_count": len(figures),
                "subtitle": _chapter_subtitle(entry.number, entry.path),
            })
        extras = []
        for entry in discover_extras(snapshot=snapshot, include_interactive=True):
            scripts = entry.scripts
            kind_counts: dict[str, int] = {}
            for s in scripts:
                kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
            figures = _list_extra_figures(entry.slug, entry.scripts)
            extras.append({
                "slug": entry.slug,
                "title": entry.title,
                "relative": entry.relative,
                "scripts": [_script_meta(s) for s in scripts],
                "kind_counts": kind_counts,
                "figure_count": len(figures),
                "subtitle": entry.family,
                "family": entry.family,
                "summary": entry.summary,
                "chapters": list(entry.chapters),
                "sections": list(entry.sections),
            })
        demos = []
        for entry in discover_demos(snapshot=snapshot, include_interactive=True):
            scripts = entry.scripts
            kind_counts: dict[str, int] = {}
            for s in scripts:
                kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
            figures = _list_demo_figures(entry.slug, scripts)
            demos.append({
                "slug": entry.slug,
                "title": entry.title,
                "relative": entry.relative,
                "scripts": [_script_meta(s) for s in scripts],
                "kind_counts": kind_counts,
                "figure_count": len(figures),
                "subtitle": "Application demo",
                "summary": entry.summary,
                "chapters": list(entry.chapters),
            })
        return {
            "chapters": chapters,
            "extras": extras,
            "demos": demos,
            "docs": _list_doc_pages(),
            "repo": str(REPO_ROOT),
        }

    def _chapter_payload(self, number: int) -> dict[str, Any]:
        """Return chapter-specific metadata used for discovery and display."""
        snapshot = discover_snapshot()
        entry = snapshot.chapters.get(number)
        if entry is None:
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown chapter {number!r}")
        scripts = entry.scripts
        chapter_dir = entry.path
        readme_path = chapter_dir / "README.md"
        return {
            "number": number,
            "title": f"Chapter {number:02d}",
            "subtitle": _chapter_subtitle(number, chapter_dir),
            "relative": str(chapter_dir.relative_to(REPO_ROOT)),
            "scripts": [_script_meta(s) for s in scripts],
            "figures": _list_figures(number, scripts),
            "docs": _list_chapter_docs(number),
            "readme_html": _md_to_html(readme_path.read_text(encoding="utf-8",
                                                              errors="replace"))
            if readme_path.is_file() else None,
            "readme_source": str(readme_path.relative_to(REPO_ROOT))
            if readme_path.is_file() else None,
        }

    def _extra_payload(self, topic: str) -> dict[str, Any]:
        """Return extras-topic metadata used for discovery and display."""
        snapshot = discover_snapshot()
        entry = snapshot.extras.get(topic)
        if entry is None:
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown extras topic {topic!r}")
        scripts = entry.scripts
        topic_dir = entry.path
        readme_path = topic_dir / "README.md"
        registry = _extra_registry_payload(topic)
        return {
            "slug": topic,
            "title": registry["title"],
            "subtitle": registry["family"],
            "relative": str(topic_dir.relative_to(REPO_ROOT)),
            "scripts": [_script_meta(s) for s in scripts],
            "figures": _list_extra_figures(topic, scripts),
            "docs": [],
            **registry,
            "readme_html": _md_to_html(readme_path.read_text(encoding="utf-8",
                                                              errors="replace"))
            if readme_path.is_file() else None,
            "readme_source": str(readme_path.relative_to(REPO_ROOT))
            if readme_path.is_file() else None,
        }

    def _demo_payload(self, slug: str) -> dict[str, Any]:
        """Return demo-topic metadata used for discovery and display."""
        snapshot = discover_snapshot()
        entry = snapshot.demos.get(slug)
        if entry is None:
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown demo topic {slug!r}")
        scripts = entry.scripts
        topic_dir = entry.path
        readme_path = topic_dir / "README.md"
        registry = _demo_registry_payload(slug)
        return {
            "slug": slug,
            "title": registry["title"],
            "subtitle": "Application demo",
            "relative": str(topic_dir.relative_to(REPO_ROOT)),
            "scripts": [_script_meta(s) for s in scripts],
            "figures": _list_demo_figures(slug, scripts),
            "docs": [],
            **registry,
            "readme_html": _md_to_html(readme_path.read_text(encoding="utf-8",
                                                              errors="replace"))
            if readme_path.is_file() else None,
            "readme_source": str(readme_path.relative_to(REPO_ROOT))
            if readme_path.is_file() else None,
        }

    def _doc_payload(self, doc_id: str) -> dict[str, Any]:
        """Return documentation metadata used by the local browser UI."""
        rel = doc_id.replace("__", "/")
        path = _safe_subpath(DOCS_DIR, rel)
        if path is None or path.suffix.lower() != ".md":
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown doc {rel!r}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "title": _doc_title(path),
            "source": str(path.relative_to(REPO_ROOT)),
            "html": _md_to_html(text),
        }

    def _requested_script(self, data: dict[str, Any]) -> ScriptEntry:
        """Resolve exactly one typed owner and script from one fresh snapshot."""
        selectors = [key for key in ("chapter", "topic", "demo") if key in data]
        name = data.get("script")
        if (
            len(selectors) != 1 or set(data) - {"chapter", "topic", "demo", "script"}
            or not isinstance(name, str) or not name
            or Path(name).name != name or "\\" in name
        ):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Provide one owner and a script filename")
        selector = selectors[0]
        value = data[selector]
        if selector == "chapter":
            if type(value) is not int or value < 1:
                raise _RequestError(HTTPStatus.BAD_REQUEST, "chapter must be a positive integer")
        elif not isinstance(value, str) or re.fullmatch(r"[a-z0-9_]+", value) is None:
            raise _RequestError(HTTPStatus.BAD_REQUEST, f"Invalid {selector} slug")
        snapshot = discover_snapshot()
        owners = {"chapter": snapshot.chapters, "topic": snapshot.extras, "demo": snapshot.demos}
        entry = owners[selector].get(value)
        if entry is None:
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown {selector} {value!r}")
        match = next((script for script in entry.scripts if script.name == name), None)
        if match is None or not match.path.resolve().is_relative_to(REPO_ROOT.resolve()):
            raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown contained script {name!r}")
        return match

    def _run_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run one admitted non-interactive script; preserve its process result."""
        match = self._requested_script(data)
        if match.kind == "interactive":
            raise _RequestError(
                HTTPStatus.BAD_REQUEST, "Interactive scripts require /api/launch-interactive",
            )
        completed = run_script(match, save=True, capture_output=True,
                               timeout=300, quiet=True)
        return {
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "").splitlines()[-25:],
            "stderr_tail": (completed.stderr or "").splitlines()[-25:],
        }

    def _launch_interactive_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Support request handling for the local chapter browser UI."""
        match = self._requested_script(data)
        if match.kind != "interactive":
            raise _RequestError(HTTPStatus.BAD_REQUEST, "The selected script is not interactive")
        env = _build_env()
        env.pop("MPLBACKEND", None)  # let it pick the system display
        proc = subprocess.Popen(  # noqa: S603 — sandboxed to discovered scripts
            [sys.executable, str(match.path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"pid": proc.pid, "script": match.name}

    # ---------- response helpers ------------------------------------------
    def _serve_static_file(self, base: Path, rel: str) -> None:
        """Serve a static local artifact through the web interface."""
        path = _safe_subpath(base, rel)
        if path is None:
            return self._send_error(HTTPStatus.NOT_FOUND, f"No such file {rel!r}")
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        try:
            data = path.read_bytes()
        except OSError as exc:
            return self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        self._send_bytes(data, mime)

    def _send_html(self, html: str) -> None:
        """Serialize and send an HTTP response from the local server."""
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, payload: Any) -> None:
        """Serialize and send an HTTP response from the local server."""
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, mime: str,
                    status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serialize and send an HTTP response from the local server."""
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        """Serialize and send an HTTP response from the local server."""
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    """Adds the version string used by the SPA shell."""

    daemon_threads = True
    app_version: str = "0.1.0"
    serve_thread: threading.Thread | None = None

    def __init__(self, server_address, handler, bind_and_activate=True):
        """Capture the explicitly selected host before stdlib DNS normalization."""
        self.requested_host = server_address[0].lower()
        super().__init__(server_address, handler, bind_and_activate)


def _resolve_version() -> str:
    """Support request handling for the local chapter browser UI."""
    try:
        from .. import __version__
        return __version__
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _pick_free_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if free, else bind to ephemeral and report it."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, preferred))
        return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]


def run_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    block: bool = True,
    log_level: int = logging.INFO,
) -> _Server:
    """Start the web UI.

    When ``block=True`` (the default) this serves forever until interrupted
    with ``Ctrl+C`` / ``SIGINT``. When ``block=False`` the server runs in a
    background thread and the constructor returns the underlying
    :class:`ThreadingHTTPServer` instance so callers (e.g. tests) can shut
    it down manually.
    """
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(message)s")
    port = _pick_free_port(host, port)
    server = _Server((host, port), _Handler)
    server.app_version = _resolve_version()
    url = f"http://{host}:{port}/"
    LOG.info("Active Inference web UI ready: %s", url)
    LOG.info("Repo root: %s", REPO_ROOT)
    if not block:
        thread = threading.Thread(target=server.serve_forever, daemon=True,
                                  name="ActiveInferenceWebServer")
        server.serve_thread = thread
        thread.start()
        if open_browser:
            _open_browser_async(url)
        return server
    if open_browser:
        _open_browser_async(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Received Ctrl+C; shutting down.")
    finally:
        server.server_close()
    return server


def _open_browser_async(url: str) -> None:
    """Open the local web interface asynchronously after startup."""
    def _open() -> None:
        """Open the local web interface asynchronously after startup."""
        time.sleep(0.6)  # give serve_forever a chance to start accepting
        try:
            webbrowser.open(url, new=2)
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Could not open browser for %s", url, exc_info=True)

    threading.Thread(target=_open, daemon=True, name="OpenBrowser").start()


# Convenience alias used by ``active_inference.run_web``.
launch = run_server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(
        prog="active_inference.web",
        description="Launch the local web UI for the chapter orchestrators.",
    )
    p.add_argument("--host", default=os.environ.get("AIWEB_HOST", DEFAULT_HOST),
                   help=f"Bind address (default: {DEFAULT_HOST}).")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("AIWEB_PORT", DEFAULT_PORT)),
                   help=f"TCP port (default: {DEFAULT_PORT}). "
                        "If taken, an ephemeral port is chosen.")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the default web browser.")
    p.add_argument("--verbose", action="store_true",
                   help="DEBUG-level server logging.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the user-facing entry point for this interface."""
    args = _parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        block=True,
        log_level=logging.DEBUG if args.verbose else logging.INFO,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
