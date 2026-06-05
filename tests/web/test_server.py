"""Smoke tests for the local web UI.

The server is started in-process on an ephemeral port so the tests do not
collide with any developer-launched instance. Requests go through
``urllib.request`` to avoid pulling in `requests` as a test dep.
"""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from active_inference.web import server as web_server
from active_inference.web.templates import CSS, JS, render_index_html
from active_inference.extra_topics import extra_topic_slugs


@pytest.fixture(scope="module")
def server():
    srv = web_server.run_server(
        host="127.0.0.1", port=0, open_browser=False, block=False,
    )
    # Wait briefly for the thread to start accepting.
    time.sleep(0.2)
    yield srv
    srv.shutdown()
    if srv.serve_thread is not None:
        srv.serve_thread.join(timeout=2)
    srv.server_close()


def _url(server, path: str) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{path}"


def _get(server, path: str):
    with urlopen(_url(server, path), timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


def _post(server, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = Request(_url(server, path), data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _close_http_error(exc_info) -> HTTPError:
    err = exc_info.value
    err.close()
    return err


class TestRoutes:
    def test_index_html(self, server):
        status, mime, body = _get(server, "/")
        assert status == 200
        assert "text/html" in mime
        text = body.decode("utf-8")
        assert "Active Inference Fundamentals" in text
        assert "/static/app.js" in text

    def test_static_css(self, server):
        status, mime, body = _get(server, "/static/app.css")
        assert status == 200
        assert "text/css" in mime
        assert b":root" in body

    def test_static_js(self, server):
        status, mime, body = _get(server, "/static/app.js")
        assert status == 200
        assert "javascript" in mime
        assert b"renderChapter" in body

    def test_index_api_lists_chapters(self, server):
        status, _, body = _get(server, "/api/index")
        assert status == 200
        data = json.loads(body)
        numbers = sorted(c["number"] for c in data["chapters"])
        assert numbers[:3] == [1, 2, 3]
        extra_slugs = {e["slug"] for e in data["extras"]}
        assert "entropy" in extra_slugs
        assert "variational_free_energy" in extra_slugs
        assert set(extra_topic_slugs()).issubset(extra_slugs)
        # Doc pages should at least include the architecture / cookbook entries.
        titles = {d["title"] for d in data["docs"]}
        assert "Architecture" in titles
        assert "Cookbook" in titles

    def test_chapter_payload_shape(self, server):
        status, _, body = _get(server, "/api/chapter/1")
        assert status == 200
        data = json.loads(body)
        assert data["number"] == 1
        names = {s["name"] for s in data["scripts"]}
        assert "01_box_scenario.py" in names
        assert isinstance(data["figures"], list)
        assert isinstance(data["docs"], list)

    def test_extra_payload_shape(self, server):
        status, _, body = _get(server, "/api/extra/entropy")
        assert status == 200
        data = json.loads(body)
        assert data["slug"] == "entropy"
        assert data["family"] == "Information And Variational Inference"
        assert data["sections"]
        names = {s["name"] for s in data["scripts"]}
        assert "visualize_entropy.py" in names
        assert "simulate_entropy.py" in names
        assert "interactive_entropy.py" in names
        assert isinstance(data["figures"], list)
        assert data["readme_html"]

    def test_unknown_chapter_returns_500(self, server):
        with pytest.raises(HTTPError) as exc:
            _get(server, "/api/chapter/99")
        err = _close_http_error(exc)
        assert err.code in (400, 500)

    def test_doc_api_renders_markdown(self, server):
        status, _, body = _get(server, "/api/doc/architecture.md")
        assert status == 200
        data = json.loads(body)
        assert data["title"]
        assert "<h1>" in data["html"]
        assert "</p>" in data["html"] or "<table>" in data["html"]

    def test_path_traversal_rejected(self, server):
        with pytest.raises(HTTPError) as exc:
            _get(server, "/docs-raw/..%2F..%2Fpyproject.toml")
        err = _close_http_error(exc)
        assert err.code == 404

    def test_figure_serves_when_present(self, server):
        # Chapter 1 has been rendered as part of the menu smoke run earlier;
        # if not, we just skip — the route logic is exercised by the 404 test.
        import urllib.error

        try:
            status, mime, body = _get(server, "/figures/01/01_box_scenario_stream.png")
        except urllib.error.HTTPError as exc:
            exc.close()
            pytest.skip("Chapter 1 figures not rendered yet")
        assert status == 200
        assert mime.startswith("image/")
        assert len(body) > 100


class TestMarkdownConverter:
    def test_headings(self):
        html = web_server._md_to_html("# Title\n\nSome **bold** text.\n")
        assert "<h1>Title</h1>" in html
        assert "<strong>bold</strong>" in html

    def test_code_block_preserves_indent(self):
        md = "```python\ndef f(x):\n    return x\n```\n"
        html = web_server._md_to_html(md)
        assert '<pre><code class="lang-python">' in html
        assert "def f(x):" in html
        assert "    return x" in html

    def test_table(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        html = web_server._md_to_html(md)
        assert "<table>" in html
        assert "<th>a</th>" in html
        assert "<td>1</td>" in html


class TestRunEndpoint:
    def test_run_invalid_script_returns_error(self, server):
        with pytest.raises(HTTPError) as exc:
            _post(server, "/api/run",
                  {"chapter": 1, "script": "nonexistent_script.py"})
        err = _close_http_error(exc)
        assert err.code == 500  # ValueError → 500

    def test_run_rejects_interactive_scripts(self, server):
        # Chapter 2 has interactive_explorer.py
        with pytest.raises(HTTPError) as exc:
            _post(server, "/api/run",
                  {"chapter": 2, "script": "interactive_explorer.py"})
        err = _close_http_error(exc)
        assert err.code == 500

    def test_run_chapter_1_script(self, server):
        # 04_inverse_problem is cheap and exercises numpy + matplotlib end-to-end.
        status, data = _post(server, "/api/run",
                             {"chapter": 1, "script": "04_inverse_problem.py"})
        assert status == 200
        assert data["returncode"] == 0

    def test_run_extra_script(self, server):
        status, data = _post(server, "/api/run",
                             {"topic": "entropy", "script": "visualize_entropy.py"})
        assert status == 200
        assert data["returncode"] == 0


class TestTemplates:
    def test_render_index_html_substitutes(self):
        html = render_index_html(host="127.0.0.1", port=1234, version="9.9.9")
        assert "127.0.0.1:1234" in html
        assert "v9.9.9" in html

    def test_index_html_has_required_elements(self):
        html = render_index_html(host="127.0.0.1", port=8765, version="0.1.0")
        # Theme + accessibility metadata
        assert 'name="theme-color" content="#000000"' in html
        assert 'rel="icon"' in html
        assert "/favicon.svg" in html
        # Core UI mounts
        assert 'id="tabs"' in html
        assert 'id="content"' in html
        assert 'id="dialog"' in html
        assert 'id="lightbox"' in html
        assert 'id="toast"' in html

    def test_footer_attribution(self):
        html = render_index_html(host="127.0.0.1", port=8765, version="0.1.0")
        assert "activeinference.institute" in html
        assert "textbook-group.activeinference.institute" in html
        assert "Active Inference Institute" in html

    def test_js_about_panel_mentions_textbook_group(self):
        # The About panel is rendered client-side from this JS string, so we
        # assert on the source rather than spinning a browser.
        assert "textbook-group.activeinference.institute" in JS
        assert "Active Inference Institute" in JS

    def test_css_and_js_non_empty(self):
        assert len(CSS) > 500
        assert len(JS) > 500

    def test_css_uses_monochrome_palette(self):
        # Black background, white-ish text, red accent — and *not* the old
        # navy / sky-blue palette.
        assert "--bg: #000000" in CSS
        assert "--accent: #ef4444" in CSS
        assert "#38bdf8" not in CSS  # sky blue removed
        assert "#0f172a" not in CSS  # slate navy removed
        assert "#a78bfa" not in CSS  # purple removed


class TestEnrichedMetadata:
    def test_index_includes_kind_counts_and_subtitle(self, server):
        status, _, body = _get(server, "/api/index")
        assert status == 200
        data = json.loads(body)
        ch1 = next(c for c in data["chapters"] if c["number"] == 1)
        assert ch1["subtitle"] == "The Hypothesis-Testing Brain"
        assert ch1["kind_counts"]["concept"] == 4
        assert ch1["figure_count"] >= 0
        assert "repo" in data
        entropy = next(e for e in data["extras"] if e["slug"] == "entropy")
        assert entropy["kind_counts"]["visualize"] == 1
        assert entropy["kind_counts"]["simulate"] == 1
        assert entropy["family"] == "Information And Variational Inference"
        assert entropy["figure_count"] >= 0

    def test_script_meta_keys(self, server):
        status, _, body = _get(server, "/api/chapter/2")
        data = json.loads(body)
        script = next(s for s in data["scripts"]
                      if s["name"] == "example_2_2_linear_probabilistic.py")
        assert script["docstring"]
        assert script["example_number"] == "2.2"
        assert "size_human" in script
        assert "mtime_human" in script

    def test_chapter_payload_has_readme(self, server):
        _, _, body = _get(server, "/api/chapter/1")
        data = json.loads(body)
        assert data["readme_html"]
        assert "<h1>" in data["readme_html"]
        assert data["readme_source"].endswith("README.md")

    def test_figure_meta_when_present(self, server):
        _, _, body = _get(server, "/api/chapter/1")
        data = json.loads(body)
        figures = [f for f in data["figures"] if f["extension"] == "png"]
        if not figures:
            pytest.skip("No figures rendered yet")
        f = figures[0]
        assert f["size"] > 0
        assert f["size_human"]
        assert f["mtime_human"]
        # PNG dimensions should parse from the file header.
        assert f["width"] is not None and f["width"] > 0


class TestFavicon:
    def test_favicon_svg(self, server):
        status, mime, body = _get(server, "/favicon.svg")
        assert status == 200
        assert "svg" in mime
        assert b"<svg" in body
        assert b"#ef4444" in body  # the red dot

    def test_favicon_ico_alias_returns_svg(self, server):
        status, mime, body = _get(server, "/favicon.ico")
        # Same bytes / mime as /favicon.svg — we don't actually ship an ICO.
        assert status == 200
        assert "svg" in mime
        assert b"<svg" in body


class TestImageHelpers:
    def test_human_bytes(self):
        from active_inference.web.server import _human_bytes
        assert _human_bytes(500) == "500 B"
        assert _human_bytes(2048) == "2.0 KB"
        assert _human_bytes(3 * 1024 * 1024) == "3.0 MB"

    def test_example_number_extraction(self):
        from active_inference.web.server import _extract_example_number
        assert _extract_example_number("example_2_5_nonlinear.py") == "2.5"
        assert _extract_example_number("example_3_10_big.py") == "3.10"
        assert _extract_example_number("01_box_scenario.py") == "01"
        assert _extract_example_number("animation_em_steps.py") is None

    def test_chapter_subtitle_static_map(self):
        from active_inference.web.server import _chapter_subtitle
        assert _chapter_subtitle(1) == "The Hypothesis-Testing Brain"
        assert _chapter_subtitle(2) == "Hidden State Estimation"
        assert _chapter_subtitle(3) == "Combining Learning and Inference"

    def test_png_dimensions(self):
        # Use a known chapter-1 figure if it exists.
        from active_inference.web.server import _image_dimensions, OUTPUT_DIR
        figs = list((OUTPUT_DIR / "chapter_01").glob("*.png")) if OUTPUT_DIR.is_dir() else []
        if not figs:
            pytest.skip("No PNGs to test dimensions against")
        w, h = _image_dimensions(figs[0])
        assert isinstance(w, int) and isinstance(h, int)
        assert w > 0 and h > 0
