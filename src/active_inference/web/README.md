# `active_inference.web` — local browser UI

A stdlib-only HTTP server that exposes a tab-per-chapter web interface
over the same discovery layer the text menu uses. Launched from
[`run.sh --web`](../../../run.sh).

## What you get

A single-page app at `http://127.0.0.1:8765/` with:

- **One tab per chapter** with subtitle (e.g. "The Hypothesis-Testing
  Brain") and a per-chapter sidebar badge showing
  `cached-figures / scripts`.
- **Metric strip** at the top of every chapter tab: example count,
  animation count, visualization count, cached-figure count (red).
- **Figure gallery** split into "Figures" (PNG/JPG/SVG) and
  "Animations" (GIF). Each figure card shows the filename, format
  badge, dimensions, file size, age, and the script that wrote it
  (click to scroll to its row). Clicking a thumbnail opens a
  full-screen lightbox.
- **Script rows** showing kind chip, name, the book example number
  (e.g. `§ 2.5`), the first line of the script's docstring, file path,
  and file size / modtime chips.
- **Filter input** above the gallery — press <kbd>/</kbd> to focus.
  Substring match against script names, docstrings, figure filenames,
  and generators. <kbd>Esc</kbd> clears.
- **Render buttons.** Each non-interactive script has a "Render"
  button that POSTs back to the server, runs the orchestrator with
  `--save` under `MPLBACKEND=Agg`, and refreshes the gallery on
  completion. Failures auto-open a stderr / stdout dialog.
- **Interactive launchers.** `interactive_*.py` scripts get a "Launch
  on host" button that starts the slider window on the machine
  running the server.
- **Inline chapter README** is rendered server-side at the bottom of
  the tab.
- **Docs sidebar** with the top-level cross-cutting pages plus a
  per-chapter docs list (concept map, topic walkthroughs, statistics
  references).
- **Mono + red theme.** Pure black background, white text, grey
  chrome; red is reserved for active tabs, primary actions, errors,
  and the favicon dot.

## Module layout

| File             | Role |
|------------------|------|
| `server.py`      | `http.server` subclass + route handlers, subprocess runner, tiny Markdown converter. |
| `templates.py`   | Inline CSS / JS / HTML — no CDN dependencies. |
| `__init__.py`    | Public API: `launch`, `run_server`, `main`. |
| `__main__.py`    | Wires `python -m active_inference.web`. |

## Entry points

```bash
# Launch (default: 127.0.0.1:8765, auto-opens the browser)
./run.sh --web

# Variants
./run.sh --web --no-browser --port 8080
uv run python -m active_inference.web --verbose
uv run active-inference-web              # PEP 621 console script

# From within the text menu, press `w`
./run.sh
> w
```

If the chosen port is taken, the server falls back to an ephemeral one
and prints the actual URL.

## HTTP routes

| Method | Path                            | Returns |
|--------|---------------------------------|---------|
| GET    | `/`                             | SPA shell |
| GET    | `/favicon.svg` (or `.ico`)      | Inline SVG favicon (mono + red dot) |
| GET    | `/static/app.css`               | CSS bundle |
| GET    | `/static/app.js`                | JS bundle |
| GET    | `/api/index`                    | `{chapters: [{number, title, subtitle, scripts[], kind_counts, figure_count, relative}], docs, repo}` |
| GET    | `/api/chapter/<N>`              | `{scripts (with size, mtime, docstring, example_number), figures (with size, mtime, width, height, generated_by), docs, readme_html, readme_source}` |
| GET    | `/api/doc/<id>`                 | `{title, source, html}` (Markdown → HTML) |
| GET    | `/figures/<NN>/<name>`          | Static file from `output/figures/chapter_<NN>/` |
| GET    | `/docs-raw/<path>`              | Static file from `docs/` |
| POST   | `/api/run`                      | Run a non-interactive script and return `{returncode, stdout_tail, stderr_tail}` |
| POST   | `/api/launch-interactive`       | Open a slider script on the host and return `{pid, script}` |

Every path is rejected if it escapes its allowed base (figures /
docs /chapters) via the `_safe_subpath` resolver.

## Design constraints

- **Stdlib only.** `http.server`, `socketserver`, `subprocess`, `json`,
  `re`, `urllib.parse`, `webbrowser`. No Flask, no Jinja, no JS
  framework. This matches the `menu/` constraint and keeps the install
  surface tiny.
- **No domain logic.** All numerical work happens in the chapter
  scripts, which the server invokes via `active_inference.menu.runner`.
- **Single-page app.** Tabs switch client-side; the server is pure JSON
  + static assets. Markdown → HTML conversion is done server-side by a
  small ~150-line converter in `server.py` so the browser doesn't need
  a library.
- **Localhost only by default.** Binding to `0.0.0.0` is possible via
  `--host`, but the project intentionally defaults to `127.0.0.1`.

## Programmatic surface

```python
from active_inference.web import run_server, launch

# Blocking (like the CLI):
run_server(host="127.0.0.1", port=8765)

# Non-blocking (for tests / embedding):
srv = run_server(block=False, open_browser=False, port=0)
try:
    ...  # hit srv.server_address with urllib / requests
finally:
    srv.shutdown()
    srv.server_close()
```
