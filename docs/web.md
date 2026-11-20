# Local web UI

The repository ships a small browser interface for browsing chapter
figures, animations, and scripts without leaving the local machine. It
is implemented in `src/active_inference/web/` and is a peer of the text
menu — both share the same discovery layer in
`src/active_inference/menu/runner.py`.

> The footer of every page links to the
> [Active Inference Institute](https://activeinference.institute/) and to
> [textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/)
> for cohort registration. If you fork the UI, please keep that
> attribution.

## Launch

```bash
./run.sh --web                          # default http://127.0.0.1:8765/
./run.sh --web --no-browser             # don't auto-open the browser
./run.sh --web --port 8080              # custom port (ephemeral if taken)
uv run python -m active_inference.web   # equivalent
uv run active-inference-web --verbose   # PEP 621 console script
```

From inside the text menu (`./run.sh`), press `w` to start the same
server.

The server binds to `127.0.0.1` by default — the page is reachable only
from the local machine. Set `AIWEB_HOST=0.0.0.0` / `--host 0.0.0.0` to
expose it on the LAN (use cautiously: there is no authentication).

## What's on the page

The sidebar lists detected chapters, extras topics, demo topics, and the
available documentation pages. Chapter, extras, and demo tabs share
owner-qualified script controls and the relevant gallery/documentation views:

1. **Figure gallery** — every PNG / GIF in
   `output/figures/chapter_<N>/`. Click a thumbnail to open the full
   file.
2. **Numbered examples & concept demos** — the scripts that mirror book
   examples. Each row has:
   - **Show command** — opens a dialog with the exact `uv run python`
     command to invoke the script by hand.
   - **Render** — POSTs to `/api/run`; the server runs the script with
     `--save` under `MPLBACKEND=Agg` and the gallery refreshes.
3. **Animations** — same pattern, for `animation_*.py`.
4. **Visualizations** — same pattern, for `visualize_*.py`.
5. **Interactive scripts** — scripts containing `interactive` in the
   name, including registry-backed extras launchers. The **Launch on host**
   button starts the slider window on the machine running the server.
6. **Chapter documentation** — links to the matching pages under
   `docs/chapters/`, `docs/topics/`, and `docs/statistics/`. Clicking
   a link renders the Markdown inside the same tab.

## HTTP surface

All endpoints return JSON unless otherwise noted.

| Method | Path                              | Purpose |
|--------|-----------------------------------|---------|
| GET    | `/`                               | SPA shell (HTML). |
| GET    | `/favicon.svg` (or `.ico`)        | Inline SVG favicon (mono with red dot). |
| GET    | `/static/app.css`, `app.js`       | Inline asset bundles. |
| GET    | `/api/index`                      | Chapters, extras, and demos with script inventories, per-kind counts, and metadata. |
| GET    | `/api/chapter/<N>`                | Scripts (with size, mtime, docstring, example number), figures (with size, mtime, dimensions, generated_by), inline-rendered README, and chapter doc links. |
| GET    | `/api/extra/<topic>`              | Extras scripts, figures, README HTML, family, summary, chapters, and book sections. |
| GET    | `/api/demo/<slug>`                | Demo scripts, figures, README HTML, summary, chapters, and source API metadata. |
| GET    | `/api/doc/<id>`                   | Rendered Markdown for one doc page. |
| GET    | `/figures/<NN>/<file>`            | Static file under `output/figures/`. |
| GET    | `/figures/extras/<topic>/<file>`  | Static file under `output/figures/extras/<topic>/`. |
| GET    | `/figures/demo/<slug>/<file>`     | Static file under the selected demo figure directory. |
| GET    | `/docs-raw/<path>`                | Static file under `docs/`. |
| POST   | `/api/run`                        | Render one owner-qualified chapter, extras, or demo script. |
| POST   | `/api/launch-interactive`         | Launch a slider script on the host. |

Static paths and selected script targets must remain within their admitted
repository roots.

Execution requests require UTF-8 `application/json`, one unambiguous
Content-Length, and a body no larger than 1 MiB. Supply `script` plus exactly
one owner: integer `chapter`, string `topic`, or string `demo`. For example:

```json
{"demo":"selected_demo","script":"visualize_selected_demo.py"}
```

The server validates Host and rejects foreign/null Origin and cross-site
browser execution requests before dispatch. Requests without browser Origin
metadata remain available to local CLI clients. These checks are not
authentication; do not expose this execution service to untrusted users.

Malformed requests return 400, forbidden provenance 403, missing resources
404, missing length 411, oversized bodies 413, unsupported media types 415,
and internal failures 500. A completed script response preserves its actual
return code; dispatch success does not imply scientific success.

## Design

- **Stdlib only.** Uses `http.server`, `socketserver`, `subprocess`,
  and a tiny Markdown converter built into `server.py`. No Flask, no
  Jinja, no JS framework, no CDN dependencies.
- **Single page, vanilla JS.** Tab switching, gallery rendering, the
  filter input, the lightbox, and the action menu are all in
  `templates.JS`.
- **Operation-local discovery.** `discover_snapshot()` captures immutable
  chapter/extras/demo inventories. Menu, batch, and HTTP operations reuse
  their capture without rebuilding shared global maps. Later operations see
  later filesystem changes. This is not an atomic filesystem snapshot.
- **Owner-qualified execution.** Browser requests identify chapter, extras,
  or demo ownership explicitly. Command dialogs show the exact relative
  script path; completion refreshes that owner's tab.
- **Shared subprocess runner.** Non-interactive rendering uses `run_script`
  with `--save` and `MPLBACKEND=Agg`. Interactive launch accepts a host
  process launch, not proof that a GUI window opened.

## Visual identity

- **Mono palette.** Pure black background, near-black panels
  (`#121212` / `#161616`), white / off-white text, gray chrome.
- **A single red accent (`#ef4444` family)**, used sparingly: the
  active sidebar tab, the **Render** button, the H2 accent bar, the
  `cached figures` metric, the favicon dot, error toasts, and the
  animation badge. Everything else is greyscale so the red keeps its
  signal value.
- **No purple / green / yellow / blue.** If you find yourself reaching
  for a new color, use a different *gray* tone instead.

## Power-user features

- Press <kbd>/</kbd> from any chapter tab to focus the filter input.
  Filter substrings match against script names, docstrings, figure
  filenames, and the script that generated each figure.
- Press <kbd>Esc</kbd> to clear the filter, close a dialog, or close
  the lightbox.
- Click any figure to open the lightbox. Click the "generator" name
  in a figure caption to scroll to the script row that produced it
  (briefly highlighted in red).
  Demo generator names resolve against the captured script inventory, including
  the `.py` suffix and figure-name prefixes; unmatched figures have no generator
  link rather than an invented script destination.
- The **Command** button on each script row opens a copyable shell
  command — `Copy` puts it on the clipboard via the Async Clipboard
  API.
- If a render fails, the output dialog automatically opens with the
  last 25 lines of `stderr` (with a tab to view `stdout`).

## When to use which front end

| You want to… | Use |
|---|---|
| Re-render every figure for CI / a fresh checkout | `./run.sh --all` or `scripts/run_all_figures.py` |
| Pick a script from a quick keyboard menu | `./run.sh` |
| Browse generated figures alongside chapter docs | `./run.sh --web` |
| Embed discovery in another Python program | `from active_inference.menu.runner import …` |
| Embed the server in another Python program | `from active_inference.web import run_server` (with `block=False`) |

## Troubleshooting

- *"Port 8765 already in use"* — the server falls back to an ephemeral
  port automatically and prints the URL. To pin a specific port, kill
  the conflicting process (or pass `--port`).
- *"Launch on host" does nothing when accessed remotely* — interactive
  matplotlib windows need a display. Run the server on the machine
  whose display you want to use; SSH X-forwarding works too.
- *Figures don't refresh after Render* — the JS reloads the chapter
  tab on success. If the network panel shows `404 GET /figures/…`,
  the chapter script may have failed; click **Show command** and try
  running the script in the terminal to read the traceback.
