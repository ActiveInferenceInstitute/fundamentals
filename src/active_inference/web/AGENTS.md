# `web/` — agent guide

User-facing local web UI launched by `./run.sh --web`. Rules:

1. **Stdlib only.** No Flask, no Jinja, no requests, no JS framework, no
   CDN links. The page must render with the network unplugged.
2. **No domain logic.** This is presentation + subprocess plumbing.
   Numerical work belongs in `core/` / `estimators/`. Discovery belongs
   in `menu/runner.py` (which this module reuses).
3. **Path safety.** Every file served on a public path passes through
   `server._safe_subpath`, which canonicalizes the request and refuses
   anything outside the allowed base directory.

## Files

| File             | Role |
|------------------|------|
| `server.py`      | Routes, JSON payloads, subprocess runner, Markdown converter. |
| `templates.py`   | Inline CSS / JS / HTML (string constants only). |
| `__init__.py`    | Re-exports `launch`, `main`, `run_server`. |
| `__main__.py`    | `python -m active_inference.web` entry. |

## Adding a new route

1. Add a `do_GET` / `do_POST` branch in `server._Handler`.
2. Implement the payload helper as a method on the handler.
3. If it returns JSON, prefer the helper `_send_json`. If it serves a
   file, route through `_serve_static_file` so the safety check fires.
4. Add a test in `tests/web/test_server.py` (the fixture spins the
   server up in-process on an ephemeral port).

## Adding a new doc category to a chapter tab

Edit `server._CHAPTER_DOC_MAP`. The values are paths *relative to
`docs/`*. Missing files are silently skipped, so the map can list
forward-looking pages safely.

## Adding a new top-level doc page to the sidebar

Edit the priority list in `server._list_doc_pages`. Keep entries short
— they show up as nav buttons.

## Frontend conventions

- Vanilla JS, no transpile step. ES2020 is fine (`async`/`await`,
  optional chaining, template literals).
- Tab routing is purely client-side (no `window.location.hash` yet).
  Server endpoints take chapter / doc IDs in the URL.
- CSS is a single string in `templates.py`. Keep the palette mono:
  black background, near-black panels, gray chrome, white text. New
  CSS variables go on `:root`.

## Visual identity (the rule that matters)

Red is signal, not decoration. Use `--accent` (`#ef4444`) only for:

- the active sidebar tab,
- primary actions (`Render`, `Launch on host`, the `accent-bar` next
  to H2 / H3),
- the `cached figures` metric value,
- error toasts and the failure-output dialog,
- the favicon dot,
- the GIF badge in figure captions.

Everything else uses gray (`--text-2`, `--muted`, `--muted-2`,
`--border-hi`). Never introduce blue / purple / green / yellow.

When adding a chip / badge for a new kind of thing, prefer
*outline-only with gray text* and reserve color only when the user
*needs* to scan for it across a page.

## Metadata to surface

Anything cheap to compute server-side that helps the user pick a
script or interpret a figure goes in the JSON payload. The current
contract:

- Scripts: `size`, `size_human`, `mtime`, `mtime_human`, `docstring`
  (first line of the module docstring), `example_number`
  (e.g. `2.5`).
- Figures: `size`, `size_human`, `mtime`, `mtime_human`, `width`,
  `height` (parsed from PNG / GIF / JPEG headers), `extension`,
  `generated_by` (best-effort script-name match).

If you add a new field, also update `templates.JS` to render it and
extend `tests/web/test_server.py::TestEnrichedMetadata`.

## When NOT to touch this folder

- "Just one little extra Python package" — push back. The stdlib-only
  rule is what makes this useful in environments without network
  access (and what keeps the test surface tiny).
- Anything that needs WebSockets — open an issue first. The current
  page is request/response only by design.
