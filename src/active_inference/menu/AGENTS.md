# `menu/` — agent guide

User-facing text menu for `run.sh`. Three rules:

1. **Stdlib only.** No third-party imports inside this subpackage. The
   menu must run wherever Python 3.9+ runs.
2. **No domain logic.** This is presentation + subprocess plumbing. If a
   change pulls in `numpy`/`matplotlib`/`active_inference.core` symbols
   it belongs in `core/`, `estimators/`, or `visualizations/`, not here.
3. **Stay aligned with `web/`.** Both UIs consume `runner.py`. If you
   add a discovery field or filter, update the sibling
   [`active_inference.web`](../web/) so the two stay in sync.

## Files

| File         | Role |
|--------------|------|
| `runner.py`  | Discovery (`discover_chapters`, `discover_scripts`) and execution (`run_script`, `run_chapter`, `run_all_chapters`). |
| `tui.py`     | Argparse + interactive loop. `main()` is the entry point. |
| `__main__.py`| Wires `python -m active_inference.menu`. |

## Adding a new chapter

Discovery is folder-driven; no code change required. As long as the new
folder matches `chapters/chapter_<NN>/`, it will appear in the menu
automatically. Make sure:

* New scripts accept `--save` (consistent with the rest of the repo).
* `tests/chapters/test_smoke.py` picks them up.

## Adding a new non-interactive flag

1. Add it to `_parse_args` in `tui.py`.
2. Thread the value through to `run_chapter` / `run_all_chapters` in
   `runner.py`.
3. Document it in [`README.md`](README.md) and the repo-root `run.sh`
   block.

## What never lives here

* Matplotlib calls (those are in `chapters/` or `visualizations/`).
* Numerical work (those are in `core/` or `estimators/`).
* Long-running per-script logic — keep the menu's job to "list, dispatch,
  collect exit codes".
