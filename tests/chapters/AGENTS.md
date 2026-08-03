# `tests/chapters/` — agent guide

Subprocess-based smoke tests for the chapter orchestrators in
`chapters/chapter_0{1..10}/`.

## Discovery model

Scripts are discovered through the shared **menu-runner inventory**
(`active_inference.menu.runner.discover_chapters` /
`discover_scripts`), the same discovery layer used by `run.sh`, the web
UI, and `scripts/run_all_figures.py` — not by a hard-coded list:

```python
# tests/chapters/test_smoke.py
from active_inference.menu.runner import discover_chapters, discover_scripts

def _all_chapter_scripts() -> list[Path]:
    scripts: list[Path] = []
    for chapter in discover_chapters():            # chapters 1–14
        scripts.extend(entry.path for entry in discover_scripts(chapter.number))
    return scripts

CHAPTER_ALL_SCRIPTS = _all_chapter_scripts()
```

A new chapter folder or script is therefore picked up automatically. The
price is that the `example_*.py` / `animation_*.py` / `visualize_*.py` /
`interactive_*.py` filename conventions are an implicit contract — name
files consistently so classification (and the web/menu UI) stays correct.

## When to edit `test_smoke.py`

- A new file-name pattern is introduced (extend the discovery or
  classification calls).
- A script needs a different timeout (animations already get 240 s).
- New per-chapter special cases are needed (see `_run` and the
  per-script timeout table).

## Conventions

- All scripts run with `MPLBACKEND=Agg` and `PYTHONWARNINGS=error` so no display
  is required and warning regressions fail the smoke run.
- `interactive_*.py` scripts are filtered out of the smoke run via
  `_is_interactive`.
- Each script runs in its own subprocess — failures in one do not affect
  the rest of the run.

## Don't put

- Visual-content assertions — those belong in
  `tests/visualizations/`.
- Long-running performance tests — keep this directory smoke-only so
  CI stays fast.
