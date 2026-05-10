# `tests/chapters/` — agent guide

Subprocess-based smoke tests for the chapter orchestrators in
`chapters/chapter_0{1,2,3}/`.

## Discovery model

Scripts are discovered by **glob pattern**, not by a hard-coded list:

```python
CHAPTER_1_SCRIPTS    = sorted(CHAPTER_DIRS[1].glob("0*.py"))
CHAPTER_2_EXAMPLES   = sorted(CHAPTER_DIRS[2].glob("example_*.py"))
CHAPTER_2_ANIMATIONS = sorted(CHAPTER_DIRS[2].glob("animation_*.py"))
CHAPTER_3_EXAMPLES   = sorted(CHAPTER_DIRS[3].glob("example_*.py"))
CHAPTER_3_ANIMATIONS = sorted(CHAPTER_DIRS[3].glob("animation_*.py"))
```

A new chapter script is therefore picked up automatically. The price is
that the glob patterns are an implicit contract — name files
`example_*.py` / `animation_*.py` / `visualize_*.py` /
`interactive_*.py` consistently.

## When to edit `test_smoke.py`

- A new chapter is added (extend `CHAPTER_DIRS`).
- A new file-name pattern is introduced (extend the glob calls).
- A script needs a longer timeout (animations already get 240 s).

## Conventions

- All scripts run with `MPLBACKEND=Agg` so no display is required.
- `interactive_*.py` scripts are filtered out of the smoke run via
  `_is_interactive`.
- Each script runs in its own subprocess — failures in one do not affect
  the rest of the run.

## Don't put

- Visual-content assertions — those belong in
  `tests/visualizations/`.
- Long-running performance tests — keep this directory smoke-only so
  CI stays fast.
