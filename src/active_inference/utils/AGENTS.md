# `utils/` — agent guide

Small infrastructure helpers shared across the package. This folder sits at
the **bottom** of the dependency graph: nothing in `utils/` may import from
`core/`, `estimators/`, or `visualizations/`.

## When to add a file here

A new module in `utils/` is justified only when the helper is:

- **Domain-agnostic** (no statistics, no models — just paths, grids,
  loggers, RNG plumbing).
- **Used by ≥ 2 other subpackages** or by chapter orchestrators directly.

If the helper is statistical, it goes in `core/`. If it produces a figure,
it goes in `visualizations/`.

## Files at a glance

| File | Purpose |
|---|---|
| `grids.py` | `make_grid`, `make_2d_grid` — discretized state domains. |
| `logging.py` | `get_logger` — idempotent stdlib logger factory. |
| `io.py` | `default_figure_dir`, `default_data_dir`, `ensure_dir`. |

## Conventions

- Public path returns are `pathlib.Path`, not `str`.
- Logger factories are idempotent — calling with the same name returns the
  same logger and never stacks duplicate handlers.
- Grid constructors validate their bounds and raise `ValueError` for
  non-finite or inverted ranges.

## Minimum review checklist

1. Type hints on every public function.
2. Numpy-style docstring with at least Parameters and Returns.
3. Unit test in `tests/utils/`.
4. Re-export via `utils/__init__.py`.

## Dependency graph

```
utils/  →  numpy (only in grids.py), pathlib, logging, sys
```
