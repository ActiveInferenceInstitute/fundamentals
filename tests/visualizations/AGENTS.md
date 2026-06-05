# `tests/visualizations/` — agent guide

Tests for `src/active_inference/visualizations/`. These tests run under
`MPLBACKEND=Agg` so they work in CI / headless containers.

## Naming

```
src/active_inference/visualizations/<module>.py   ↔   tests/visualizations/test_<module>.py
```

`interactive.py` is tested here under `Agg`: constructors return figures,
slider callbacks are triggered programmatically, and updated plotted values
must remain finite. Live display behavior can still be checked through the
chapter and extras interactive wrappers when changing GUI ergonomics.

## When you add a new figure helper

1. Add a test class to the matching `test_<module>.py`.
2. The test must verify:
   - the helper returns the expected type (`Figure` / `FuncAnimation`);
   - when `save_path` is supplied, the file exists with a non-zero size;
   - validation raises on bad input shapes.

## Conventions

- Import order matters for the `Agg` backend:
  ```python
  import matplotlib
  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt  # noqa: E402
  ```
- Use the `tmp_path` fixture for save paths.
- Use the `_close_figures` autouse fixture pattern to keep memory bounded.
- Animations issue a `UserWarning` when garbage-collected unsaved — this
  is harmless in tests and intentionally not silenced.
