# src/active_inference/utils/ — Utilities

Small helpers used across the package: grid construction, logging, and
output-path conventions.

## Files

| File | What it defines |
|---|---|
| [`grids.py`](grids.py) | `make_grid`, `make_2d_grid` |
| [`logging.py`](logging.py) | `get_logger` |
| [`io.py`](io.py) | `default_figure_dir`, `default_data_dir`, `ensure_dir` |
| `__init__.py` | Re-exports all public names |

## Public API

```python
from active_inference.utils.grids import make_grid, make_2d_grid
from active_inference.utils.logging import get_logger
from active_inference.utils.io import default_figure_dir, default_data_dir, ensure_dir
```

Also available from the top-level package:
```python
from active_inference import make_grid, get_logger
```

### `grids.py`

- `make_grid(low, high, n_points=500)` → evenly-spaced 1-D `np.ndarray`.
  Validates finite bounds, `low < high`, `n_points >= 2`.
- `make_2d_grid(x_low, x_high, y_low, y_high, n_x=200, n_y=200)` →
  `(x_array, y_array)` tuple for 2-D joint density grids.

### `logging.py`

- `get_logger(name="active_inference", level=logging.INFO)` → configured
  `logging.Logger` with a `StreamHandler(stdout)` and a consistent format
  (`[HH:MM:SS] LEVEL  name: message`). Idempotent — repeated calls for the
  same name return the same logger without adding duplicate handlers.

### `io.py`

- `default_figure_dir()` → `Path("output/figures")` (relative to repo root).
- `default_data_dir()` → `Path("output/data")` (relative to repo root).
- `ensure_dir(path)` → create `path` (and parents) if missing; return it.

## Design Decisions

- **`__file__`-relative paths:** `io.py` computes the repo root as
  `Path(__file__).resolve().parents[3]`, so `default_figure_dir()` always
  points to the right place regardless of the working directory.
- **No external dependencies** beyond numpy (used only in `grids.py`).
- **Minimal, focused functions** — each does one thing well.

## Testing

Utility functions are exercised indirectly through every chapter script and
through the unit tests in `tests/test_generative.py` (which uses `make_grid`)
and `tests/test_estimators.py` (which uses `gradient_descent`). No dedicated
test file for utils — their logic is trivial enough that integration testing
provides sufficient coverage.