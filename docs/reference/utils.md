# `active_inference.utils` — module reference

Small helpers shared across the package. Three modules, no domain logic.

## `utils.grids`

Constructs the discretized state domains used by `GridBayesianInference`
and the visualization helpers.

| Symbol | Role |
|---|---|
| `make_grid(low, high, n_points=500)` | Evenly-spaced 1-D `np.linspace` grid; validates bounds and finiteness. |
| `make_2d_grid(x_low, x_high, y_low, y_high, n_x=200, n_y=200)` | Two coordinate vectors for rectangular joint-density grids. |

```python
from active_inference import make_grid

x_grid = make_grid(0.0, 5.0, 500)
```

## `utils.logging`

Lightweight stdlib-logger factory so chapter scripts share a consistent
format and don't double-attach handlers when the same logger name is
requested multiple times.

| Symbol | Role |
|---|---|
| `get_logger(name="active_inference", level=logging.INFO)` | Returns an idempotent, formatted, non-propagating logger. |

```python
from active_inference import get_logger

LOG = get_logger("ch3.ex5")
LOG.info("Posterior mode = %.4f", 2.176)
```

## `utils.io`

Path conventions for figures and serialized data produced by chapter
scripts.

| Symbol | Role |
|---|---|
| `default_figure_dir()` | `<repo>/output/figures/` |
| `default_data_dir()` | `<repo>/output/data/` |
| `ensure_dir(path)` | `mkdir -p` semantics; returns the resolved `Path`. |

```python
from active_inference.utils.io import default_figure_dir, ensure_dir

out = ensure_dir(default_figure_dir() / "chapter_03")
fig.savefig(out / "posterior.png")
```

## Conventions

- All paths are returned as `pathlib.Path` objects.
- Logger names should mirror the chapter / module hierarchy
  (`"ch3.ex5"`, `"core.inference"`).
- No module here imports from `active_inference.core` / `estimators` /
  `visualizations` — `utils` sits at the bottom of the dependency graph.
