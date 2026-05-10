# `tests/visualizations/` — tests for `src/active_inference/visualizations/`

One test file per source module. `interactive.py` is intentionally not
covered here — its slider widgets require a live event loop and are
exercised by chapter scripts under manual review.

| Source file | Test file |
|---|---|
| `visualizations/plotting.py` | [`test_plotting.py`](test_plotting.py) |
| `visualizations/animations.py` | [`test_animations.py`](test_animations.py) |
| `visualizations/interactive.py` | (manually verified — see Chapter 2 `interactive_explorer.py`) |

## Running

```bash
pytest tests/visualizations -v
```

## What's covered

- Static helpers (`plot_prior_likelihood_posterior`,
  `plot_generating_function`, etc.) save non-empty PNGs.
- `confidence_ellipse` returns a valid `matplotlib.patches.Ellipse` with
  correct width / height for `n_std`.
- `save_or_show` returns the resolved path on save and `None` otherwise.
- `animate_*` builders return `FuncAnimation` objects.
- `save_animation` writes a non-trivially sized GIF and closes the
  underlying figure handle.

## Conventions

- All tests force `matplotlib.use("Agg", force=True)` *before* importing
  anything that touches `pyplot`, so no display is required.
- Tests use the `tmp_path` fixture for output paths so nothing leaks into
  the repo's `output/` directory.
- An autouse fixture closes every figure after each test to keep memory
  bounded across the run.
