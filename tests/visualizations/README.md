# `tests/visualizations/` — tests for `src/active_inference/visualizations/`

One test file per source module. Slider widgets are instantiated under the
headless `Agg` backend and their callbacks are exercised by moving sliders
programmatically, so interactive behavior has automated coverage without
requiring a display.

| Source file | Test file |
|---|---|
| `visualizations/plotting.py` | [`test_plotting.py`](test_plotting.py) |
| `visualizations/animations.py` | [`test_animations.py`](test_animations.py) |
| `visualizations/diagnostics.py` | [`test_diagnostics.py`](test_diagnostics.py) |
| `visualizations/unified.py` (Ch.4–10) | [`test_unified.py`](test_unified.py) |
| `visualizations/interactive.py` | [`test_interactive.py`](test_interactive.py) |
| `visualizations/variational.py`, `style.py` | (exercised via `test_unified.py` + chapter smoke tests; no dedicated file) |

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
- Diagnostic plots (calibration, coverage, PPC, QQ, score/KL traces) render.
- The composable `unified` layer renders Ch.4–10 figures with expected panel
  counts, labels, legends, stat boxes, and public result traces.
- Interactive constructors update finite plotted values when sliders move,
  including the registry-driven extras topic slider.
- Chapter 8 learning/attention and message-passing figures, plus Ch.9–10
  POMDP visual helpers, are covered structurally here and by chapter smoke tests.

## Conventions

- All tests force `matplotlib.use("Agg", force=True)` *before* importing
  anything that touches `pyplot`, so no display is required.
- Tests use the `tmp_path` fixture for output paths so nothing leaks into
  the repo's `output/` directory.
- An autouse fixture closes every figure after each test to keep memory
  bounded across the run.
