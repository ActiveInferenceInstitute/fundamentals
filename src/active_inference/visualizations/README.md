# src/active_inference/visualizations/ — Plotting and Interactive Tools

Static matplotlib figures and interactive widget-based simulations for the
chapter orchestrators. Every function is self-contained: pass data, get a
figure, optionally save it.

## Files

| File | What it defines |
|---|---|
| [`plotting.py`](plotting.py) | Static figure helpers (6 plot functions + `save_or_show`) |
| [`interactive.py`](interactive.py) | Matplotlib slider widgets (2 interactive functions) |
| `__init__.py` | Re-exports all public names |

## Public API

### Static Plots (`plotting.py`)

| Function | Signature | Produces |
|---|---|---|
| `plot_prior_likelihood_posterior(result, *, title, truth, save_path, show)` | `InferenceResult` → 3-panel axes | Prior / likelihood / posterior densities |
| `plot_generating_function(x_grid, f_x, *, samples_x, samples_y, title, save_path, show)` | Grid + function values → axes | `y = g(x)` with optional sample scatter |
| `plot_likelihood_ridge(x_grid, likelihoods, *, labels, title, save_path, show)` | Grid + list of densities → axes | Stacked ridge plot of per-sample likelihoods |
| `plot_joint_heatmap(x_grid, y_grid, joint, *, title, save_path, show)` | 2-D grid + joint density → axes | Heatmap of `p(x, y)` |
| `plot_gradient_descent(history, losses, *, truth, title, save_path, show)` | Iterate + loss arrays → axes | Loss curve and iterate trajectory side-by-side |
| `plot_precision_comparison(results, *, title, save_path, show)` | List of `(label, InferenceResult)` → axes | Overlay of multiple posteriors for precision studies |
| `save_or_show(fig, save_path, *, show, dpi)` | Figure → file or screen | Utility: save to disk or display |

All plotting functions accept:
- An optional `ax` parameter is **not** used (each creates its own figure/axes)
  because the chapter orchestrators manage multi-panel layouts themselves.
- An optional `save_path` (path or string). If provided, the figure is saved
  and not shown (unless `show=True`).
- An optional `show=False` to display interactively.

### Interactive Widgets (`interactive.py`)

| Function | Sliders | Description |
|---|---|---|
| `interactive_inference(x_low, x_high, n_grid, beta0, beta1, sigma2_y_init, s2_x_init, m_x_init, y_init)` | `y`, `m_x`, `s2_x`, `sigma2_y` | Full 4-slider exploration of prior/likelihood/posterior |
| `interactive_precision(x_low, x_high, beta0, beta1, y_obs, m_x)` | `log10(s2_x / sigma2_y)` | Single-slider precision ratio sweep |

These use `matplotlib.widgets.Slider` (not ipywidgets), so they work in any
environment with a display and matplotlib installed.

## Design Decisions

- **No global state.** Each function creates and returns a new figure.
- **`save_or_show` is the single I/O gateway.** Every plot function delegates
  to it, so switching between save and display is always one argument.
- Return the `Figure` object so callers can compose or further customize.

## Dependencies

`numpy`, `matplotlib`. No ipywidgets dependency (unlike many Jupyter-based tools).

## Testing

Visualization code is exercised by dedicated unit tests in
`tests/visualizations/` (plotting, animations, diagnostic figures) and
indirectly by the chapter smoke tests in
`tests/chapters/test_smoke.py`, which run every orchestrator and verify
it exits 0. Visual inspection remains the final arbiter for layout and
colour decisions.