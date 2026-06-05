# src/active_inference/visualizations/ — Plotting and Interactive Tools

Static matplotlib figures and interactive widget-based simulations for the
chapter orchestrators. Every function is self-contained: pass data, get a
figure, optionally save it.

## Files

| File | What it defines |
|---|---|
| [`style.py`](style.py) | The shared visual vocabulary: `COLORS` (Okabe-Ito colourblind-safe palette), `DEFAULT_RC` (bold, slide-sized typography applied at import), `annotate_stat_box`, `annotate_point`, `figure_style`. |
| [`plotting.py`](plotting.py) | Static figure helpers for Chapters 1–3 (prior/likelihood/posterior, generating function, gradient descent, 2-D Gaussians, …) + `save_or_show`. |
| [`variational.py`](variational.py) | Chapter 4 figures — VFE surface/contour, density evolution, the five-form decomposition, surprisal relationship. |
| [`unified.py`](unified.py) | The composable Chapter 4–10 layer: recognition dynamics, generalized filtering, Ch.6 correlated/vector generalized-coordinate plots, Ch.7 multivariate active-inference plots, Chapter 8 learning/attention and message passing, dynamic discrete beliefs, policy EFE decomposition, parameter learning, bandit, factorial, and hierarchical plotters, plus the `panel_grid`/`finalize`/`layer_colors` primitives. |
| [`diagnostics.py`](diagnostics.py) | Calibration / coverage / posterior-predictive diagnostic plots. |
| [`animations.py`](animations.py) | `FuncAnimation` builders (pillow GIF), including composable recognition, hierarchical PC, Ch.7 multivariate active inference, Chapter 8 learning/attention, dynamic discrete belief, policy EFE trade-off, parameter-learning, precision, and bandit animations plus `save_animation`. |
| [`interactive.py`](interactive.py) | Matplotlib slider widgets (no ipywidgets). |
| `__init__.py` | Re-exports all public names. |

## Public API

The authoritative, always-current API listing lives in
[`docs/reference/visualizations.md`](../../../docs/reference/visualizations.md)
(every function, its signature, and what it produces). It is kept in sync with
`__all__`; consult it rather than duplicating signatures here. A few cross-cutting
conventions:

- **`ax` is supported where it composes.** Several helpers (e.g.
  `plot_2d_gaussian`) accept an optional `ax: Optional[plt.Axes]` and draw into it;
  the `unified` plotters return a multi-panel `Figure`. Functions that own their
  whole figure simply create one.
- **`save_or_show(fig, save_path, *, show, dpi)` is the single I/O gateway** — pass
  a `save_path` to write a PNG, omit it to display.
- **Style is centralized.** Colours come from `style.COLORS` (colourblind-safe);
  fonts/line-widths from `DEFAULT_RC`. Nothing hard-codes hex or `figsize` in the
  composable layer, so the whole package re-skins from one place.

### Interactive Widgets (`interactive.py`)

| Function | Sliders | Description |
|---|---|---|
| `interactive_inference(x_low, x_high, n_grid, beta0, beta1, sigma2_y_init, s2_x_init, m_x_init, y_init)` | `y`, `m_x`, `s2_x`, `sigma2_y` | Full 4-slider exploration of prior/likelihood/posterior |
| `interactive_precision(x_low, x_high, beta0, beta1, y_obs, m_x)` | `log10(s2_x / sigma2_y)` | Single-slider precision ratio sweep |
| `interactive_topic_demo(slug)` | `simulation blend` | Registry-driven extras slider over the topic's static and simulation demos |

These use `matplotlib.widgets.Slider` (not ipywidgets), so they work in any
environment with a display and matplotlib installed.

## Design Decisions

- **No global state.** Each function creates and returns a new figure.
- **`save_or_show` is the single I/O gateway.** Every plot function delegates
  to it, so switching between save and display is always one argument.
- **Composable by result object.** The `unified` plotters and animators accept
  reusable traces or result objects, so chapter scripts stay thin and the shared
  visual grammar is tested once.
- Return the `Figure` object so callers can compose or further customize.
- **Accessibility first.** Colourblind-safe palette, bold large fonts, a legend on
  every panel, and statistics/analytical annotations baked into the shared layer.

## Dependencies

`numpy`, `matplotlib`. No ipywidgets dependency (unlike many Jupyter-based tools).

## Testing

Visualization code is exercised by dedicated unit tests in
`tests/visualizations/` (plotting, animations, diagnostic figures, and
interactive slider callbacks) and indirectly by the chapter smoke tests in
`tests/chapters/test_smoke.py`, which run every orchestrator and verify
it exits 0. Rendered-output checks catch blank/corrupt artifacts; visual
review remains useful for final layout judgement when changing pedagogy.
