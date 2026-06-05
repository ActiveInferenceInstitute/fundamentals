# `visualizations/` — agent guide

Plotting and animation helpers. May import from `core/` and `utils/`. Must
**not** import from `estimators/` (estimators produce data; this folder
visualizes it). Chapter scripts compose helpers from this folder; keep
chart-style decisions here, not in chapter orchestrators.

## When to add a file here

| You want to add… | Add it as… |
|---|---|
| A reusable static figure pattern | A function in `plotting.py`. |
| A slider widget that ships in a chapter window | A function in `interactive.py`. |
| A `FuncAnimation` builder | A function in `animations.py`. |
| Brand-new visualization category | A new module beside the existing three. |

## Conventions

- Every helper accepts an optional `ax` argument so callers can compose
  multi-panel figures.
- Every helper accepts an optional `save_path`. Use `save_or_show(fig,
  save_path, show=show)` to dispatch — never duplicate that logic.
- Animations always set `_fig` on the returned object so `save_animation`
  can close the figure handle cleanly afterwards.
- Default to `style.COLORS`, the repo-wide Okabe-Ito colourblind-safe palette.
  Use perceptually ordered colormaps only for scalar fields and heatmaps where
  continuous value order is part of the concept.
- Default `dpi=150`, `constrained_layout=True`.
- No `seaborn`, `plotly`, or other plotting libraries — keep the
  dependency surface minimal.

## Minimum review checklist

1. Headless-safe: `MPLBACKEND=Agg` must produce identical output.
2. Test in `tests/visualizations/` that the helper writes a non-zero file
   and returns the right type (`Figure` or `FuncAnimation`).
3. Update `docs/visualizations.md` if it adds a public symbol.

## Dependency graph

```
visualizations/  →  matplotlib, numpy, active_inference.core, active_inference.utils
```
