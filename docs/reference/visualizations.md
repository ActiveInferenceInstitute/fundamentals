# `active_inference.visualizations` — module reference

Reusable figure helpers for the chapter orchestrators. Three modules: static
plots, interactive sliders, and matplotlib animations. Every function accepts
an optional `ax` (for composition) and an optional `save_path` (for headless
rendering).

## `visualizations.plotting`

Static matplotlib helpers. Each returns the `Figure` so callers can keep
composing or save themselves.

| Symbol | Role |
|---|---|
| `save_or_show(fig, save_path=None, *, show=False, dpi=150)` | Either save to disk or call `plt.show()`; returns the resolved path. |
| `plot_prior_likelihood_posterior(result, *, truth=None, save_path=None, ...)` | Three-panel prior / likelihood / posterior figure. |
| `plot_generating_function(x, f_x, samples_x=None, samples_y=None, ...)` | `y = g(x)` curve with optional sample scatter. |
| `plot_likelihood_ridge(x_grid, likelihoods, labels=None, ...)` | Vertically stacked ridge plot of per-sample likelihoods. |
| `plot_joint_heatmap(x_grid, y_grid, joint, ...)` | 2-D heatmap of `p(x, y)`. |
| `plot_gradient_descent(history, losses, *, truth=None, ...)` | Side-by-side loss-vs-iter and iterate-vs-iter view. |
| `plot_precision_comparison(results, ...)` | Overlay multiple posteriors for a precision sweep. |
| `confidence_ellipse(mean, cov, *, n_std=2.0, **kwargs)` | Eigen-aligned `Ellipse` patch for a 2-D Gaussian. |
| `plot_2d_gaussian(mean, cov, *, samples=None, truth=None, ...)` | 1-σ / 2-σ ellipses + optional sample scatter. |

## `visualizations.interactive`

Slider-driven exploration with no `ipywidgets` dependency — purely
`matplotlib.widgets.Slider`.

| Symbol | Role |
|---|---|
| `interactive_inference(...)` | Live sliders for observation, prior mean, prior variance, likelihood variance. |
| `interactive_precision(...)` | Single-slider sweep of the prior-vs-data precision ratio. |

```python
from active_inference.visualizations import interactive_inference

interactive_inference()
import matplotlib.pyplot as plt; plt.show()
```

## `visualizations.animations`

Matplotlib `FuncAnimation` builders. The bundled pillow writer means GIF
output works with no FFmpeg / ImageMagick install.

| Symbol | Role |
|---|---|
| `animate_sequential_posterior(x_grid, posteriors, *, truth=None, prior=None, ...)` | One frame per assimilated observation. |
| `animate_gradient_descent(loss_grid, x_grid, history, losses, ...)` | Iterate rolling down the loss curve + sync'd loss trace. |
| `animate_2d_posterior(means, covs, *, truth=None, prior_mean=None, prior_cov=None, ...)` | 1-/2-σ ellipses tightening over frames. |
| `animate_em_convergence(log_likelihoods, Theta_history, ...)` | LL curve + heat-map of loadings per iteration. |
| `save_animation(anim, path, *, fps=12, dpi=110)` | Save to GIF via the bundled pillow writer. |

## Conventions

- Every helper accepts an optional `save_path`; figures are dpi 150 by
  default and laid out with `constrained_layout=True`.
- Animations always set `_fig` on the returned object so `save_animation`
  can close the figure cleanly afterwards.
- Color choices follow matplotlib's `tab10` / `viridis` / `magma` palettes
  to stay readable in print and on dark backgrounds.
