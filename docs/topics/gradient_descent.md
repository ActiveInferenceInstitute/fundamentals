# Gradient descent

Gradient descent is the workhorse iterative optimizer used whenever a
closed-form maximum-likelihood / maximum-a-posteriori solution is
unavailable, undesirable, or just being demonstrated alongside the
analytic answer. The package ships a tight, dataclass-returning helper
that supports analytic gradients, a finite-difference fallback, and full
iterate / loss history.

## In this codebase

- **Generic 1-D minimizer:**
  `estimators.gradient_descent.gradient_descent(loss_fn, x0, learning_rate, max_iter, grad_fn=None, tol=1e-8, record_history=True)`.
- **Vectorized linear-regression descent:**
  `estimators.linear_regression.gd_linear_regression(X, y, learning_rate, ..., l2=0.0)` with the same dataclass shape.
- **Analytic gradients used by chapter scripts:**
  `mle_grad_x`, `map_grad_x`, `squared_loss_grad`.
- **Result types:** `GradientDescentResult` (1-D),
  `GDRegressionResult` (multivariate).

## End-to-end snippet

```python
import numpy as np
from active_inference import (
    gradient_descent, gd_linear_regression, mle_linear_regression,
)

# 1-D minimization with an analytic gradient.
result = gradient_descent(
    loss_fn=lambda x: (x - 3.0) ** 2,
    grad_fn=lambda x: 2.0 * (x - 3.0),
    x0=0.0, learning_rate=0.1, max_iter=200,
)
print(result.x, result.converged, result.n_iterations)

# Vectorized regression with L2 regularization.
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + 0.5 + rng.normal(scale=0.1, size=200)

iterative = gd_linear_regression(X, y, learning_rate=1e-3, max_iter=2000)
analytic = mle_linear_regression(X, y)
print("‖θ_GD − θ_OLS‖:", np.linalg.norm(iterative.theta - analytic))
```

## The stability bound

For a least-squares loss the gradient is linear in `θ` with Hessian
``H = XᵀX + λ I``. The descent recursion contracts iff::

    learning_rate · λ_max(H) < 2

In practice we usually pick ``learning_rate ∈ [1e-4, 1e-2]`` for the
chapter scripts and check convergence against the closed-form solution.

## Pitfalls

- The 1-D minimizer's `tol` is on `|x_{k+1} − x_k|`, not on the loss.
  A perfectly flat loss landscape (e.g., a vanishing gradient) will
  also trigger convergence.
- `record_history=True` is the default because chapter scripts often
  animate trajectories; turn it off in tight inner loops.
- Without `grad_fn`, the helper falls back to centered finite
  differences with `h=1e-5`. That is fine for verification but slow for
  larger problems.

## See also

- [`learning_and_inference.md`](learning_and_inference.md) — when to
  reach for GD vs the closed-form alternative.
- [`../chapters/chapter_02.md`](../chapters/chapter_02.md) — Example
  2.10 (the canonical GD demo).
- [`../chapters/chapter_03.md`](../chapters/chapter_03.md) — Example 3.2
  (multivariate GD on the regression loss surface).
- [`../reference/estimators.md`](../reference/estimators.md) — full API.
