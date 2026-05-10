# `estimators/` — agent guide

Algorithms that consume a `core/` model and produce a point estimate, a
parameter posterior, or learn parameters. Allowed imports: `numpy`,
`active_inference.core`. Disallowed: `matplotlib`, `visualizations/`.

## When to add a file here

| You want to add… | Add it as… |
|---|---|
| A new closed-form estimator for an existing model | A function in the appropriate file (e.g., `mle.py` or `linear_regression.py`). |
| A new optimizer | A new module (`adam.py`, `lbfgs.py`) with a result dataclass. |
| A new conjugate Bayesian update | A class in `linear_regression.py` if it's about θ; otherwise a new module. |
| A new EM-style algorithm | A new module beside `em.py`. |

## Conventions

- Closed-form estimators come **first**; gradient-based versions are added
  alongside for verification, not as a replacement.
- Every estimator returns a dataclass when there is more than one piece of
  information to convey (final iterate + history + convergence flag).
- All loss functions are in **negative log-likelihood** form so a single
  generic minimizer can be used.
- Validate `learning_rate > 0`, `max_iter >= 1`, finite inputs at the top
  of every public function; raise `ValueError` with the offending value in
  the message.

## Minimum review checklist

1. Numpy-style docstring with Parameters / Returns and an example.
2. Unit test in `tests/estimators/` that:
   - matches the closed-form solution to the gradient-descent solution;
   - exercises a non-default branch (e.g., L2 regularization, no intercept);
   - validates input checking.
3. Re-export via `estimators/__init__.py` and `active_inference/__init__.py`.
4. Update `docs/estimators.md` and the chapter `README.md` if a chapter
   orchestrator uses it.

## Dependency graph

```
estimators/  →  numpy, active_inference.core
```
