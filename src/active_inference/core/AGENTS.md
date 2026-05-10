# `core/` — agent guide

Mathematical primitives. Sits at the bottom of the dependency graph: nothing
in this folder may import from `estimators/`, `visualizations/`, or
`utils/`.

## When to add a file here

Add a new `core/` module only if its content is **canonical math** that
multiple downstream layers will need:

- A new family of densities → extend `distributions.py`.
- A new flavor of generative process / model → add a class to
  `generative_process.py` or `generative_model.py` (or split into a new
  module if the class set grows past ~3).
- A new closed-form inference recipe → new module beside `lgs.py`
  (e.g., `kalman.py`, `hierarchical.py`).

If the addition is an *algorithm* over an existing model, it belongs in
`estimators/` instead.

## Conventions

- Variances and covariances, never standard deviations.
- Vectorized first; never write a Python loop where `numpy` broadcasting
  works.
- Validate inputs at construction time and raise `ValueError` with a clear
  message — don't push validation to `infer()` / `sample()`.
- Use Cholesky-based solves over explicit `np.linalg.inv` for any
  multivariate work.

## Minimum review checklist for a new method

1. Type-hint every public argument and return.
2. Numpy-style docstring with a Parameters / Returns block.
3. Unit test in `tests/core/` with at least:
   - one happy-path test against a hand-computed value,
   - one validation test that raises on bad input,
   - one numerical-stability test (large / small variances, batched input).
4. Mention it in `docs/core.md` and the public-surface `__init__.py`.

## Dependency graph

```
core/  →  numpy, scipy.linalg.solve_triangular
```

That is the entire allowed import set. If you find yourself reaching for
`matplotlib` here, push the plotting code up into `visualizations/`.
