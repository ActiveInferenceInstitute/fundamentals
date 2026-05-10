# `tests/estimators/` — agent guide

Tests for `src/active_inference/estimators/`. Strict naming:

```
src/active_inference/estimators/<module>.py   ↔   tests/estimators/test_<module>.py
```

## When adding a new estimator

1. Add a test class to the matching `test_<module>.py`.
2. The test must cover:
   - **Happy path:** estimator recovers the truth at high N (use ≥ 200
     samples and `pytest.approx(rel=0.05)`).
   - **Closed-form / iterative agreement:** if both forms exist, assert
     `gd.x ≈ analytic.x` to `atol ≤ 5e-3`.
   - **Determinism:** runs twice with the same seed produce identical
     results.
   - **Validation:** raises `ValueError` for negative learning rate,
     non-finite inputs, mismatched shapes.
3. For iterative algorithms, also assert **monotone improvement** of the
   loss (or marginal log-likelihood) over iterations.

## Tips

- Use `np.random.default_rng(seed)` explicitly to keep tests
  reproducible across platforms.
- Factor analysis is identifiable only up to a rotation — compare
  *subspaces* via the singular values of `Q_true^T @ Q_est` rather than
  the raw loadings.
- Gradient descent stability requires `lr · λ_max(XᵀX) < 2`; the test
  helpers default to safe rates (`1e-3` / `1e-4`).
