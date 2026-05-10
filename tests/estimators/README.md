# `tests/estimators/` — tests for `src/active_inference/estimators/`

One test file per source module.

| Source file | Test file |
|---|---|
| `estimators/mle.py` | [`test_mle.py`](test_mle.py) |
| `estimators/map.py` | [`test_map.py`](test_map.py) |
| `estimators/gradient_descent.py` | [`test_gradient_descent.py`](test_gradient_descent.py) |
| `estimators/linear_regression.py` | [`test_linear_regression.py`](test_linear_regression.py) |
| `estimators/em.py` | [`test_em.py`](test_em.py) |

## Running

```bash
pytest tests/estimators -v
```

## What's covered

- Closed-form MLE / MAP solutions against hand-computed values.
- Gradient descent converges to the analytic answer (with both analytic
  and finite-difference gradients).
- Bayesian linear regression posterior tightens monotonically with N.
- L2 regularization shrinks the gradient-descent solution.
- Factor-analysis EM log-likelihood is monotone non-decreasing.
- Factor-analysis EM recovers the true subspace (compared via singular
  values of the QR decomposition since FA is identifiable only up to
  rotation).
- Validation: negative variances, zero learning rates, mismatched shapes.
