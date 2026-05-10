# `tests/core/` — tests for `src/active_inference/core/`

One test file per source module, plus a dedicated split for the
multivariate distributions because they are large enough to deserve their
own file.

| Source file | Test file |
|---|---|
| `core/distributions.py` (univariate helpers) | [`test_distributions.py`](test_distributions.py) |
| `core/distributions.py` (multivariate helpers) | [`test_distributions_mvn.py`](test_distributions_mvn.py) |
| `core/generative_process.py` | [`test_generative_process.py`](test_generative_process.py) |
| `core/generative_model.py` | [`test_generative_model.py`](test_generative_model.py) |
| `core/inference.py` | [`test_inference.py`](test_inference.py) |
| `core/lgs.py` | [`test_lgs.py`](test_lgs.py) |

## Running

```bash
# all core tests
pytest tests/core -v

# a single module
pytest tests/core/test_lgs.py -v
```

## What's covered

- Density helpers normalize correctly via trapezoid integration.
- Log-PDF matches the exponential of the PDF (numerical stability).
- Generative processes recover known means and covariances from samples.
- Generative models validate input shapes and raise on bad inputs.
- `GridBayesianInference` posterior matches sequential Bayesian updating.
- `LinearGaussianSystem.posterior_batch` recovers the truth as N grows.
