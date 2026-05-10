# Effective sample size

When samples are drawn with non-uniform importance weights, the
*effective sample size* (ESS) summarizes how many equally-weighted draws
they are statistically worth. ESS bottoms out at 1 when all the mass
sits on a single sample and tops out at ``N`` when the weights are
uniform.

## Definition (Kish)

For weights ``w_1, …, w_N``::

    ESS = (Σ w_i)² / Σ w_i²

The implementation works in **log space** so it stays stable for
weights that span many orders of magnitude.

## API

| Symbol | Signature | Notes |
|---|---|---|
| `effective_sample_size` | `(log_weights) -> float` | Input is 1-D log-weights; output is a scalar in `[1, N]`. |
| `logsumexp` | `(a, axis=None) -> array` | Numerical primitive used internally; safely handles all-(-inf) slices. |

Both live in `active_inference.core.diagnostics`.

## Tests that pin it down

`tests/core/test_diagnostics.py::TestESS` covers:

- Uniform log-weights give exactly ``ESS = N``.
- A weight vector with all mass on one entry collapses to ``ESS = 1``.
- Two equal weights with the rest negligible give ``ESS = 2``.
- A 2-D input raises `ValueError`.

`tests/core/test_diagnostics.py::TestLogSumExp` separately covers
overflow safety, axis handling, and the ``-∞`` boundary case.

## End-to-end snippet

```python
import numpy as np
from active_inference import effective_sample_size, logsumexp

uniform = effective_sample_size(np.zeros(100))         # → 100.0
peaked  = effective_sample_size(np.array([0.0] + [-1e6] * 99))   # → 1.0

# logsumexp avoids the obvious overflow trap.
print(logsumexp(np.array([1000.0, 1000.0, 1000.0])))   # → 1000 + log 3
```

## Pitfalls

- ``ESS`` is computed from *log*-weights, not raw weights. Take the
  log first, or pass `np.log(...)` if you have weights already.
- A small ESS does **not** automatically mean the answer is wrong. It
  means the variance of the importance-weighted estimator is large.
  Resample or extend the proposal before drawing conclusions.
- ``logsumexp`` is also exported at the top-level package because it
  shows up everywhere; prefer it over `np.log(np.sum(np.exp(...)))`.

## See also

- [`../topics/bayesian_inference.md`](../topics/bayesian_inference.md)
  — log-space normalization of grid posteriors uses the same trick.
- [`../reference/core.md`](../reference/core.md) — full API.
