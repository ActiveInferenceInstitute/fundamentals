# `tests/core/` — agent guide

Tests for `src/active_inference/core/`. The naming rule is strict:

```
src/active_inference/core/<module>.py   ↔   tests/core/test_<module>.py
```

The one allowed exception is `core/distributions.py`, which is split into
`test_distributions.py` (univariate) and `test_distributions_mvn.py`
(multivariate) because the file gets too long otherwise.

## When adding a new core method

1. Add a test class to the matching file:
   - new univariate density → `test_distributions.py`,
   - new multivariate helper → `test_distributions_mvn.py`,
   - new generative process / model → matching `test_*` file,
   - new inference engine → its own file (e.g., `test_kalman.py`).
2. Cover at least:
   - happy-path against a hand-computed value;
   - normalization / determinism property;
   - input-validation `ValueError`;
   - numerical-stability case (very small / large variance).

## Tips

- Use `pytest.approx(rel=1e-12)` for log-PDF round-trips.
- Trapezoid-integrate over a dense grid (4001 points) for unit-mass
  checks; coarser grids alias.
- Pass `np.random.default_rng(seed)` explicitly — never the global RNG.
