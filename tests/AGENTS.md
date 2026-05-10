# `tests/` — agent guide

The test directory **mirrors `src/active_inference/` one-for-one**. Every
new module under `src/` must get a matching test file under `tests/`. New
chapter scripts must be registered in `tests/chapters/test_smoke.py`.

## Layout invariant

```
src/active_inference/<sub>/<module>.py   ↔   tests/<sub>/test_<module>.py
```

Examples:

| Source file | Test file |
|---|---|
| `src/active_inference/core/distributions.py` | `tests/core/test_distributions.py` (+ `test_distributions_mvn.py` for the multivariate split) |
| `src/active_inference/estimators/em.py` | `tests/estimators/test_em.py` |
| `src/active_inference/utils/grids.py` | `tests/utils/test_grids.py` |
| `src/active_inference/visualizations/plotting.py` | `tests/visualizations/test_plotting.py` |

## When you add a public symbol

1. Add a test class to the matching `tests/<sub>/test_<module>.py`.
2. The test must include at minimum:
   - one happy-path assertion against a hand-computed value;
   - one validation assertion (raises on bad input);
   - one numerical-stability or edge-case assertion (large/small values,
     batched input, empty arrays where applicable).
3. If a chapter script uses the new symbol, also confirm it is exercised
   by the chapter smoke test in `tests/chapters/test_smoke.py`.

## Conventions

- **Group with classes**, not bare functions: `class TestGaussian: ...`.
- **Name tests declaratively**: `test_pdf_integrates_to_one`, not
  `test_1`.
- **No mocks** unless you are testing failure-injection paths. Real
  `numpy` / `matplotlib` calls are fast and catch real bugs.
- **No `print()`** inside tests — let pytest capture output.
- **Always close matplotlib figures** with the `_close_figures` autouse
  fixture pattern in `tests/visualizations/`.

## Running locally

```bash
# fast inner loop — skip subprocess smoke tests
pytest tests/core tests/estimators tests/utils tests/visualizations -q

# full suite incl. chapter smoke tests
pytest

# coverage
pytest --cov=active_inference --cov-report=term-missing
```

## Don't put

- Domain logic — that belongs in `src/active_inference/`.
- Generated artifacts — they go to `output/`.
- Throwaway scratch scripts — write a new test class instead.
