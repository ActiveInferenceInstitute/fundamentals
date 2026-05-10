# `tests/utils/` — agent guide

Tests for `src/active_inference/utils/`.

## Naming

```
src/active_inference/utils/<module>.py   ↔   tests/utils/test_<module>.py
```

## Conventions

- Use the `tmp_path` pytest fixture for any test that touches the
  filesystem — never write into `output/`.
- Logger tests use unique logger names per test (`test.foo`,
  `test.bar`) so handlers from one test do not leak into another.
- For grid validation, prefer `pytest.raises(ValueError)` blocks over
  inspecting return values for sentinels.
