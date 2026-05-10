# `tests/utils/` — tests for `src/active_inference/utils/`

One test file per source module.

| Source file | Test file |
|---|---|
| `utils/grids.py` | [`test_grids.py`](test_grids.py) |
| `utils/io.py` | [`test_io.py`](test_io.py) |
| `utils/logging.py` | [`test_logging.py`](test_logging.py) |

## Running

```bash
pytest tests/utils -v
```

## What's covered

- Grid endpoints, uniform spacing, validation of inverted / non-finite
  bounds.
- 2-D grid shapes and endpoint preservation.
- Default figure / data directory paths point under the repo's `output/`.
- `ensure_dir` is idempotent and accepts both `Path` and `str`.
- Logger factory is idempotent (no duplicated handlers), formats include
  level + name + message, and `propagate=False` is enforced.
