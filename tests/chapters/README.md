# `tests/chapters/` — smoke tests for chapter orchestrators

Subprocess-based smoke tests that run every chapter script with `--save`
and assert exit code 0.

| File | Coverage |
|---|---|
| [`test_smoke.py`](test_smoke.py) | Chapters 1, 2, 3 — examples + auxiliary + animations. |

## Running

```bash
# All chapter smoke tests (~ 60s on a laptop including animations)
pytest tests/chapters -v

# Skip the animations (faster inner loop)
pytest tests/chapters -v -k 'not animation'
```

## What's checked

For each script under `chapters/chapter_0{1,2,3}/`:

1. The Python interpreter resolves `active_inference` imports.
2. The script's `argparse` accepts `--save`.
3. The script exits with code 0.
4. STDERR is empty (no traceback).
5. For animations, the timeout is bumped to 240 s to allow the pillow
   GIF encoder to finish.

The smoke tests do **not** check the visual content of generated figures
— that is left to manual review and the unit tests in
`tests/visualizations/`.

## Adding a new chapter script

1. Drop the script in the appropriate `chapters/chapter_<N>/` folder.
2. Run it locally once with `--save` to make sure it produces a figure
   without crashing.
3. The smoke test discovers it automatically by glob (`example_*.py`,
   `animation_*.py`, etc.) — no test code change required.
