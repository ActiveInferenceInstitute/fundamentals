# `tests/demo/` — smoke tests for application demo orchestrators

Subprocess smoke tests that run every non-interactive demo script under
`demo/<slug>/` with `--save` and assert exit code 0 plus fresh NPZ+JSON
raw-data sidecars under `output/data/demo/<slug>/`.

| File | Coverage |
|---|---|
| [`test_demo_smoke.py`](test_demo_smoke.py) | Every script discovered by `active_inference.menu.runner.discover_demo_scripts` for the slugs declared by `active_inference.demo_topics.demo_topic_slugs()`. |

## Running

```bash
pytest tests/demo -v

# demo smoke + workflow tests together
pytest tests/demo tests/test_demo_workflows -v
```

## What's checked

For each demo script, the subprocess run:

1. Resolves `active_inference` imports via `PYTHONPATH=src`.
2. Accepts `--save` through `argparse`.
3. Exits with code 0 and empty STDERR under `PYTHONWARNINGS=error`.
4. Writes a fresh NPZ+JSON sidecar pair under `output/data/demo/<slug>/`.

Interactive-only demo wrappers (if any) are exercised through the shared
visualization constructor tests rather than as subprocess `--save` cases.
