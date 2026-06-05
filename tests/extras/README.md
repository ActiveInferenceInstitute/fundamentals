# `tests/extras/` - smoke tests for extras orchestrators

Subprocess smoke tests that run every non-interactive
`extras/<topic>/{visualize,simulate,animation}_<topic>.py` script with `--save`
under `MPLBACKEND=Agg`. Interactive extras launchers are validated through
script-discovery/provenance tests and the shared visualization constructor
tests, not as subprocess `--save` cases.

| File | Coverage |
|---|---|
| [`test_smoke.py`](test_smoke.py) | Registry-driven smoke coverage for every slug returned by `active_inference.extra_topics.extra_topic_slugs()`. |

## Running

```bash
pytest tests/extras -v
```

## What's Checked

For each non-interactive extras script:

1. The script imports `active_inference` through the same subprocess path a
   user would use.
2. The script accepts `--save`.
3. The script exits with code 0.
4. A fresh NPZ+JSON sidecar pair is written under
   `output/data/extras/<topic>/`.

The suite also checks registry invariants, source API references, README
coverage, declared interactive wrappers, and finite/dynamic animation raw data.
Rendered PNG/GIF content is checked by
`scripts/validate_rendered_figures.py --root output/figures`.
