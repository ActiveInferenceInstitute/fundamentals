# `tests/extras/` - agent guide

Smoke tests for extras topic orchestrators.

## Rules

1. Keep tests subprocess-based so `argparse`, import paths, `--save`, and raw
   data export all run as a user would invoke them.
2. Discover topic slugs from `active_inference.extra_topics`; do not maintain
   a separate hard-coded topic list.
3. Require fresh NPZ+JSON sidecars under `output/data/extras/<topic>/`.
4. Use `MPLBACKEND=Agg`, `PYTHONWARNINGS=error`, and a `src/` `PYTHONPATH`
   entry for every subprocess.
5. Keep `interactive_*.py` out of subprocess `--save` smoke tests; validate
   those wrappers through discovery/provenance tests and
   `tests/visualizations/test_interactive.py`.

## Running

```bash
pytest tests/extras -v
```
