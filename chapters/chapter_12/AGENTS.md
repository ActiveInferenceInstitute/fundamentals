# `chapters/chapter_12/` - agent guide

Chapter 12 scripts are thin wrappers for factor-graph and message-passing
helpers in `active_inference`.

## Rules

- Keep factor algebra in `src/active_inference/core/factor_graph.py`.
- Use `--save` plus `save_chapter_data(12, ...)` for every non-interactive script.
- Do not import from sibling chapter scripts.
