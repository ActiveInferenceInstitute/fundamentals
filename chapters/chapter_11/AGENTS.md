# `chapters/chapter_11/` - agent guide

Chapter 11 scripts promote Part III planning extensions from extras sketches into
thin, tested chapter orchestrators.

## Rules

- Keep scripts as wrappers around `active_inference` APIs.
- Preserve `--save`; stochastic additions must also accept `--seed`.
- Pair every saved figure with `save_chapter_data(11, ...)`.
