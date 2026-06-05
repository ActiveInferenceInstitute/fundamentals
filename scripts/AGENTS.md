# `scripts/` — agent guide

Top-level batch entry points. These are *not* part of the importable
package; they are convenience wrappers around the chapter orchestrators
in `chapters/` and validation utilities for chapter/extras artifacts.

## What lives here

| File | Purpose |
|---|---|
| `run_all_figures.py` | Render every chapter or extras script that accepts `--save` to `output/figures/`. Supports `--chapters`, `--extras`, `--no-chapters`, `--clean`, `--keep-going`, `--no-animations`. |
| `validate_book_topic_coverage.py` | Validate `docs/reference/book_topic_matrix.md` against the live extras registry, topic folders, READMEs, declared scripts, and, with `--require-rendered`, expected extras PNG/GIF plus NPZ+JSON artifacts. |
| `validate_orchestrator_provenance.py` | Validate that chapter/extras wrappers are thin, import `active_inference`, avoid sibling-wrapper imports, and match registry-declared extras scripts. |
| `validate_rendered_figures.py` | Validate generated PNG/GIF artifacts for corruption, blank output, tiny dimensions, and trivial GIFs. |
| `validate_raw_data_exports.py` | Validate generated NPZ+JSON raw-data sidecars for missing partners, invalid arrays, manifest drift, and required extras topics. |
| `run_all_chapter_01.sh` | Bash wrapper invoking `run_all_figures.py --chapters 1`. |
| `run_all_chapter_02.sh` | Same for Chapter 2. |
| `run_all_chapter_03.sh` | Same for Chapter 3. |

## Conventions

- Every render script must run with `MPLBACKEND=Agg` and `PYTHONWARNINGS=error`
  so it works in CI / headless containers and fails on warning regressions.
- Bash wrappers are 3 lines: `cd` to repo root, then exec the Python
  driver with `--chapters <N> "$@"` so callers can pass extra flags
  through (e.g., `--clean`, `--no-animations`).
- New batch tasks should land here only if they orchestrate work across
  multiple chapter or extras folders. Chapter-specific helpers belong inside
  `chapters/chapter_<N>/`; topic-specific helpers belong inside
  `extras/<topic>/`.

## Adding a new wrapper

1. Drop a new file in this folder.
2. `chmod +x` if it's a shell script.
3. Add a one-line description to `scripts/README.md`.
4. Hook it into CI by mentioning it in the root `README.md` "Running
   everything" section.
