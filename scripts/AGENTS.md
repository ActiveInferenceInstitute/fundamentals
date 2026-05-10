# `scripts/` — agent guide

Top-level batch entry points. These are *not* part of the importable
package; they are convenience wrappers around the chapter orchestrators
in `chapters/`.

## What lives here

| File | Purpose |
|---|---|
| `run_all_figures.py` | Render every chapter script that accepts `--save` to `output/figures/`. Supports `--chapters`, `--clean`, `--keep-going`, `--no-animations`. |
| `run_all_chapter_01.sh` | Bash wrapper invoking `run_all_figures.py --chapters 1`. |
| `run_all_chapter_02.sh` | Same for Chapter 2. |
| `run_all_chapter_03.sh` | Same for Chapter 3. |

## Conventions

- Every script must run with `MPLBACKEND=Agg` so it works in CI / headless
  containers.
- Bash wrappers are 3 lines: `cd` to repo root, then exec the Python
  driver with `--chapters <N> "$@"` so callers can pass extra flags
  through (e.g., `--clean`, `--no-animations`).
- New batch tasks should land here only if they orchestrate work across
  multiple chapter folders. Chapter-specific helpers belong inside
  `chapters/chapter_<N>/`.

## Adding a new wrapper

1. Drop a new file in this folder.
2. `chmod +x` if it's a shell script.
3. Add a one-line description to `scripts/README.md`.
4. Hook it into CI by mentioning it in the root `README.md` "Running
   everything" section.
