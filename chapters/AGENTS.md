# `chapters/` — agent guide

This directory holds the **thin orchestrator** layer. Each subfolder mirrors
one chapter of the book and contains short scripts that wire library
components from `active_inference` together to reproduce a figure or
numerical example.

## Hard rules

1. A chapter script imports **only** from `active_inference` and the Python
   standard library. No cross-script imports between chapters.
2. Every script is **≤ ~120 lines**. If you cross that, push the new logic
   into `src/active_inference/`.
3. Every script accepts at minimum `--save` and (where stochastic) `--seed`
   command-line flags.
4. Every script is registered in
   [`tests/chapters/test_smoke.py`](../tests/chapters/test_smoke.py) so it
   runs headlessly under `MPLBACKEND=Agg` on every CI run.

## Naming conventions

| Pattern | Meaning |
|---|---|
| `0N_<topic>.py` (chapter_01) | Concept demos that are not numbered examples in the book. |
| `example_N_M_<topic>.py` | Mirrors the book's numbered example *N.M*. |
| `visualize_<topic>.py` | Standalone visualization not tied to a numbered example. |
| `interactive_<topic>.py` | Slider-driven exploration (skipped in headless tests). |
| `animation_<topic>.py` | Saves a GIF; longer pytest timeout is enforced. |

## Adding a new chapter

1. `mkdir chapters/chapter_<N>` and add `README.md` + `AGENTS.md`.
2. Add the chapter's directory to `CHAPTER_DIRS` in
   `tests/chapters/test_smoke.py`.
3. Add the chapter's directory to `CHAPTER_DIRS` in
   `scripts/run_all_figures.py`.
4. Add a one-line entry to `README.md` and `docs/architecture.md` and a
   concept map to `docs/chapter_<N>_overview.md`.

## Running

```bash
# every chapter, every script, headless
python scripts/run_all_figures.py

# a single chapter
python scripts/run_all_figures.py --chapters 3
./scripts/run_all_chapter_03.sh

# a single script
python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save
```
