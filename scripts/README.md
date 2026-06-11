# scripts/ — Batch Runners and Pipeline

Utility scripts for rendering figures and validating chapter or extras
orchestrators in bulk. For interactive day-to-day use prefer the top-level
[`run.sh`](../run.sh) menu; the files here are still wired into CI and the
historical batch workflow.

## Files

| File | Description |
|---|---|
| `run_all_figures.py` | Render all chapter figures to `output/figures/`. |
| `validate_book_topic_coverage.py` | Check the book-topic coverage matrix against the live extras registry and folders; `--require-rendered` also verifies expected extras PNG/GIF and NPZ+JSON artifacts. |
| `validate_orchestrator_provenance.py` | Check that chapter and extras wrappers route through `active_inference`, avoid sibling-wrapper imports, and expose every registry-declared extras wrapper. |
| `validate_rendered_figures.py` | Check rendered PNG/GIF artifacts for corruption, blank output, tiny dimensions, and trivial GIFs. |
| `validate_raw_data_exports.py` | Check `output/data/` NPZ+JSON sidecars for missing partners, invalid arrays, and manifest drift. |
| `validate_source_spine.py` | Check the inspected PDF ledger: Chapters 1-14, Appendices A-D, and no Chapter 15. |
| `run_all_chapter_01.sh` | Shell shortcut for `--chapters 1`. |
| `run_all_chapter_02.sh` | Shell shortcut for `--chapters 2`. |
| `run_all_chapter_03.sh` | Shell shortcut for `--chapters 3`. |

## Usage

```bash
# Render everything (all discovered chapters)
uv run python scripts/run_all_figures.py

# Render specific chapters
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 4 5

# Clean old generated figure media before re-rendering
uv run python scripts/run_all_figures.py --clean

# Skip slow GIF renderers
uv run python scripts/run_all_figures.py --no-animations

# Render extras only
uv run python scripts/run_all_figures.py --no-chapters --extras entropy expected_free_energy
uv run python scripts/run_all_figures.py --no-chapters --extras

# Continue even if one script fails
uv run python scripts/run_all_figures.py --keep-going

# Validate rendered artifacts after a render
uv run python scripts/validate_rendered_figures.py --root output/figures
uv run python scripts/validate_book_topic_coverage.py
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_orchestrator_provenance.py
uv run python scripts/validate_source_spine.py --require-pdf
uv run python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14
uv run python scripts/validate_raw_data_exports.py --root output/data

# Combine flags
uv run python scripts/run_all_figures.py --clean --keep-going --chapters 2

# Same flow via the top-level text menu (recommended for interactive use)
./run.sh --all
./run.sh --chapter 3
```

If `uv` is not available, drop the `uv run` prefix — the scripts also work
with the system `python`.

## Details

`run_all_figures.py`:

- Sets `MPLBACKEND=Agg` so it works on headless servers and in CI.
- Sets `PYTHONWARNINGS=error` so chapter scripts cannot hide warning regressions.
- Adds `src/` to `PYTHONPATH` so `import active_inference` works without
  installing the package.
- `--clean` removes stale generated media files while preserving hand-maintained
  documentation in `output/figures/` and stale raw-data `.npz` / `.json`
  files while preserving documentation in `output/data/`.
- Every `--save` chapter script is expected to produce both visual artifacts
  and raw-data sidecars through `save_chapter_data` (usually called indirectly
  by shared figure/animation helpers). Extras scripts use `save_extra_data`
  under `output/data/extras/<topic>/`.
- Chapter 1: runs files matching `0*.py` in `chapters/chapter_01/`.
- Chapter 2: runs all `example_*.py` + `visualize_*.py` + `animation_*.py`
  files, skipping anything with `interactive` in the name.
- Chapters 3 and later: same conventions as chapter 2 (`--chapters` accepts
  every chapter discovered under `chapters/chapter_NN/`).
- Reports success/failure per script and exits non-zero on first failure
  (unless `--keep-going`).

The shell wrappers are 3-line passthroughs to `run_all_figures.py
--chapters <N>`. They exist so contributors can re-render a single chapter
without remembering the underlying argparse vocabulary.

## Relationship to `run.sh`

`run.sh` at the repo root dispatches to one of two front ends:

- `python -m active_inference.menu` (default) — the text menu.
- `python -m active_inference.web` (when `--web` is passed) — the local
  browser UI.

Both discover chapter and extras scripts folder-by-folder (they do **not** call
into `run_all_figures.py`) and apply the same conventions
(`MPLBACKEND=Agg`, `--save`, skip `interactive_*`). Pick whichever
surface fits the situation:

| Use case | Reach for |
|---|---|
| Hands-on exploration in the terminal | `./run.sh` |
| Browser gallery + render buttons | `./run.sh --web` |
| CI / nightly figure regen | `scripts/run_all_figures.py` |
| Book-topic coverage QA | `scripts/validate_book_topic_coverage.py` |
| Book-topic rendered extras QA | `scripts/validate_book_topic_coverage.py --require-rendered` |
| PDF source-spine QA | `scripts/validate_source_spine.py --require-pdf` |
| Orchestrator/source-method QA | `scripts/validate_orchestrator_provenance.py` |
| Artifact QA after render | `scripts/validate_rendered_figures.py` |
| Raw-data QA after render | `scripts/validate_raw_data_exports.py` |
| Re-render a single chapter | any of the three |
| Re-render extras topics | `uv run python -m active_inference.menu --extras` |
| Programmatic discovery | `active_inference.menu.runner` |

## CI Integration

These scripts are used by the GitHub Actions CI workflow to generate
figures and verify they render without errors on every push / PR.
