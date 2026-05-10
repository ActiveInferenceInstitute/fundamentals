# scripts/ — Batch Runners and Pipeline

Utility scripts for rendering figures and running chapter orchestrators in
bulk. For interactive day-to-day use prefer the top-level
[`run.sh`](../run.sh) menu; the files here are still wired into CI and the
historical batch workflow.

## Files

| File | Description |
|---|---|
| `run_all_figures.py` | Render all chapter figures to `output/figures/`. |
| `run_all_chapter_01.sh` | Shell shortcut for `--chapters 1`. |
| `run_all_chapter_02.sh` | Shell shortcut for `--chapters 2`. |
| `run_all_chapter_03.sh` | Shell shortcut for `--chapters 3`. |

## Usage

```bash
# Render everything (chapters 1, 2, and 3)
uv run python scripts/run_all_figures.py

# Render specific chapters
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 2
uv run python scripts/run_all_figures.py --chapters 3

# Clean old figures before re-rendering
uv run python scripts/run_all_figures.py --clean

# Skip slow GIF renderers
uv run python scripts/run_all_figures.py --no-animations

# Continue even if one script fails
uv run python scripts/run_all_figures.py --keep-going

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
- Adds `src/` to `PYTHONPATH` so `import active_inference` works without
  installing the package.
- Chapter 1: runs files matching `0*.py` in `chapters/chapter_01/`.
- Chapter 2: runs all `example_*.py` + `visualize_*.py` + `animation_*.py`
  files, skipping anything with `interactive` in the name.
- Chapter 3: same conventions as chapter 2.
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

Both discover chapter scripts folder-by-folder (they do **not** call
into `run_all_figures.py`) and apply the same conventions
(`MPLBACKEND=Agg`, `--save`, skip `interactive_*`). Pick whichever
surface fits the situation:

| Use case | Reach for |
|---|---|
| Hands-on exploration in the terminal | `./run.sh` |
| Browser gallery + render buttons | `./run.sh --web` |
| CI / nightly figure regen | `scripts/run_all_figures.py` |
| Re-render a single chapter | any of the three |
| Programmatic discovery | `active_inference.menu.runner` |

## CI Integration

These scripts are used by the GitHub Actions CI workflow to generate
figures and verify they render without errors on every push / PR.
