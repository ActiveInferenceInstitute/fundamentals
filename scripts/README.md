# scripts/ — Batch Runners and Pipeline

Utility scripts for rendering figures and validating chapter or extras
orchestrators in bulk. For interactive day-to-day use prefer the top-level
[`run.sh`](../run.sh) menu; the files here are still wired into CI and the
historical batch workflow.

## Files

| File | Description |
|---|---|
| `run_all_figures.py` | Render all chapter figures to `output/figures/`. |
| `export_notebooks.py` | Export chapter, extras, and demo orchestrators to `output/notebooks/`. |
| `validate_notebook_exports.py` | Validate exported `.ipynb` files against discovery inventory. |
| `validate_book_topic_coverage.py` | Check the book-topic coverage matrix against the live extras registry and folders; `--require-rendered` also verifies expected extras PNG/GIF and NPZ+JSON artifacts. |
| `validate_orchestrator_contracts.py` | Enforce chapter/extras filename discovery, allowed imports, and non-interactive `--save` support; warn on duplicate stems and line-count drift. |
| `validate_orchestrator_provenance.py` | Check that chapter and extras wrappers route through `active_inference`, avoid sibling-wrapper imports, and expose every registry-declared extras wrapper. |
| `validate_rendered_figures.py` | Check rendered PNG/GIF artifacts for corruption, blank output, tiny dimensions, and trivial GIFs. |
| `validate_raw_data_exports.py` | Check `output/data/` NPZ+JSON sidecars for missing partners, invalid arrays, and manifest drift. |
| `validate_source_spine.py` | Check the inspected PDF ledger: Chapters 1-14, Appendices A-D, and no Chapter 15. |
| `gnn_parser_probe.py` | Parse `specs/weather_pomdp.gnn.md` with the installed upstream `gnn` package and print what it read as JSON. Run by `validate_gnn_spec.py` under the pinned revision; not called directly. |
| `render_manuscript.py` | The authoritative manuscript render path: validates declared variables, the figure registry, bibliography keys, and section labels, then assembles `manuscript/` and renders the formats enabled in `manuscript/config.yaml`. `--check` stops after validation. |
| `validate_gnn_spec.py` | Validate the Chapter 9 GNN declaration with the parser revision pinned in `specs/gnn-pin.json` and require it to agree with `active_inference.gnn_spec`. Verifies the parser digests *before* parsing, binds the pin's tag to its commit (`--gnn-root` checkout or `--verify-tag-remote`; mismatch fails as `tag-commit-mismatch`), and compares the installed runtime's recorded revision with the pin. Exits 2 when no pinned runtime is available (a reported *not run*, never a pass). |
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
uv run python scripts/validate_orchestrator_contracts.py
uv run python scripts/validate_orchestrator_provenance.py
uv run python scripts/validate_source_spine.py --require-pdf
uv run python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14
uv run python scripts/validate_raw_data_exports.py --root output/data

# Validate the GNN declaration against the pinned upstream parser
# (see docs/gnn.md; exits 2 when no pinned runtime or checkout is available)
uv run python scripts/validate_gnn_spec.py --gnn-runtime /path/to/gnn-venv/bin/python --verify-tag-remote --json
GNN_ROOT=/path/to/Generalized_Notation_Notation GNN_PYTHON=/path/to/python uv run python scripts/validate_gnn_spec.py

# Validate and render the manuscript (see docs/gate.md)
uv run python scripts/render_manuscript.py --check
uv run python scripts/render_manuscript.py

# Export Jupyter notebooks
uv sync --extra notebooks
uv run python scripts/export_notebooks.py
uv run python scripts/export_notebooks.py --chapters 1 2 3
uv run python scripts/validate_notebook_exports.py

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
| Orchestrator contract QA | `scripts/validate_orchestrator_contracts.py` |
| Orchestrator/source-method QA | `scripts/validate_orchestrator_provenance.py` |
| Artifact QA after render | `scripts/validate_rendered_figures.py` |
| Raw-data QA after render | `scripts/validate_raw_data_exports.py` |
| GNN declaration QA (needs the pinned parser runtime) | `scripts/validate_gnn_spec.py` |
| Manuscript validation / render | `scripts/render_manuscript.py` |
| Notebook export | `scripts/export_notebooks.py` |
| Notebook QA after export | `scripts/validate_notebook_exports.py` |
| Re-render a single chapter | any of the three |
| Re-render extras topics | `uv run python -m active_inference.menu --extras` |
| Programmatic discovery | `active_inference.menu.runner` |

## CI Integration

`.github/workflows/ci.yml` runs four jobs, and every job calls one arm of
`./gate.sh` after installing the locked environment with `uv sync --locked`
(the lock must be current with `pyproject.toml`):

| Job | Arm | What it runs |
|---|---|---|
| `lint` | `./gate.sh lint` | `ruff check .` once, not once per interpreter |
| `tests (py3.11)` / `tests (py3.14)` | `./gate.sh fast` | the unit, contract, and documentation suites |
| `repository validators` | `./gate.sh validators`, `./gate.sh manuscript-check` | the validators listed above, plus the manuscript contract check |
| `gnn conformance (pinned)` | `./gate.sh gnn` | the pinned-runtime provisioning, `validate_gnn_spec.py`, and `tests/test_gnn_spec.py` with that runtime so the pinned-parser case executes rather than skipping |

Every job installs `uv` and its interpreter through the shared composite step
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`.

Not in CI, and why: the manuscript HTML/PDF render (`./gate.sh manuscript`)
needs `pandoc` and `xelatex` on the runner; the batch renderers (`run_all_figures.py`,
`export_notebooks.py`, the `run_all_chapter_*.sh` wrappers) write the gitignored
`output/` trees of a working copy; the full `uv run pytest` suite renders every
chapter, extras, and demo orchestrator through its `--save` CLI in a temporary
root and stays a local gate; `--require-pdf` forms need the machine-local
inspected source PDF; and `validate_raw_data_exports.py` needs generated
sidecars, which are gitignored.
