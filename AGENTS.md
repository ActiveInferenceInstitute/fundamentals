# AGENTS.md — Fundamentals of Active Inference (Python Companion)

This file guides AI assistants working with the Fundamentals of Active Inference
Python companion repository.

The project is open source and maintained by the
[Active Inference Institute](https://activeinference.institute/); ongoing
cohort-based textbook reading groups run at
[textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/).
Keep that link present in user-facing docs and the web-UI footer if you
restructure them.

## Repository Overview

This repo provides a tested Python implementation of the algorithms described in
Sanjeev V. Namjoshi's *Fundamentals of Active Inference* (MIT Press, 2026),
plus thin orchestrator scripts that reproduce every figure and numerical example
in Chapters 1–3.

## Architecture

The codebase follows a two-layer architecture:

- **Layer 1 (library):** `src/active_inference/` — reusable, documented, tested
  classes and functions organized into six subpackages: `core`, `estimators`,
  `visualizations`, `utils`, plus the strict-UI peers `menu` (stdlib text
  menu) and `web` (stdlib local web server).
- **Layer 2 (orchestrators):** `chapters/chapter_01/` through
  `chapters/chapter_03/` — thin scripts (≤ ~120 lines each) that wire
  library components together to produce figures and numerical results.

All business logic lives in `src/`. Scripts orchestrate; they do not contain
domain logic.

## Key Directories

| Directory | Purpose |
|---|---|
| `run.sh` | Top-level thin wrapper around the chapter-runner text menu. |
| `pyproject.toml` / `uv.lock` | PEP 621 metadata; uv is the recommended package manager. |
| `src/active_inference/` | Python package (import as `active_inference`) |
| `src/active_inference/core/` | Distributions, generative process/model, exact inference, LGS, diagnostics, composition, posterior protocol, validators |
| `src/active_inference/estimators/` | MLE, MAP, gradient descent, linear regression, EM/factor analysis |
| `src/active_inference/visualizations/` | Static plots, interactive slider widgets, matplotlib animations, diagnostic figures, repo-wide style |
| `src/active_inference/utils/` | Grid constructors, logger factory, path helpers |
| `src/active_inference/menu/` | Stdlib-only text menu used by `run.sh` |
| `src/active_inference/web/` | Stdlib-only local HTTP server used by `run.sh --web` |
| `chapters/chapter_01/` | 4 orchestrator scripts for Part I Ch. 1 concepts |
| `chapters/chapter_02/` | 10 examples + 2 auxiliary + 2 animations |
| `chapters/chapter_03/` | 7 examples + 8 animations + 3 diagnostic visualizations |
| `tests/` | pytest unit tests + chapter smoke tests |
| `docs/` | Architecture, notation, chapter concept maps, topic walkthroughs, uv workflow |
| `scripts/` | Batch figure renderer (`run_all_figures.py`) + per-chapter shell wrappers |
| `output/figures/` | Generated PNGs (gitignored, regenerated via scripts) |

## Ground Truth Sources

- **Active chapter list:** Chapters 1–3 are present on disk; Ch. 1–2 are fully
  documented in the book, Ch. 3 scripts mirror the manuscript examples.
- **Canonical import surface:** Defined in `src/active_inference/__init__.py`
  and its `__all__` list.
- **Notation mapping:** `docs/notation.md` maps every symbol to its Python
  identifier.
- **Architecture diagram:** `docs/architecture.md` contains the layered design
  and key types table.

## Conventions

- All variances are *variances*, not standard deviations.
- Densities are evaluated on 1-D NumPy grids; integration uses `np.trapezoid`.
- Every chapter script accepts `--save` for headless rendering and `--seed`
  for reproducibility.
- Random number generators are passed explicitly via `numpy.random.Generator`
  — no global state.
- Chapter scripts import only from `active_inference` or the Python standard
  library — never from other chapter scripts.
- `MPLBACKEND=Agg` is used in all CI and smoke-test contexts so no display is
  required.

## Testing

```bash
# Full suite (unit tests + chapter smoke tests)
uv run pytest                                # or: pytest

# Unit tests only — skip the slow subprocess smoke tests
uv run pytest tests/core tests/estimators tests/utils tests/visualizations

# Smoke tests (run every chapter script with --save)
uv run pytest tests/chapters -v

# Coverage
uv run pytest --cov=active_inference --cov-report=term-missing
```

Coverage targets: 90%+ for `src/active_inference/` (excluding thin wrappers
that mainly glue imports).

## How to Add a New Example

1. Add a method or class to the appropriate `src/active_inference/` submodule
   (with corresponding unit tests in `tests/<sub>/`).
2. Create a thin orchestrator in the appropriate `chapters/chapter_<N>/`
   directory (≤ ~120 lines; imports only from `active_inference`).
3. Accept `--save` and `--seed` CLI flags.
4. Document the script in the chapter's `README.md`.
5. The `tests/chapters/test_smoke.py` parametrize globs and the
   `active_inference.menu` discovery both pick the file up automatically
   as long as it follows the `example_*.py` / `animation_*.py` /
   `visualize_*.py` / `0N_*.py` naming convention.
6. `scripts/run_all_figures.py` discovers the same way — no edit needed.

## Environment management

- The recommended workflow is `uv sync` + `uv run` (see
  [`docs/uv.md`](docs/uv.md)). Plain `pip install -e ".[dev]"` is still
  supported.
- `uv.lock` is committed; `.venv/` is git-ignored.
- After changing dependencies in `pyproject.toml`, run `uv lock` and
  commit the regenerated lockfile.