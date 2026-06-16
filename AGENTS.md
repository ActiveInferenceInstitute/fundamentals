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
plus thin orchestrator scripts that reproduce the figures and numerical examples
of Chapters 1–14 and source-map Appendix A-D coverage where executable.
Chapters 1–10 are the stable Part I/II spine; Chapters 11–14 are first-class
Part III companion implementations that must stay source-backed as manuscript
examples are audited into tests.

## Architecture

The codebase follows a two-layer architecture:

- **Layer 1 (library):** `src/active_inference/` — reusable, documented, tested
  classes and functions organized into six subpackages: `core`, `estimators`,
  `visualizations`, `utils`, plus the strict-UI peers `menu` (stdlib text
  menu) and `web` (stdlib local web server).
- **Layer 2 (orchestrators):** `chapters/chapter_01/` through
  `chapters/chapter_14/` plus `extras/<topic>/` — thin scripts (≤ ~120
  lines each) that wire library components together to produce figures and
  numerical results.

All business logic lives in `src/`. Scripts orchestrate; they do not contain
domain logic.

## Key Directories

| Directory | Purpose |
|---|---|
| `run.sh` | Top-level thin wrapper around the chapter-runner text menu. |
| `pyproject.toml` / `uv.lock` | PEP 621 metadata; uv is the recommended package manager. |
| `src/active_inference/` | Python package (import as `active_inference`) |
| `src/active_inference/source_spine.py` | Static PDF-derived ledger: Ch.1–14 and Appendices A-D; explicitly no Ch.15 for the inspected source. |
| `src/active_inference/core/` | Distributions, generative process/model, exact inference, LGS, diagnostics, composition, posterior protocol, validators, Appendix math/noise/model comparison, variational free energy (Ch.4), thermodynamic/FEP bridge helpers, predictive coding (Ch.5), generalized filtering (Ch.6), continuous active inference (Ch.7), continuous learning/attention/hierarchy (Ch.8), discrete POMDP + learning + factorial/hierarchical depth (Ch.9–10) in `pomdp.py`, and Part III helpers. |
| `src/active_inference/estimators/` | MLE, MAP, gradient descent, linear regression, EM/factor analysis, variational inference (Ch.4), predictive coding (Ch.5), generalized filtering (Ch.6), active inference (Ch.7), continuous learning/attention (Ch.8), POMDP planning + learning + two-armed bandit + hierarchical agent (Ch.9–10) |
| `src/active_inference/visualizations/` | Static plots, Ch.4 variational figures, the composable Ch.4–10 `unified` layer, interactive slider widgets, matplotlib animations, diagnostic figures, repo-wide colourblind-safe style |
| `src/active_inference/utils/` | Grid constructors, logger factory, path helpers, and `save_chapter_data` / `save_extra_data` NPZ+JSON export helpers |
| `src/active_inference/menu/` | Stdlib-only text menu used by `run.sh` |
| `src/active_inference/web/` | Stdlib-only local HTTP server used by `run.sh --web` |
| `chapters/chapter_01/` | 4 orchestrator scripts for Part I Ch. 1 concepts |
| `chapters/chapter_02/` | 10 examples + 2 auxiliary + 2 animations |
| `chapters/chapter_03/` | 7 examples + 8 animations + 3 diagnostic visualizations |
| `chapters/chapter_04/` | 5 examples + 1 animation + 3 visualizations + 1 interactive (variational inference) |
| `chapters/chapter_05/` | 4 examples + 2 animations (predictive coding) |
| `chapters/chapter_06/` | generalized filtering for perception (§6.1–6.6), including correlated embedding orders and Example 6.7 |
| `chapters/chapter_07/` | active generalized filtering / action (§7.1–7.5), including multivariate AIF in generalized coordinates |
| `chapters/chapter_08/` | learning, attention, and hierarchy in continuous state-spaces (§8.1–8.6) |
| `chapters/chapter_09/` | discrete POMDP active inference (§9.1–§9.6) |
| `chapters/chapter_10/` | learning & extensions (§10.1–10.4): 8 examples + 1 visualization + 3 animations |
| `chapters/chapter_11/` | Part III planning extensions: free-energy variants, sophisticated inference, tree search, preferences, forgetting, structure learning |
| `chapters/chapter_12/` | factor graphs, belief propagation, smoothing, VMP, and hybrid message bridges |
| `chapters/chapter_13/` | robotics navigation/control and social-inference application examples |
| `chapters/chapter_14/` | ergodic density, entropy bounds, Bayesian mechanics, and Markov blankets |
| `extras/` | cross-cutting topic orchestrators beyond the chapter spine, generated from the `active_inference.extra_topics` registry |
| `tests/` | pytest unit tests + chapter/extras smoke tests + provenance validators |
| `docs/` | Architecture, notation, chapter concept maps, topic walkthroughs, uv workflow, coverage/provenance references |
| `scripts/` | Batch figure renderer (`run_all_figures.py`), source-spine/raw-data/rendered/provenance validators, and per-chapter shell wrappers |
| `output/figures/` | Generated PNGs/GIFs (gitignored, regenerated via scripts) |
| `output/data/` | Generated raw-data NPZ arrays plus JSON metadata/manifests for every saved non-interactive chapter or extras artifact |

## Ground Truth Sources

- **Source spine:** `src/active_inference/source_spine.py` records the inspected
  PDF as Chapters 1–14 and Appendices A-D, with no Chapter 15. Run
  `uv run python scripts/validate_source_spine.py --require-pdf` after edits
  that touch chapter/appendix source status.
- **Active chapter list:** Chapters 1–14 are present on disk; each chapter's
  non-interactive scripts are covered by smoke tests. Ch.1–10 are the stable
  declared scope; Ch.11–14 are PDF-grounded Part III companion demos and should
  not be described as manuscript-complete until source-backed worked-example
  tests exist.
- **Canonical import surface:** Defined in `src/active_inference/__init__.py`
  and its `__all__` list.
- **Notation mapping:** `docs/notation.md` maps every symbol to its Python
  identifier.
- **Architecture diagram:** `docs/architecture.md` contains the layered design
  and key types table.
- **Extras registry and provenance:** `src/active_inference/extra_topics.py`
  declares topic modes and source API references; `docs/reference/book_topic_matrix.md`
  and `docs/reference/orchestrator_provenance.md` document the coverage and
  source-method contracts.

## Conventions

- All variances are *variances*, not standard deviations.
- Densities are evaluated on 1-D NumPy grids; integration uses `np.trapezoid`.
- Every chapter and extras script accepts `--save` for headless rendering;
  stochastic scripts also accept `--seed` for reproducibility.
- With `--save`, every non-interactive chapter or extras script must produce
  both the visual artifact and at least one raw-data sidecar. Chapters write
  `output/data/chapter_NN/<stem>.npz` + `<stem>.json`; extras write
  `output/data/extras/<topic>/<stem>.npz` + `<stem>.json`. Use
  `save_chapter_data` / `save_extra_data` directly for bespoke exports or the
  shared visualization save helpers for figure-derived exports.
- Random number generators are passed explicitly via `numpy.random.Generator`
  — no global state.
- Chapter and extras scripts import only from `active_inference`, the Python
  standard library, and the canonical scientific plotting dependencies already
  required by the script surface (`numpy`, `matplotlib`, and, where already
  needed by an existing workflow, `scipy`). They never import from other chapter
  or extras scripts. Keep domain logic in `src/active_inference/`; direct
  scientific imports in orchestrators are for CLI/rendering glue only.
- Extras JSON manifests include registry source API references for saved
  artifacts; update the registry and tests together when a topic's underlying
  method changes.
- `MPLBACKEND=Agg` is used in all CI and smoke-test contexts so no display is
  required.


## Orchestrator Contract Details

- `scripts/validate_orchestrator_contracts.py` is the executable gate for
  filename discovery, allowed imports, and non-interactive `--save` support.
  Use `--strict-warnings` only when intentionally turning soft debt into a
  blocking gate.
- Delegator wrappers such as `run.sh`, menu/web launchers, generated extras
  wrappers that call `main_visualize` / `main_simulate` / `main_animation`, and validator
  scripts are not chapter/extras orchestrators. They may import the internal
  `active_inference.menu`, `active_inference.web`, or validation helpers needed
  to dispatch or inspect workflows, but they should not contain domain logic.
- Shared helper scripts under `scripts/` should stay validator/maintenance
  oriented and stdlib-first. If they need numerical source behavior, import it
  from `active_inference` instead of reaching into chapter or extras wrappers.

## Testing

```bash
# Full suite (unit tests + chapter smoke tests)
uv run pytest                                # or: pytest

# Unit tests only — skip the slow subprocess smoke tests
uv run pytest tests/core tests/estimators tests/utils tests/visualizations

# Smoke tests (run every chapter script with --save)
uv run pytest tests/chapters -v

# Validate generated raw-data sidecars
uv run python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14

# Validate all generated chapter and extras raw-data sidecars
uv run python scripts/validate_raw_data_exports.py --root output/data

# Validate orchestrator contracts, rendered extras coverage, and source-method provenance
uv run python scripts/validate_orchestrator_contracts.py
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_orchestrator_provenance.py
uv run python scripts/validate_source_spine.py --require-pdf

# Coverage
uv run pytest --cov=active_inference --cov-report=term-missing
```

Coverage targets: 90%+ for `src/active_inference/` (excluding thin wrappers
that mainly glue imports).

## How to Add a New Example

1. Add a method or class to the appropriate `src/active_inference/` submodule
   (with corresponding unit tests in `tests/<sub>/`).
2. Create a thin orchestrator in the appropriate `chapters/chapter_<N>/`
   directory or `extras/<topic>/` directory (≤ ~120 lines; imports only from
   `active_inference`, stdlib, and narrowly scoped `numpy` / `matplotlib` /
   existing `scipy` rendering glue). Put reusable workflow logic in
   `src/active_inference/`, not in the wrapper.
3. Accept `--save`; add `--seed` whenever the script samples or otherwise
   depends on pseudo-randomness.
4. Document the script in the chapter or extras topic `README.md`.
5. Ensure `--save` writes reconstructable raw data. Prefer plotting through
   `save_or_show` / `save_animation`; if the script has non-plotted arrays,
   call `save_chapter_data(chapter, stem, arrays, metadata, figures=...)` for
   chapter scripts or `save_extra_data(topic, stem, arrays, metadata,
   figures=...)` for extras.
6. The smoke tests and `active_inference.menu` discovery pick the file up
   automatically as long as it follows the `example_*.py` / `animation_*.py` /
   `visualize_*.py` / `interactive_*.py` / `0N_*.py` naming convention.
7. `scripts/run_all_figures.py` discovers the same way — no edit needed.

## Environment management

- The recommended workflow is `uv sync` + `uv run` (see
  [`docs/uv.md`](docs/uv.md)). Plain `pip install -e ".[dev]"` is still
  supported.
- `uv.lock` is committed; `.venv/` is git-ignored.
- After changing dependencies in `pyproject.toml`, run `uv lock` and
  commit the regenerated lockfile.
