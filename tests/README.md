# `tests/` — pytest suite

The test directory mirrors `src/active_inference/` one-for-one, plus
`chapters/` and `extras/` subfolders for subprocess smoke tests.

```
tests/
├── core/            ← mirrors src/active_inference/core
├── estimators/      ← mirrors src/active_inference/estimators
├── utils/           ← mirrors src/active_inference/utils
├── visualizations/  ← mirrors src/active_inference/visualizations
├── menu/            ← mirrors src/active_inference/menu
├── web/             ← mirrors src/active_inference/web
├── chapters/        ← subprocess smoke tests for chapters/chapter_0{1..10}/
├── extras/          ← subprocess smoke tests for extras/<topic>/
└── test_orchestrator_provenance.py
```

## Running

```bash
# Full suite (unit tests + chapter smoke tests + animation tests)
pytest

# Unit tests only — skip the slow subprocess smoke tests
pytest tests/core tests/estimators tests/utils tests/visualizations

# Chapter smoke tests only
pytest tests/chapters -v

# Extras smoke tests only
pytest tests/extras -v

# A single module
pytest tests/core/test_distributions.py -v

# With coverage
pytest --cov=active_inference --cov-report=term-missing
```

## Per-folder coverage

| Folder | Mirrors | Files | What's tested |
|---|---|---|---|
| `core/` | `src/active_inference/core/` | Mirrors core modules including `test_continuous_learning.py` and `test_thermodynamics.py` | Densities, generative process/model, grid Bayes, LGS, variational free energy (Ch.4), thermodynamic bridge helpers, predictive coding (Ch.5), generalized filtering (Ch.6), active inference (Ch.7), continuous learning/attention/hierarchy (Ch.8), POMDPs (Ch.9–10), diagnostics, compose, posterior protocol, validators, types. |
| `estimators/` | `src/active_inference/estimators/` | Mirrors estimator modules including `test_continuous_learning.py` | Closed-form vs iterative MLE/MAP, OLS / BLR, EM, variational inference (Ch.4), predictive coding (Ch.5), generalized filtering (Ch.6), active inference (Ch.7), continuous learning/attention (Ch.8), POMDP estimators, cross-estimator recovery. |
| `utils/` | `src/active_inference/utils/` | `test_grids.py`, `test_io.py`, `test_logging.py` | Grid validation, path conventions, idempotent logger factory. |
| `visualizations/` | `src/active_inference/visualizations/` | `test_plotting.py`, `test_animations.py`, `test_diagnostics.py`, `test_interactive.py`, `test_unified.py` | Figures save correctly, animations are valid `FuncAnimation` objects, diagnostic plots, interactive slider callbacks, the composable Ch.4–10 `unified` layer. |
| `menu/` | `src/active_inference/menu/` | `test_runner.py` | Chapter/script discovery, classification, menu rendering, resolution. |
| `web/` | `src/active_inference/web/` | `test_server.py` | Routes, markdown converter, run endpoint, templates, metadata, assets. |
| `chapters/` | `chapters/chapter_0{1..10}/` | `test_smoke.py` | Every non-interactive chapter script (examples, animations, visualizations) runs to exit 0 with `--save` and `PYTHONWARNINGS=error`. |
| `extras/` | `extras/<topic>/` | `test_smoke.py` | Every extras topic script runs to exit 0 with `--save` and writes fresh raw-data sidecars. |
| `test_orchestrator_provenance.py` | `chapters/`, `extras/`, `scripts/` | root test | The non-mutating provenance validator rejects sibling-wrapper imports and enforces the `active_inference` source-method boundary. |

## Design Decisions

- **No mocks.** All tests run real `numpy`/`scipy`/`matplotlib` code on
  real arrays. Visualization tests use `MPLBACKEND=Agg` so no display is
  required.
- **`tests/chapters/test_smoke.py`** uses `subprocess` so each chapter
  script runs through its `argparse` path exactly as a user would invoke
  it.
- **`tests/extras/test_smoke.py`** applies the same subprocess contract to
  cross-cutting topic orchestrators.
- **Test files mirror module file names** (`test_<module>.py`).
- **Test classes group related assertions**; test methods are named
  declaratively (e.g., `test_pdf_integrates_to_one`,
  `test_batch_matches_sequential`).

## Coverage targets

| Layer | Target | Rationale |
|---|---|---|
| `core/` | ≥ 90% | Critical math; bugs propagate everywhere. |
| `estimators/` | ≥ 90% | Algorithms must be verified against closed forms. |
| `utils/` | ≥ 95% | Tiny modules; cheap to fully cover. |
| `visualizations/` | ≥ 80% | GUI display branches are excluded, but constructors, callbacks, figures, animations, and saved artifacts are tested under `Agg`. |
| `chapters/` | smoke only | Exit-code 0 with `--save` is the contract. |
