# `tests/` — pytest suite

The test directory mirrors `src/active_inference/` one-for-one, plus a
`chapters/` subfolder for the headless smoke tests that exercise every
chapter orchestrator.

```
tests/
├── core/            ← tests/core mirrors src/active_inference/core
├── estimators/      ← tests/estimators mirrors src/active_inference/estimators
├── utils/           ← tests/utils mirrors src/active_inference/utils
├── visualizations/  ← tests/visualizations mirrors src/active_inference/visualizations
└── chapters/        ← subprocess smoke tests for chapters/chapter_0{1,2,3}/
```

## Running

```bash
# Full suite (unit tests + chapter smoke tests + animation tests)
pytest

# Unit tests only — skip the slow subprocess smoke tests
pytest tests/core tests/estimators tests/utils tests/visualizations

# Smoke tests only
pytest tests/chapters -v

# A single module
pytest tests/core/test_distributions.py -v

# With coverage
pytest --cov=active_inference --cov-report=term-missing
```

## Per-folder coverage

| Folder | Mirrors | Files | What's tested |
|---|---|---|---|
| `core/` | `src/active_inference/core/` | `test_distributions.py`, `test_distributions_mvn.py`, `test_generative_process.py`, `test_generative_model.py`, `test_inference.py`, `test_lgs.py` | Univariate + multivariate densities, generative process/model classes, grid Bayesian inference, closed-form LGS posterior. |
| `estimators/` | `src/active_inference/estimators/` | `test_mle.py`, `test_map.py`, `test_gradient_descent.py`, `test_linear_regression.py`, `test_em.py` | Closed-form vs iterative MLE/MAP, OLS / BLR, factor-analysis EM monotonicity & subspace recovery. |
| `utils/` | `src/active_inference/utils/` | `test_grids.py`, `test_io.py`, `test_logging.py` | Grid validation, path conventions, idempotent logger factory. |
| `visualizations/` | `src/active_inference/visualizations/` | `test_plotting.py`, `test_animations.py` | Figures save correctly, animations are valid `FuncAnimation` objects. |
| `chapters/` | `chapters/chapter_0{1,2,3}/` | `test_smoke.py` | Every chapter script runs to exit 0 with `--save`. |

## Design Decisions

- **No mocks.** All tests run real `numpy`/`scipy`/`matplotlib` code on
  real arrays. Visualization tests use `MPLBACKEND=Agg` so no display is
  required.
- **`tests/chapters/test_smoke.py`** uses `subprocess` so each chapter
  script runs through its `argparse` path exactly as a user would invoke
  it.
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
| `visualizations/` | ≥ 80% | Some matplotlib branches are GUI-only. |
| `chapters/` | smoke only | Exit-code 0 with `--save` is the contract. |
