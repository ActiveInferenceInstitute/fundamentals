# chapters/ — Chapter Orchestrators

Each subdirectory corresponds to a chapter of the book and contains thin
orchestrator scripts that reproduce its figures and numerical examples.

## Structure

```
chapters/
├── README.md                  ← this file
├── AGENTS.md
├── chapter_01/                ← The Hypothesis-Testing Brain
│   ├── 01_box_scenario.py
│   ├── 02_three_perspectives.py
│   ├── 03_bayes_intuition.py
│   └── 04_inverse_problem.py
├── chapter_02/                ← Hidden State Estimation
│   ├── example_2_1_linear_deterministic.py
│   ├── example_2_2_linear_probabilistic.py
│   ├── example_2_3_precision.py
│   ├── example_2_4_nonlinear_deterministic.py
│   ├── example_2_5_nonlinear_probabilistic.py
│   ├── example_2_6_imperfect_model.py
│   ├── example_2_7_multiple_samples.py
│   ├── example_2_8_mle_analytic.py
│   ├── example_2_9_map_analytic.py
│   ├── example_2_10_gradient_descent.py
│   ├── visualize_generative_model.py
│   ├── interactive_explorer.py
│   ├── animation_sequential.py
│   └── animation_gradient_descent.py
└── chapter_03/                ← Combining Learning and Inference
    ├── example_3_1_linear_regression_mle.py
    ├── example_3_2_linear_regression_gd.py
    ├── example_3_3_multiple_regression.py
    ├── example_3_4_multivariate_gaussian.py
    ├── example_3_5_bayesian_linear_regression.py
    ├── example_3_6_lgs_food_localization.py
    ├── example_3_7_factor_analysis_em.py
    ├── animation_bimodal_emergence.py
    ├── animation_blr_predictive_band.py
    ├── animation_blr_tightening.py
    ├── animation_em_convergence.py
    ├── animation_em_steps.py
    ├── animation_lgs_online.py
    ├── animation_precision_sweep.py
    ├── animation_sufficient_statistics.py
    ├── visualize_calibration.py
    ├── visualize_coverage.py
    └── visualize_posterior_predictive.py
```

## Running a single script

```bash
uv run python chapters/chapter_01/03_bayes_intuition.py --save
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py
uv run python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save
```

All scripts accept `--save` to write figures to `output/figures/` and
`--seed` for reproducibility. Chapter 2/3 scripts additionally accept
`--x-true`, `--y-obs`, `--n-samples`, etc. — see each script's docstring
or `--help`.

## Running all figures

The top-level [`run.sh`](../run.sh) menu is the simplest entry point:

```bash
./run.sh                        # interactive menu
./run.sh --all                  # every chapter, every script
./run.sh --chapter 3            # one chapter
./run.sh --no-animations        # skip GIF renderers
./run.sh --keep-going           # don't abort on first failure
```

The older batch pipeline still works:

```bash
uv run python scripts/run_all_figures.py              # chapters 1, 2, and 3
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 2
uv run python scripts/run_all_figures.py --chapters 3
uv run python scripts/run_all_figures.py --clean      # delete old figures first
```

Shell shortcuts are also available:

```bash
./scripts/run_all_chapter_01.sh
./scripts/run_all_chapter_02.sh
./scripts/run_all_chapter_03.sh
```

## Conventions

- Every script is a **thin orchestrator**: it imports from `active_inference`
  and standard library, never from sibling scripts.
- All business logic lives in `src/active_inference/`.
- Each script ends with `if __name__ == "__main__": main()`.
- Default seeds are fixed so figures are deterministic on re-run.

## Smoke Tests

Chapter scripts are exercised by `tests/chapters/test_smoke.py`, which
runs every script with `--save` and asserts exit code 0. The
[`run.sh`](../run.sh) text menu uses the same discovery rules.

## Coverage

| Chapter   | Scripts                                                                  | Status |
|-----------|--------------------------------------------------------------------------|--------|
| Chapter 1 | 4 concept orchestrators                                                  | Complete |
| Chapter 2 | 14 (10 examples + 2 auxiliary + 2 animations)                            | Complete |
| Chapter 3 | 18 (7 examples + 8 animations + 3 diagnostic visualizations)             | Scripts present; text pending |
