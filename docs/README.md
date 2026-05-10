# `docs/` — documentation hub

Reference documentation for the Active Inference Fundamentals Python
companion. Pages are organized into four subfolders by *audience and
purpose*, plus several cross-cutting files at the root.

> Open-source companion maintained by the
> [Active Inference Institute](https://activeinference.institute/).
> Ongoing reading-group cohorts are open to anyone — register at
> [textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/).

## Map

```
docs/
├── architecture.md            ← cross-cutting · system design + layer diagram
├── notation.md                ← cross-cutting · symbol-to-identifier table
├── cookbook.md                ← copy-paste recipes for the 10 most-used workflows
├── reading_order.md           ← reader-path guide
├── uv.md                      ← quick reference for the uv workflow
├── web.md                     ← local browser UI launched by `run.sh --web`
├── chapters/                  ← per-book-chapter concept maps
│   ├── chapter_01.md
│   ├── chapter_02.md
│   └── chapter_03.md
├── topics/                    ← cross-cutting concept walkthroughs
│   ├── active_inference.md
│   ├── bayesian_inference.md
│   ├── bayesian_mechanics.md
│   ├── free_energy_principle.md
│   ├── generative_models.md
│   ├── gradient_descent.md
│   ├── inverse_problem.md
│   ├── learning_and_inference.md
│   └── multivariate_gaussians.md
├── statistics/                ← statistical-tool reference
│   ├── calibration.md
│   ├── divergences.md
│   ├── effective_sample_size.md
│   ├── entropy.md
│   ├── posterior_predictive.md
│   └── scoring_rules.md
└── reference/                 ← per-subpackage API reference
    ├── core.md
    ├── estimators.md
    ├── utils.md
    └── visualizations.md
```

## Pick the right page

| You want to know… | Open |
|---|---|
| How to set up an environment | [`uv.md`](uv.md) |
| Browse figures + docs in a local web UI | [`web.md`](web.md) |
| Where to start reading | [`reading_order.md`](reading_order.md) |
| A copy-paste-ready recipe for a common workflow | [`cookbook.md`](cookbook.md) |
| How the package is organized | [`architecture.md`](architecture.md) |
| Which Python identifier maps to a book symbol | [`notation.md`](notation.md) |
| What Chapter *N* of the book covers and which scripts mirror it | [`chapters/chapter_<N>.md`](chapters/) |
| How Bayesian inference / regression / EM / ... works in this codebase | [`topics/`](topics/) |
| How a specific statistical tool is implemented | [`statistics/`](statistics/) |
| Every public symbol in a subpackage | [`reference/`](reference/) |

## Update policy

Each subfolder has its own `AGENTS.md` describing when and how to extend
it. The high-level rule is:

- `chapters/`: edit when a chapter script is added / removed / renamed.
- `topics/`: edit when a *concept* gains a new component — rare.
- `statistics/`: edit when `core.diagnostics` or `core.distributions`
  gains a new tool.
- `reference/`: edit whenever the public API of a subpackage changes
  (i.e., when `__all__` changes).

The two root files are cross-cutting and update with any structural change
to the codebase.
