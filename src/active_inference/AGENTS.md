# `src/active_inference/` — agent guide

The Python package itself. Importable as `active_inference` once the repo
is installed (`uv pip install -e .` or `pip install -e .`). Six
subpackages, each with a clear role and a strict dependency direction.

## Layered subpackages

```
chapters/  ──── thin orchestrators (not in this folder)
                          │
       ┌──────────────────┼──────────────────┐
       ▼                                     ▼
menu/  ─── stdlib text menu             web/  ─── stdlib local web UI
       (used by run.sh)                       (used by run.sh --web)
                          │
                          ▼
visualizations/  ─── matplotlib helpers, GIF animations, sliders, diagnostic plots, style
                          │
                          ▼
estimators/      ─── MLE, MAP, GD, BLR, EM
                          │
                          ▼
core/            ─── distributions, gen-process/model, exact inference, LGS, diagnostics, composition, validators
                          │
                          ▼
utils/           ─── grids, paths, logger
```

Both `menu/` and `web/` consume the same discovery layer
(`menu/runner.py`) — adding a chapter or script wires both UIs at once.

Higher layers may import from lower layers; the reverse is forbidden.
This is enforced by convention only — there is no import-linter, but
every PR is reviewed against this graph.

## Public-surface policy

`__init__.py` re-exports the canonical public names so chapter
orchestrators can `from active_inference import …`. The rules:

1. **Anything in `__all__` is part of the public API** and is covered by
   semver-style backwards compatibility.
2. **Anything not in `__all__` is internal** — chapter scripts must not
   import it.
3. New public symbols require an update to `__all__`, the relevant
   `docs/<module>.md`, and a unit test in `tests/<module>/`.

## When to add a new subpackage

Almost never. The current six-package layout already maps cleanly onto the
book's structure (data + densities → estimation → results → visualization
→ text UI / web UI). If you find yourself wanting a seventh, prefer:

- A new module inside an existing subpackage, or
- A new top-level helper file pulled into one of the existing folders.

`menu/` and `web/` are strict UI layers: stdlib-only, no domain
imports. Keep numerical code out of them.

## Conventions

- All public functions are type-hinted.
- All numerical work is `numpy`; multivariate solves go through
  `scipy.linalg.solve_triangular` for stability; nothing else.
- Random number generators are passed in via `rng: np.random.Generator`.
- Variances and covariances, never standard deviations.
- Variances are always positive; matrices are always symmetric and PSD.
- Loggers come from `utils.logging.get_logger`, never from `print()`.

## Testing

`pytest tests/` runs the entire suite. The directory layout under
`tests/` mirrors this folder one-for-one:

```
tests/core/           ←→  src/active_inference/core/
tests/estimators/     ←→  src/active_inference/estimators/
tests/utils/          ←→  src/active_inference/utils/
tests/visualizations/ ←→  src/active_inference/visualizations/
tests/menu/           ←→  src/active_inference/menu/    (smoke + discovery)
tests/web/            ←→  src/active_inference/web/     (in-process HTTP smoke)
tests/chapters/       ←→  chapters/  (subprocess smoke tests)
```

Coverage targets: 90%+ for `core/` and `estimators/`; 80%+ for
`visualizations/` (some matplotlib branches are GUI-only and skipped in
headless tests).
