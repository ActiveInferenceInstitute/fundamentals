# `docs/reference/` — API reference per src subpackage

One Markdown file per subpackage under `src/active_inference/`. These
pages enumerate every public symbol with its signature, purpose, and
shape contract — the closest thing this repository has to generated
docs.

## Pages

| File | Mirrors | Public symbols |
|---|---|---|
| [`core.md`](core.md) | `src/active_inference/core/` | distributions (univariate + multivariate), generative process / model, exact inference, LGS, **diagnostics**. |
| [`estimators.md`](estimators.md) | `src/active_inference/estimators/` | MLE, MAP, gradient descent, linear regression, Bayesian linear regression, factor-analysis EM. |
| [`utils.md`](utils.md) | `src/active_inference/utils/` | grids, paths, logger factory. |
| [`visualizations.md`](visualizations.md) | `src/active_inference/visualizations/` | plotting helpers, slider widgets, GIF animations, statistical-diagnostic figures. |

## How each page is structured

1. **Subpackage overview** — one paragraph on what it is for.
2. **Module table** — each `.py` file mapped to its purpose.
3. **Public API table** — every symbol in `__all__` with signature.
4. **Conventions** — package-wide invariants (variances vs std, RNG
   passing, log-space arithmetic).
5. **See also** — relative links to topic / statistics pages that
   *use* the API.

## See also

- [`../topics/`](../topics/) — concept-driven pages that exercise the
  APIs documented here.
- [`../statistics/`](../statistics/) — narrower pages on individual
  statistical tools.
- [`../architecture.md`](../architecture.md) — layered design diagram
  showing how subpackages depend on each other.
