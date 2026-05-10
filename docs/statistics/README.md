# `docs/statistics/` — statistical-tool reference

One page per family of statistical tools shipped in
`active_inference.core.diagnostics`. These pages exist to make the
library's statistical claims auditable: each page lists the formula,
the closed-form variant where available, the implementation, and the
unit tests that lock the behavior down.

## Pages

| File | Covers |
|---|---|
| [`divergences.md`](divergences.md) | KL divergence (grid + Gaussian closed forms). |
| [`entropy.md`](entropy.md) | Differential entropy (grid + Gaussian closed forms). |
| [`scoring_rules.md`](scoring_rules.md) | Log score and CRPS for Gaussian forecasts. |
| [`calibration.md`](calibration.md) | Coverage, calibration curves, ECE. |
| [`posterior_predictive.md`](posterior_predictive.md) | Bayesian posterior-predictive checks. |
| [`effective_sample_size.md`](effective_sample_size.md) | Kish ESS in log space. |

## How each page is structured

1. **Definition** — the formula, in our own prose.
2. **Closed form** when one exists (often for Gaussians).
3. **API** — function names, argument tables, return shapes.
4. **Tests that pin it down** — pointers to the relevant `tests/core/`
   classes.
5. **Pitfalls** — actual numerical / interpretation gotchas.
6. **See also** — relative links into `topics/` and `reference/`.

## See also

- [`../topics/`](../topics/) — concept walkthroughs that *use* these
  tools.
- [`../reference/core.md`](../reference/core.md) — full API listing
  including `core.diagnostics`.
