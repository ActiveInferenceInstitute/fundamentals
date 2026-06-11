# `docs/topics/` — concept-by-concept walkthroughs

Topic pages explain a single idea that may span more than one book
chapter. They are the natural home for cross-cutting design notes that
would clutter a single chapter page. The runnable `extras/` curriculum is
broader and is indexed in
[`../reference/book_topic_matrix.md`](../reference/book_topic_matrix.md).

## Pages

| File | What it covers |
|---|---|
| [`bayesian_inference.md`](bayesian_inference.md) | Bayes' theorem, the four ingredients, grid approximation, and how the package realizes them. |
| [`generative_models.md`](generative_models.md) | The process / model split, the linear-Gaussian recipe, and the multivariate generalization. |
| [`learning_and_inference.md`](learning_and_inference.md) | When parameters are unknown: MLE, MAP, Bayesian linear regression, and EM. |
| [`multivariate_gaussians.md`](multivariate_gaussians.md) | Cholesky-based density / sampling, Mahalanobis distance, confidence ellipses. |
| [`gradient_descent.md`](gradient_descent.md) | Vanilla GD, learning-rate stability bound, finite-difference fallback, regularization. |
| [`inverse_problem.md`](inverse_problem.md) | Why naive inversion fails, how priors / hierarchies / noise restore identifiability. |
| [`active_inference.md`](active_inference.md) | The active-inference frame: perception, action, and expected free energy in context. |
| [`free_energy_principle.md`](free_energy_principle.md) | The Free Energy Principle and how variational free energy operationalizes it. |
| [`thermodynamic_bridge.md`](thermodynamic_bridge.md) | The explicit `U`, `S`, `T`, `H`, and `G` analogy layer used by extras. |
| [`bayesian_mechanics.md`](bayesian_mechanics.md) | Bayesian mechanics — the triangle linking Bayesian inference, the FEP, and active inference. |
| [`source_spine_and_appendices.md`](source_spine_and_appendices.md) | PDF source-spine status, no Chapter 15, Appendix A source map, and Appendix B-D executable helpers. |

## How to read these pages

Each topic page has the same shape:

1. **One-paragraph framing** of the idea.
2. **Implementation summary** — which functions / classes realize it.
3. **End-to-end snippet** showing the canonical use.
4. **Pitfalls / gotchas** — actual numerical behavior, not theory.
5. **Cross-links** to relevant chapter pages, statistics pages, and
   reference pages.

## See also

- [`../chapters/`](../chapters/) — per-book-chapter overviews tied to
  scripts.
- [`../statistics/`](../statistics/) — narrower pages on a single
  statistical tool.
- [`../reference/`](../reference/) — full API listing.
- [`../reference/book_topic_matrix.md`](../reference/book_topic_matrix.md) —
  book-section coverage for all extras topics.
