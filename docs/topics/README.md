# `docs/topics/` — concept-by-concept walkthroughs

Topic pages explain a single idea that may span more than one book
chapter. They are the natural home for cross-cutting design notes that
would clutter a single chapter page.

## Pages

| File | What it covers |
|---|---|
| [`bayesian_inference.md`](bayesian_inference.md) | Bayes' theorem, the four ingredients, grid approximation, and how the package realizes them. |
| [`generative_models.md`](generative_models.md) | The process / model split, the linear-Gaussian recipe, and the multivariate generalization. |
| [`learning_and_inference.md`](learning_and_inference.md) | When parameters are unknown: MLE, MAP, Bayesian linear regression, and EM. |
| [`multivariate_gaussians.md`](multivariate_gaussians.md) | Cholesky-based density / sampling, Mahalanobis distance, confidence ellipses. |
| [`gradient_descent.md`](gradient_descent.md) | Vanilla GD, learning-rate stability bound, finite-difference fallback, regularization. |
| [`inverse_problem.md`](inverse_problem.md) | Why naive inversion fails, how priors / hierarchies / noise restore identifiability. |

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
