# `docs/reference/` — API reference and coverage contracts

Most files here are one Markdown page per subpackage under
`src/active_inference/`. Those pages enumerate every public symbol with its
signature, purpose, and shape contract — the closest thing this repository has
to generated docs. The directory also contains the book-topic coverage matrix
that audits the repo-root `extras/` curriculum against the live registry.

## Pages

| File | Mirrors | Public symbols |
|---|---|---|
| [`book_topic_matrix.md`](book_topic_matrix.md) | `extras/` + `active_inference.extra_topics` | Book-section coverage by extras topic and artifact mode. |
| [`orchestrator_provenance.md`](orchestrator_provenance.md) | `chapters/`, `extras/`, `scripts/validate_orchestrator_provenance.py` | Source-method boundary for thin wrappers, source API metadata, and artifact validation gates. |
| [`core.md`](core.md) | `src/active_inference/core/` | distributions, generative process / model, exact inference, LGS, diagnostics, variational free energy (Ch.4), free-energy variants, factor-graph messages, ergodic-density helpers, thermodynamic/FEP bridge helpers, predictive coding (Ch.5), generalized filtering / action (Ch.6–7), continuous learning / hierarchy (Ch.8), and POMDPs (Ch.9–10). |
| [`estimators.md`](estimators.md) | `src/active_inference/estimators/` | MLE, MAP, gradient descent, linear regression, Bayesian linear regression, factor-analysis EM, variational inference (Ch.4), predictive coding (Ch.5), generalized filtering / action (Ch.6–7), continuous learning (Ch.8), and POMDP estimators (Ch.9–10). |
| [`utils.md`](utils.md) | `src/active_inference/utils/` | grids, paths, logger factory, and NPZ+JSON raw-data exports. |
| [`visualizations.md`](visualizations.md) | `src/active_inference/visualizations/` | plotting helpers, slider widgets, GIF animations, statistical-diagnostic figures, the composable Ch.4–10 `unified` layer, and colourblind-safe style. |

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
- [`book_topic_matrix.md`](book_topic_matrix.md) — book-grounded extras
  curriculum coverage contract.
- [`orchestrator_provenance.md`](orchestrator_provenance.md) — provenance
  contract for chapter and extras wrappers.
- [`../statistics/`](../statistics/) — narrower pages on individual
  statistical tools.
- [`../architecture.md`](../architecture.md) — layered design diagram
  showing how subpackages depend on each other.
- [`../../extras/README.md`](../../extras/README.md) — live extras topic index
  grouped by curriculum family.
