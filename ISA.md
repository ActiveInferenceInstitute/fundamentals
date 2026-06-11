---
project: fundamentals-active-inference
task: Keep the textbook companion source of truth aligned with live code
effort: E5
phase: pdf-grounded-expansion
mode: implementation
updated: 2026-06-11
---

# ISA — Fundamentals of Active Inference, Python Companion

This file is the top-level project record. It must agree with the live
repository and with the inspected PDF source spine, not with stale plans.

## Source Spine

- Authority: `/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf`.
- Observed PDF metadata: 1153 pages, LaTeX build date 2025-03-20.
- The source spine has Chapters 1-14 and Appendices A-D.
- The source spine has no Chapter 15. `chapter_15` must remain absent unless a
  different source manuscript is supplied and the ledger is updated.
- `src/active_inference/source_spine.py` records this ledger and
  `scripts/validate_source_spine.py --require-pdf` checks it against the repo.

## Live State

- Chapters 1-10 are implemented as the stable Part I/II spine.
- Chapter 8 has learning, attention, continuous hierarchy, message passing, and
  animation coverage; richer nonlinear hierarchy examples remain future work.
- Chapter 9 is complete for its declared scope: categorical POMDP state
  inference, dynamic filtering, discrete VFE, EFE, Grid World planning, and
  exploration/exploitation.
- Chapter 10 is complete for its declared scope: Dirichlet `A`/`B`/`D`
  learning, novelty, habit, policy precision, factorial depth, two-armed
  bandit, and hierarchical depth.
- Chapters 11-14 are now first-class Part III chapter folders with
  source-spine sections, reusable library helpers, docs, smoke-test discovery,
  rendered figures, and NPZ/JSON raw-data sidecars. They are PDF-audited
  conservative demos, not a claim that every manuscript equation has an exact
  numeric worked example.
- Appendices B-D contribute executable math helpers and extras topics.
  Appendix A is represented as historical/future-context documentation and
  source-map extras rather than algorithmic code.

## Current Increment

This increment expands the companion from working Part III scaffolds to a
PDF-grounded Chapter 1-14 plus Appendix A-D contract:

1. Add a static source-spine ledger and validator for Chapters 1-14,
   Appendices A-D, and explicit rejection of Chapter 15.
2. Extend `active_inference.extra_topics` to 71 source-backed topics with
   rendered-output and raw-data checks.
3. Add Appendix C/D math helpers in `core.appendix_math`, `core.noise`,
   `core.model_comparison`, and `core.free_energy_forms`.
4. Extend Part III methods in `core.pomdp_extensions`, `core.factor_graph`,
   `core.bayesian_mechanics`, `estimators.pomdp_extensions`, and
   `estimators.applications`.
5. Expand Chapter 11-14 orchestrators while preserving the thin-wrapper rule:
   computation lives in `src/active_inference`; wrappers import only from
   `active_inference` and the standard library.

## Verification Contract

The acceptance gates for this state are:

- `uv run pytest`
- `uv run pytest --cov=active_inference --cov-report=term-missing`
- `uv run ruff check .`
- `uv run python scripts/validate_source_spine.py --require-pdf`
- `uv run python scripts/validate_orchestrator_provenance.py`
- `uv run python scripts/validate_book_topic_coverage.py --require-rendered`
- `uv run python scripts/validate_raw_data_exports.py --root output/data`
- `uv run python scripts/validate_rendered_figures.py --root output/figures`

## Open Work

- Keep package coverage at or above the documented 90% target as new APIs are
  added.
- Expand Chapter 8 with richer nonlinear hierarchy examples before marking
  that subsection fully saturated.
- Replace conservative Chapter 11-14 teaching demos with exact
  manuscript-numbered worked examples when the PDF provides enough numeric
  detail to make them testable.
- Continue adding mutation-style negative controls for Part III and appendices:
  sign flips, `A` vs `A.T`, raw vs softmax/log preferences, tensor axis order,
  stale manifests/source APIs, and non-finite trajectories.

## Non-Negotiables

- Public imports remain backward compatible.
- Chapter and extras orchestrators stay thin and import from `active_inference`,
  not from sibling wrappers.
- Every non-interactive `--save` script writes both a visual artifact and a
  raw-data sidecar pair: compressed `NPZ` arrays plus a `JSON` manifest.
- The Active Inference Institute and reading-group links remain visible in
  user-facing docs.
