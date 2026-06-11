# Orchestrator Provenance Contract

The repository has two executable layers:

- `src/active_inference/` owns reusable mathematics, simulation builders,
  rendering helpers, validation, and raw-data export behavior.
- `chapters/chapter_<NN>/` and `extras/<topic>/` own thin wrappers that parse
  CLI flags and call the library.

This boundary keeps the book examples inspectable while making the underlying
methods unit-testable. Chapter and extras wrappers may import `active_inference`
and the Python standard library; they may not import sibling wrappers or hide
domain computation that belongs in `src/`.

## Chapter Wrappers

Chapter scripts mirror the Chapter 1-14 source spine. A typical script constructs a
process, model, estimator, or visualization from `active_inference`, then calls
`save_or_show`, `save_animation`, or `save_chapter_data` when `--save` is set.
The chapter smoke tests execute every non-interactive wrapper in a subprocess,
so import paths, CLI parsing, rendering, and raw-data export are tested through
the same path a user runs.

## Extras Wrappers

Extras are registry-driven. `active_inference.extra_topics.EXTRA_TOPICS`
declares the slug, family, book sections, demo kind, available modes, and
`source_apis` for each topic. The wrappers are intentionally narrow:

- `visualize_<topic>.py` calls `main_visualize(slug)`.
- `simulate_<topic>.py` calls `main_simulate(slug)` when the topic has a
  parameter-sweep simulation.
- `animation_<topic>.py` calls `main_animation(slug)` when the topic has a
  trajectory animation.
- `interactive_<topic>.py` calls `main_interactive(slug)` when the topic has a
  simulation-capable slider view.

The saved extras JSON manifests include the registry source API references, so
rendered artifacts can be traced back to tested library methods instead of only
to wrapper filenames.

## Artifact Contract

Every non-interactive `--save` run writes both a visual artifact and raw-data
sidecars:

- Chapters: `output/figures/chapter_<NN>/...` plus
  `output/data/chapter_<NN>/<stem>.npz` and `<stem>.json`.
- Extras: `output/figures/extras/<topic>/...` plus
  `output/data/extras/<topic>/<stem>.npz` and `<stem>.json`.

Interactive wrappers are GUI launchers and are intentionally excluded from
batch `--save` rendering. Their reusable constructors are covered by
`tests/visualizations/test_interactive.py`, which instantiates the figures under
`Agg`, moves sliders programmatically, and checks finite updated values.

## Validators

Use these gates together when changing orchestration, rendering, or docs:

```bash
uv run python scripts/validate_orchestrator_provenance.py
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_source_spine.py --require-pdf
uv run python scripts/validate_raw_data_exports.py --root output/data
uv run python scripts/validate_rendered_figures.py --root output/figures
```

`validate_orchestrator_provenance.py` is non-mutating. It checks that chapter
and extras scripts route through `active_inference`, avoid sibling-wrapper
imports, and that registry-declared extras wrappers exist. The book-topic
coverage validator checks the documentation matrix and, with
`--require-rendered`, confirms declared PNG/GIF artifacts and NPZ+JSON sidecars
exist for every rendered extras mode. The source-spine validator checks that
the inspected PDF remains Chapters 1-14 plus Appendices A-D and that Chapter 15
is not introduced without a new source.
