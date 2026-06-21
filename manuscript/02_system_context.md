# System Context {#sec:system_context}

## Project Boundary

An open-source Python companion to Fundamentals of Active Inference, implementing algorithms, figures, raw-data sidecars, and validators aligned to the inspected source spine.

## Source Surfaces

- `chapters/`
- `src/active_inference/`
- `scripts/`
- `tests/`
- `docs/`
- `output/`

## Template Boundary

The private project lives in the sidecar repository. Rendering and validation run through the sibling public template checkout after `link-projects` mirrors the project into `template/projects/` as a local symlink.
