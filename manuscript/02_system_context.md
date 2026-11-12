# System Context {#sec:system_context}

## Project Boundary

The companion is the `{{PACKAGE_NAME}}` package (version `{{PACKAGE_VERSION}}`) in `{{REPOSITORY_SLUG}}`: an open-source Python implementation of the textbook's algorithms, with thin orchestrators that render figures and raw-data sidecars, and validators that check the rendered artifacts. {{CHAPTER_SCRIPT_COUNT}} chapter orchestrators call the `active_inference` library; none of them re-derives the mathematics inline.

## Source Surfaces

- `chapters/`
- `src/active_inference/`
- `scripts/`
- `tests/`
- `docs/`
- `output/`

## GNN Declaration Boundary

The Chapter 9 weather model is declared once as Generalized Notation Notation under `specs/`, recorded against the `{{GNN_PIN_SCHEMA}}` pin schema. The pin names upstream tag `{{GNN_PIN_TAG}}`, commit `{{GNN_PIN_COMMIT}}`, and declared language version `{{GNN_PIN_LANGUAGE_VERSION}}`; `scripts/validate_gnn_spec.py` digest-verifies the installed parser files against that record before parsing. A pin names the reviewed revision, not what a future consumer executes.

## Render Boundary

Rendering is project-local. `scripts/render_manuscript.py` validates the declared variables, the figure registry, the bibliography keys, and the section labels, then assembles the admitted sections and calls pandoc. It reads `manuscript/config.yaml` for title, authors, bibliography, and the enabled formats. The prose does not depend on a sibling repository checkout to build.
