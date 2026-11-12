# Methods {#sec:methods}

## Manuscript Construction Method

The manuscript binds chapters, scripts, source-spine validators, rendered figures, and raw-data exports to conservative claims. Three mechanisms carry that binding:

1. **Declared variables.** `manuscript/variables.yaml` declares every token the prose uses. `scripts/render_manuscript.py` recomputes each value from tracked state (package metadata, chapter and extras directories, the GNN pin) and refuses to render when a declared value has drifted.
2. **Figure registry.** `output/figures/figure_registry.json` records, for every figure the prose cites, the label, the generated file, the producing orchestrator, the caption, and the alternative text. The renderer binds the registry alternative to the rendered image and fails on a missing file, an unregistered reference, or a registered figure no section uses.
3. **Section labels.** Each section file has exactly one H1 carrying a `{#sec:...}` label, and labels are unique, so cross-references survive reordering.

## Evidence Promotion

Future manuscript claims should be promoted in this order:

1. Identify the source file, test, generated artifact, or external reference.
2. Add or verify the project-local gate that reproduces the claim.
3. Move volatile values into declared manuscript variables when they can change across runs.
4. Add prose only after the supporting evidence path is stable.

## Template Compliance

Each section has one labelled H1, no unresolved citations, and no undeclared token. The manuscript follows the shared template conventions for Pandoc citations, `pandoc-crossref` labels, and a thin `99_references.md`; the render path itself is this repository's own `scripts/render_manuscript.py`, invoked by the `manuscript` arm of `./gate.sh`.
