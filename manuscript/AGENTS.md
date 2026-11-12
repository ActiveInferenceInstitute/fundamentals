# Manuscript Agent Notes - fundamentals

This directory follows the docxology/template manuscript contract and renders
from this checkout.

- `00_` through `89_` files are main sections.
- `S01_` files are supplemental material.
- `90_` through `99_` files are back matter; `99_references.md` stays last.
- Every manuscript section file starts with one H1 and a stable `{#sec:...}` label.
- Citations must use Pandoc syntax and resolve in `references.bib`.
- Generated numbers belong behind `{{TOKEN}}` variables declared in
  `variables.yaml`, not hard-coded prose.
- Figures are registered in `../output/figures/figure_registry.json`, which owns
  the caption and the alternative text.

## Editing Rules

- Treat this scaffold as an outline until project-specific evidence is bound.
- Do not fabricate results, benchmark numbers, citations, DOIs, or release claims.
- Keep project-specific computation in source modules and scripts; keep manuscript files as prose and evidence maps.
- Prefer explicit paths to source surfaces when describing evidence.
- If adding figures, write them under `../output/figures/`, register them, and
  cite the registry label from a `::: {#fig:...}` block.
- If adding a `{{TOKEN}}`, add its producer to `compute_variables()` in
  `scripts/render_manuscript.py` and declare it in `variables.yaml`.
- Run `./gate.sh manuscript-check` after every edit, and `./gate.sh manuscript`
  before claiming the manuscript builds.

## Current Scope

An open-source Python companion to Fundamentals of Active Inference, implementing algorithms, figures, raw-data sidecars, and validators aligned to the inspected source spine.

Evidence boundary: The textbook itself is not open source; manuscript prose must describe the companion implementation and validation without reproducing protected book content.
