# Manuscript Agent Notes - fundamentals

This directory follows the docxology/template manuscript contract:

- `00_` through `09_` files are main sections.
- `S01_` files are supplemental material.
- `98_` and `99_` files are back matter.
- Every manuscript section file starts with one H1 and a stable `{#sec:...}` label.
- Citations must use Pandoc syntax and resolve in `references.bib`.
- Generated numbers belong behind `{{TOKEN}}` variables, not hard-coded prose.

## Editing Rules

- Treat this scaffold as an outline until project-specific evidence is bound.
- Do not fabricate results, benchmark numbers, citations, DOIs, or release claims.
- Keep project-specific computation in source modules and scripts; keep manuscript files as prose and evidence maps.
- Prefer explicit paths to source surfaces when describing evidence.
- If adding figures, write them under `../output/figures/` and reference them with Pandoc-crossref labels.

## Current Scope

An open-source Python companion to Fundamentals of Active Inference, implementing algorithms, figures, raw-data sidecars, and validators aligned to the inspected source spine.

Evidence boundary: The textbook itself is not open source; manuscript prose must describe the companion implementation and validation without reproducing protected book content.
