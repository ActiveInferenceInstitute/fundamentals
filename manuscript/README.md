# Manuscript - fundamentals

This directory holds the manuscript for:

**Fundamentals of Active Inference: Python Companion**

An open-source Python companion to Fundamentals of Active Inference,
implementing algorithms, figures, raw-data sidecars, and validators aligned to
the inspected source spine.

The directory previously rendered only through a sibling template checkout. It
now carries its own render path, its own declared variables, and its own figure
registry, so it builds from this checkout alone.

## File Inventory

- `config.yaml` - title, authors, bibliography, and enabled render formats
- `variables.yaml` - declared `{{TOKEN}}` values, each with a repository producer
- `preamble.md` - fenced LaTeX passed to Pandoc as `--include-in-header`
- `references.bib` - bibliography (a placeholder until verified entries are added)
- `00_abstract.md` ... `06_limitations_and_next_steps.md` - main sections
- `S01_source_surface.md` - supplemental source-surface table
- `98_symbols_glossary.md`, `99_references.md` - back matter
- `SYNTAX.md` - the section, citation, figure, variable, and render contract
- `AGENTS.md` - agent guidance for this directory

Figures are registered in
[`../output/figures/figure_registry.json`](../output/figures/figure_registry.json).

## Source Surfaces

The canonical source-surfaces table lives in
[`S01_source_surface.md`](S01_source_surface.md); see there for the surfaces to
inspect before turning prose into claims.

## Verification and Render

```bash
# validate variables, figures, citations, and section labels (no Pandoc needed)
./gate.sh manuscript-check

# validate and render every format enabled in config.yaml
./gate.sh manuscript
```

Both commands run `scripts/render_manuscript.py`; the second writes the HTML, the
PDF, and a render receipt under `docs/_build/manuscript/`. The repository gate is
documented in [`../docs/gate.md`](../docs/gate.md).
