# Manuscript - fundamentals

This directory is a template-format manuscript scaffold for:

**Fundamentals of Active Inference: Python Companion**

An open-source Python companion to Fundamentals of Active Inference, implementing algorithms, figures, raw-data sidecars, and validators aligned to the inspected source spine.

## File Inventory

- `config.yaml`
- `preamble.md`
- `references.bib`
- `00_abstract.md`
- `01_introduction.md`
- `02_system_context.md`
- `03_methods.md`
- `04_artifacts_and_evidence.md`
- `05_reproducibility.md`
- `06_limitations_and_next_steps.md`
- `S01_source_surface.md`
- `98_symbols_glossary.md`
- `99_references.md`
- `AGENTS.md`
- `README.md`
- `SYNTAX.md`

## Source Surfaces

| Surface | Role |
|---|---|
| `chapters/` | Source directory to inspect before turning prose into claims. |
| `src/active_inference/` | Source directory to inspect before turning prose into claims. |
| `scripts/` | Source directory to inspect before turning prose into claims. |
| `tests/` | Source directory to inspect before turning prose into claims. |
| `docs/` | Source directory to inspect before turning prose into claims. |
| `output/` | Source directory to inspect before turning prose into claims. |

## Verification

From the sibling template checkout, after `link-projects` has synced the sidecar:

```bash
uv run python -m infrastructure.orchestration link-projects
uv run python -m infrastructure.validation.cli markdown projects/working/fundamentals/manuscript/
```

Render only after replacing scaffold prose with project-bound evidence and checking any project-local gates documented in the repository root.
