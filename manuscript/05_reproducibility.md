# Reproducibility {#sec:reproducibility}

## Local Commands

The manuscript renders from this checkout; no sibling repository is required.

```bash
# validate every manuscript surface without needing a TeX installation
./gate.sh manuscript-check

# render every format enabled in manuscript/config.yaml
./gate.sh manuscript
```

The renderer writes `docs/_build/manuscript/<slug>.html`, the matching PDF when
`render.formats.pdf` is enabled, and a receipt at
`docs/_build/manuscript/render-receipt.json` recording the checked sections,
resolved variables, figures, citations, pandoc version, and each output's SHA-256.

## Figures

Figures are generated, not committed as a second copy:

```bash
uv run python chapters/chapter_01/03_bayes_intuition.py --save
uv run python chapters/chapter_01/04_inverse_problem.py --save
uv run python scripts/validate_rendered_figures.py --root output/figures
```

## Reproducibility Contract

- Do not cite results that cannot be regenerated or directly traced.
- Keep generated outputs under `output/` and manuscript source under `manuscript/`.
- Keep private data, credentials, and unpublished sensitive details out of the manuscript.
- Record exact verification commands before marking this manuscript publication-ready.
