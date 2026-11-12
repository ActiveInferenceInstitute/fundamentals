# Limitations and Next Steps {#sec:limitations_next_steps}

## Current Limitations

- The manuscript is a scaffold, not a completed publication draft.
- It contains no external citations yet; `references.bib` is intentionally a placeholder, so `bibliography.fail_on_missing` currently passes only because no section cites anything.
- It cites a two-figure sample of the generated media rather than the full figure inventory. Every cited figure is registered and bound to its producer; the rest remain documented in `output/figures/`.
- It avoids quantitative claims that are not bound to tests, source ledgers, or generated outputs.
- The PDF path needs `pandoc` and `xelatex` on `PATH`; the `manuscript` gate arm names whichever is absent instead of producing partial output.

## Next Steps

1. Replace scaffold prose with source-backed sections as the book's worked examples are audited into tests.
2. Declare further manuscript variables when new prose needs run-derived numbers; the producer in `scripts/render_manuscript.py` must compute them from tracked state.
3. Populate `references.bib` with verified entries before adding citation markers.
4. Add figures only after their producers write stable files under `output/figures/`, and register each one with a caption and alternative text.
5. Inspect the rendered HTML and PDF before release work; the render receipt records what was built.

## Boundary Note

The textbook itself is not open source; manuscript prose must describe the companion implementation and validation without reproducing protected book content.
