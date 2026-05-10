# `docs/statistics/` — agent guide

One Markdown file per family of statistical tools shipped in
`active_inference.core.diagnostics`. Pages here are reference
material: precise definitions, formulas, API tables, and pitfalls.

## File contract

Each statistics page must include, in order:

1. **H1 title** matching the filename.
2. **Definition** — the mathematical statement, in our own prose.
3. **Closed-form variant** — when one exists (commonly Gaussian).
4. **API** — table of public functions / classes with signatures.
5. **Tests that pin it down** — pointers to the relevant test classes.
6. **Pitfalls** — numerical / interpretation gotchas.
7. **See also** — relative links into `topics/` and `reference/`.

## When to add a page

Add a new page only when `core.diagnostics` (or, occasionally,
`core.distributions`) gains a *family* of related tools. A single new
helper that fits inside an existing family extends the existing page.

## Avoid

- Reproducing book passages — write definitions in our own prose.
- Long mathematical derivations — link out instead.
- API tables that don't match `__all__` — verify before merging.
