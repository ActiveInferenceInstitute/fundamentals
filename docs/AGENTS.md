# `docs/` — agent guide

Reference documentation organized by *audience and purpose*. The intended
reader of any page should be obvious from the folder it lives in.

## Folder boundaries

| Folder | Audience | Lives there if it… |
|---|---|---|
| `chapters/` | Reader following the book chapter-by-chapter | Maps a book chapter to chapter orchestrator scripts. |
| `topics/` | Reader who wants a deep dive on a concept (not tied to one chapter) | Walks through a single idea with code snippets and pointers. |
| `statistics/` | Implementer needing a precise definition of a statistical tool | Documents one symbol or family from `core.diagnostics`. |
| `reference/` | Library user looking up an API | Lists every public symbol in one src subpackage. |

The root files (`architecture.md`, `notation.md`) cover cross-cutting
material and are linked from every subfolder's `README.md`.

## Naming conventions

- Files use lowercase with underscores: `bayesian_inference.md`,
  `effective_sample_size.md`.
- One **H1** per file; the filename should appear (or be paraphrased) in
  the H1.
- Tables for symbol enumeration; prose for conceptual explanations.
- All inline code references use backticks; all cross-doc links use
  relative paths so GitHub renders them.

## When to add a new page

| You want to add… | Drop it in… |
|---|---|
| A new chapter overview | `chapters/chapter_<N>.md` |
| A new concept that spans multiple chapters | `topics/<topic>.md` |
| A new statistical tool that ships in `core.diagnostics` | `statistics/<tool>.md` |
| Documentation for a new public symbol or subpackage | extend the matching `reference/<sub>.md` |

If you can't figure out the right folder, default to `topics/` and add a
note in the file header explaining the placement.

## What never lives here

- Runnable scripts → `chapters/` or `scripts/` at the repo root.
- Generated content (figures, GIFs) → `output/`.
- Mathematical derivations more than a few lines.

## Update policy

- Every PR that adds a public symbol must also update either
  `reference/<sub>.md` or `statistics/<tool>.md`.
- Every PR that adds a chapter script must also update
  `chapters/chapter_<N>.md`.
- Cross-cutting structural changes (a new subpackage, a new layer) must
  update `architecture.md` and add an entry to `notation.md` for any new
  symbol.
