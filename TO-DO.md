# Documentation review TODO

Date: 2026-08-02
Last reviewed: 2026-08-02 (DOCS-DEEP pass)

This file records findings from the repository-wide documentation review. Severity:

- Minor — typo, broken link or anchor, stale path, or formatting correction.
- Medium — stale section rewrite, documentation restructure, or missing high-value guide.
- Major — large documentation-system overhaul, new documentation site, or cross-cutting refactor.

## Minor

- [ ] Remove the remaining illustrative `../output/figures/example.png` placeholder from the manuscript syntax audit, or make its template-only status explicit in a future manuscript-template revision. Files: `manuscript/SYNTAX.md`. (Deferred: the example is intentionally non-existent template syntax, not a repository artifact.)
- [x] Replace machine-local absolute source-PDF paths with repository-portable wording and source-spine links. Files: `ISA.md`, `docs/reference/book_topic_matrix.md`, `docs/reference/source_spine.md`, `docs/topics/source_spine_and_appendices.md`. ✓ commit `10a6dfb`
- [x] Repair the dead `core.posterior` cross-reference anchor. File: `docs/cookbook.md`. ✓ commit `10a6dfb`
- [x] Repair stale API-reference paths in package guidance. Files: `src/active_inference/estimators/AGENTS.md`, `src/active_inference/visualizations/AGENTS.md`. ✓ commit `10a6dfb`

## Medium

- No additional medium findings were identified. The repository already has a documentation hub, install/usage guides, architecture notes, API references, chapter maps, cookbook, contribution instructions, citation text, and license guidance.

## Major

- No major findings were identified. The existing docs tree is coherent and has per-directory maintenance guidance; a new documentation site or heavyweight toolchain would be disproportionate to the observed gaps.

## Open / deferred

- The manuscript directory remains an explicitly marked scaffold. Its sibling-template reference and example figure path are intentional placeholders; resolving them would require importing or reproducing a separate template contract, which is outside this repository-only pass.
- No `CITATION.cff`, `SECURITY.md`, or `CONTRIBUTING.md` files were added because equivalent citation, contribution, and license guidance already exists in `README.md`, and no repository-specific security policy was present to document accurately.
