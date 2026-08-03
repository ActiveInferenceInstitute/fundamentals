# Documentation review TODO

Date: 2026-08-02
Last reviewed: 2026-08-02 (DOCS-DEEP pass, comprehensive extension)

This file records findings from the repository-wide documentation review. Severity:

- Minor — typo, broken link or anchor, stale path, or formatting correction.
- Medium — stale section rewrite, documentation restructure, or missing high-value guide.
- Major — large documentation-system overhaul, new documentation site, or cross-cutting refactor.

## Minor

- [x] Replace machine-local absolute source-PDF paths with repository-portable wording and source-spine links. Files: `ISA.md`, `docs/reference/book_topic_matrix.md`, `docs/reference/source_spine.md`, `docs/topics/source_spine_and_appendices.md`. ✓ commit `10a6dfb`
- [x] Repair the dead `core.posterior` cross-reference anchor. File: `docs/cookbook.md`. ✓ commit `10a6dfb`
- [x] Repair stale API-reference paths in package guidance. Files: `src/active_inference/estimators/AGENTS.md`, `src/active_inference/visualizations/AGENTS.md`. ✓ commit `10a6dfb`
- [x] Complete the "Expected files" tables in `output/figures/` READMEs against the scripts that produce them. Files: `output/figures/chapter_01/README.md` (add `05_belief_from_stream.gif` + PNG→"PNG and GIF" header), `chapter_11..14/README.md` (add the 2-3 missing rows each). ✓ commit `ef2f473`
- [x] Repair stale API/module attributions in demo READMEs. Files: `demo/eye_saccades/README.md` (`evaluate_policy` → `core/pomdp.py`), `demo/bicycle/README.md` (module attribution for `build_multivariate_active_agent_env`, `simulate_multivariate_active_inference`). ✓ commit `ef2f473`
- [x] Add missing module rows to `src/active_inference/` subpackage README file tables. Files: `core/README.md` (8 missing modules: generalized_filtering, active_inference, pomdp, pomdp_extensions, bayesian_mechanics, appendix_math, model_comparison, noise), `estimators/README.md` (pomdp, pomdp_extensions, applications), `utils/README.md` (notebooks). ✓ commit `ef2f473`
- [x] Complete `tests/` subdirectory README source↔test tables. Files: `tests/core/README.md` (11 missing rows), `tests/estimators/README.md` (7), `tests/utils/README.md` (2), `tests/visualizations/README.md` (variational/style rows now point at real test files). ✓ commit `ef2f473`
- [x] Fix stale chapter-scope and discovery descriptions in test docs. Files: `tests/README.md` (1-10 → 1-14 twice, add `demo/` + root-level test files to the tree/table), `tests/chapters/README.md`, `tests/chapters/AGENTS.md` (CHAPTER_DIRS glob model → menu-runner discovery), `chapters/AGENTS.md` (CHAPTER_DIRS references → folder-driven discovery note). ✓ commits `ef2f473` (+ `chapters/AGENTS.md` in a later commit)
- [x] Add the missing `tests/demo/README.md`. ✓ commit `ef2f473`
- [x] Fix stale chapter-spine range in `output/data/AGENTS.md` (1-10 → 1-14) and complete the subfolder layout in `output/figures/AGENTS.md` (chapters 11-14 + demo). ✓ commit `ef2f473`

## Medium

- Findings from the three parallel family audits (chapters, extras, docs-subfolders+manuscript) land here once their consolidated reports are folded in.

## Major

- No major findings identified in the first pass; pending the family-audit consolidation.

## Open / deferred

- The manuscript directory remains an explicitly marked scaffold. Its sibling-template reference and example figure path are intentional placeholders; resolving them would require importing or reproducing a separate template contract, which is outside this repository-only pass.
- No `CITATION.cff`, `SECURITY.md`, or `CONTRIBUTING.md` files were added because equivalent citation, contribution, and license guidance already exists in `README.md`, and no repository-specific security policy was present to document accurately.
