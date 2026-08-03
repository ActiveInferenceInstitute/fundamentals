# Documentation review TODO

Date: 2026-08-02
Last reviewed: 2026-08-02 (DOCS-DEEP pass, comprehensive extension)

This file records findings from the repository-wide documentation review. Severity:

- Minor — typo, broken link or anchor, stale path, or formatting correction.
- Medium — stale section rewrite, documentation restructure, or missing high-value guide.
- Major — large documentation-system overhaul, new documentation site, or cross-cutting refactor.

All 274 Markdown files were reviewed: the root README/AGENTS/ISA, the `docs/`
hub (root pages + chapters/ topics/ statistics/ reference/), the manuscript
scaffold, per-directory AGENTS/README pairs, and generated-output docs. The
family-level audit was fanned out to three parallel review subagents (chapters,
extras, docs-subfolders+manuscript), whose findings are folded in below.

## Minor

- [x] Replace machine-local absolute source-PDF paths with repository-portable wording and source-spine links. Files: `ISA.md`, `docs/reference/book_topic_matrix.md`, `docs/reference/source_spine.md`, `docs/topics/source_spine_and_appendices.md`. ✓ commit `10a6dfb`
- [x] Repair the dead `core.posterior` cross-reference anchor. File: `docs/cookbook.md`. ✓ commit `10a6dfb`
- [x] Repair stale API-reference paths in package guidance. Files: `src/active_inference/estimators/AGENTS.md`, `src/active_inference/visualizations/AGENTS.md`. ✓ commit `10a6dfb`
- [x] Complete the "Expected files" tables in `output/figures/` READMEs against the scripts that produce them. Files: `output/figures/chapter_01/README.md`, `chapter_11..14/README.md`. ✓ commit `ef2f473`
- [x] Repair stale API/module attributions in demo READMEs. Files: `demo/eye_saccades/README.md`, `demo/bicycle/README.md`. ✓ commit `ef2f473`
- [x] Add missing module rows to `src/active_inference/` subpackage README file tables. Files: `core/README.md` (8 modules), `estimators/README.md` (3), `utils/README.md` (1). ✓ commit `ef2f473`
- [x] Complete `tests/` subdirectory README source↔test tables (11+7+2+2 rows). Files: `tests/core|estimators|utils|visualizations/README.md`. ✓ commit `ef2f473`
- [x] Fix stale chapter-scope and discovery descriptions in test docs (`1–10` → `1–14`, CHAPTER_DIRS glob model → menu-runner discovery). Files: `tests/README.md`, `tests/chapters/README.md`, `tests/chapters/AGENTS.md`, `chapters/AGENTS.md`. ✓ commits `ef2f473`, `db3fc4e`
- [x] Add the missing `tests/demo/README.md`. ✓ commit `ef2f473`
- [x] Fix stale chapter-spine range in `output/data/AGENTS.md` and complete `output/figures/AGENTS.md` subfolder layout. ✓ commit `ef2f473`
- [x] Chapters family (subagent): stale `Lines` columns in AGENTS.md script tables. Files: `chapter_03` (example_3_5/3_7 + 8 animations), `chapter_04` (example_4_2), `chapter_06` (example_6_7), `chapter_07` (example_7_5 + animation), `chapter_09` (example_9_3). ✓ commit `c01c6b1`
- [x] Chapters family (subagent): H1 hyphen→em-dash style and missing `## Scripts` heading. Files: `chapters/chapter_08/README.md`, `chapter_11..14/README.md` + AGENTS.md. ✓ commit `c01c6b1`
- [x] Extras family (subagent): root `extras/README.md` Appendix A section map aligned with the registry (history A.1–A.1.7, future A.2/A.2.5, lineage topics A.2.1–A.2.4). ✓ commit `7b080fe`
- [x] Extras family (subagent): omitted book sections added (factor_graphs +B.12, hierarchical_message_passing +12.7, tree_policy_search +11.2.8, variational_message_passing +12.4.1; appendix_math_fundamentals range tightened to C.1–C.8, C.10, C.12, C.13). ✓ commit `7b080fe`
- [x] Extras family (subagent): normalized 142 no-trailing-slash artifact-path claims and "Run any script with `--save`" → "Run any non-interactive script with `--save`" across 58 topic READMEs. ✓ commit `7b080fe`
- [x] Docs family (subagent): `docs/statistics/entropy.md` typo ("exit" → "exist"); `docs/reading_order.md` "four audiences" → six; duplicate source-PDF sentence removed from `docs/reference/source_spine.md` and `docs/topics/source_spine_and_appendices.md`; `docs/README.md` reference tree map now lists `demos.md`. ✓ commit `8e53f09`
- [x] Docs family (subagent): `docs/reference/estimators.md` gains the missing `pc_multivariate_linear_fixed_point` API row (signature verified). ✓ commit `8e53f09`
- [x] Docs family (subagent): `docs/architecture.md` layer diagram now lists `utils/notebooks.py` and `orchestrator_workflows.py`. ✓ commit `8e53f09`

## Medium

- [x] Chapters family (subagent): stale smoke-test node IDs (`test_chapter_6/7/8/9/10_*`) replaced with the real single parametrized `test_chapter_script_runs_and_exports_raw_data`. Files: `chapters/chapter_06..10/AGENTS.md`. ✓ commit `c01c6b1`
- [x] Chapters family (subagent): Rules-only ch11–14 AGENTS.md expanded to the shared template (Scripts-with-Lines, Library Usage, Smoke Tests). ✓ commit `c01c6b1`
- [x] Chapters family (subagent): skeletal ch08/11–14 READMEs expanded with concept-map links and verified programmatic-usage snippets (every snippet executed against the real API). ✓ commit `c01c6b1`
- [x] Chapters family (subagent): `docs/chapters/` file contract relaxed to describe the two real page variants (5-step Recipe for Ch.1–5; model/architecture section for Ch.6+) and imports via table or library-surface section; `docs/chapters/chapter_10.md` gains the missing "Where the book takes this next". Files: `docs/chapters/README.md`, `docs/chapters/AGENTS.md`, `docs/chapters/chapter_10.md`. ✓ commit `8e53f09`
- [x] Docs family (subagent): manuscript source-surfaces table de-duplicated — canonical table stays in `S01_source_surface.md`; `manuscript/README.md` and `04_artifacts_and_evidence.md` cross-reference it. ✓ commit `8e53f09`

## Major

- None identified. The existing documentation system is coherent; a new docs
  site or toolchain would be disproportionate to the observed gaps.

## Open / deferred

- Extras-family M3 (boilerplate consolidation): the 57 full-template topic
  READMEs share identical Scripts/Outputs sections that drifted (m1/m4 fixed
  here). A durable fix would generate those sections from the
  `active_inference.extra_topics` registry (a small doc-gen step in
  `scripts/`); deferred because it is a code change, not a docs edit, and the
  two concrete drift symptoms are now fixed and validator-checked.
- The manuscript directory remains an explicitly marked scaffold. Its
  sibling-template reference and example figure path are intentional
  placeholders; resolving them would require importing or reproducing a
  separate template contract, which is outside this repository-only pass.
- No `CITATION.cff`, `SECURITY.md`, or `CONTRIBUTING.md` files were added
  because equivalent citation, contribution, and license guidance already
  exists in `README.md`, and no repository-specific security policy was
  present to document accurately.
