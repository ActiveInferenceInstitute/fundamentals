# Documentation review TODO

Date: 2026-08-02
Last reviewed: 2026-08-02 (DOCS-DEEP pass, comprehensive extension); re-checked 2026-11-11 (Institute improvements pass 3); re-checked 2026-11-12 (estate to-do refresh at `eb64397`)

## Estate alignment pass (2026-11-12)

`specs/gnn-pin.json` now carries the canonical `gnn-importer-pin/v1` schema
(`b3f5ab5`, #29), and `.github/GNN.md` records the pin file, the `gnn-conformance`
job key and display name, and the `./gate.sh gnn` arm (`4212d58`, #31). That arm
provisions or accepts a runtime at the pinned revision, runs
`scripts/validate_gnn_spec.py --gnn-runtime` — which verifies every recorded
parser digest and requires the pinned parser and the dependency-free reader in
`active_inference.gnn_spec` to agree — and then runs `tests/test_gnn_spec.py`.
The pin itself still names `v5.0.3` at
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, and this repository still reads it
with its own `load_pin`; the estate-wide cutover to the shared reader is open,
below.

Authored education source text now exists (`eb64397`, #34): a
`## Learning objective` section in `chapters/chapter_01/README.md` and the first
definition in `manuscript/98_symbols_glossary.md`. That is one objective and one
glossary term against fourteen chapters; the education contract needs at least one
objective per unit, so this is a start, not an adoption.

`pymdp` is not a dependency of this repository: `pyproject.toml`,
`requirements.txt`, and `uv.lock` contain no `pymdp` occurrence at all, so no
`inferactively-pymdp` floor applies here.

## Manuscript and gate pass (2026-11-12)

- The manuscript no longer renders only through a sibling template checkout.
  `scripts/render_manuscript.py` validates `manuscript/variables.yaml` against
  its repository producers, binds `output/figures/figure_registry.json` to the
  cited figures, resolves citations, and renders the formats enabled in
  `manuscript/config.yaml`. It is reachable as `./gate.sh manuscript-check` and
  `./gate.sh manuscript`, and `manuscript/README.md` and `manuscript/SYNTAX.md`
  document the contract.
- `./gate.sh` is the single gate entry point; `.github/workflows/ci.yml` calls
  its arms instead of restating commands, lint runs once instead of in both
  matrix legs, and every job uses the shared `setup-python-uv` composite step.
  [`docs/gate.md`](docs/gate.md) records the arms and their measured times.
- The deferred item below is therefore resolved for the manuscript; the figure
  sample it cites is deliberately two figures, not the whole inventory.

## Institute improvements pass 3 (2026-11-11)

Documentation-completeness re-check. `docs/README.md` now indexes the
`learn_by_hand.md` page in both its tree and its "Pick the right page" table; no
other `docs/` page is unreachable. A relative-link walk over every tracked
`*.md` reports one intentional placeholder only (`manuscript/SYNTAX.md` →
`../output/figures/example.png`, already listed under "Open / deferred"), and
`grep` finds no `docs/FAQ.md` reference anywhere in the tree. Every top-level
folder with three or more tracked files already carries a `README.md`/`AGENTS.md`
pair, so no folder guide was needed.

Environment prerequisite: the bounded offline documentation gate
`PYTHONPATH=src MPLBACKEND=Agg python -m pytest -q -x tests/utils
tests/test_documentation_contracts.py` (53 tests, previously passing) could not
be re-run here — no interpreter on this machine has the cached
`numpy`/`scipy`/`matplotlib`, and the uv cache has no `scipy` wheel, so the check
was done statically (YAML/JSON parse, link walk, contract tests inspected).

This file records findings from the repository-wide documentation review. Severity:

- Minor — typo, broken link or anchor, stale path, or formatting correction.
- Medium — stale section rewrite, documentation restructure, or missing high-value guide.
- Major — large documentation-system overhaul, new documentation site, or cross-cutting refactor.

All 284 Markdown files were reviewed: the root README/AGENTS/ISA, the `docs/`
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

- **GNN–FEP cutover to the shared reader is not applied here.**
  `specs/gnn-pin.json` names `v5.0.3` at
  `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, a revision that predates
  `gnn/importers.py`, so this repository's `active_inference.gnn_spec.load_pin`
  still reads and verifies the pin itself instead of calling
  `python -m gnn.importers check --repo .`. **Blocker:** the pin bump to `v5.1.0`
  (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`) invalidates every recorded parser
  digest, so the bump, the digest re-record, `.github/GNN.md`, and `./gate.sh gnn`
  have to change together against the new runtime. **Owner:** the `fundamentals`
  maintainers. The upstream tag exists, so nothing outside this repository blocks
  it; an unavailable runtime must keep failing the arm rather than skipping it.
- **The education contract cannot be emitted for this repository.** `courses`
  names `fundamentals` a member whose pointer is `education-contract.json` and
  whose schema is `specs/education/schema.json` (`courses`
  `specs/education/contract.json`, member `members[2]`, read at `f1e5af62`);
  neither file exists here. `$defs.unit.properties.objectives` is `minItems: 1`
  and `objectives` is required, so a unit with no authored objective cannot be
  written at all. **Missing artefact:** authored learning objectives for
  chapters 2–14 in their chapter sources — only
  `chapters/chapter_01/README.md` carries a `## Learning objective` heading — plus
  further authored glossary definitions in `manuscript/98_symbols_glossary.md`,
  which currently defines one term (`## Generative model`). **Owner:** the chapter
  authors and `fundamentals` maintainers; the lead `courses` contract cannot
  supply them, because its content covers the Institute's courses, not this
  book's chapters. No bundle can be emitted without them, and recasting a chapter
  description as an objective would invent the statement the contract has to
  quote. The vendored contract and the full triage live in open PR #33, which
  stays open rather than record a partial adoption.
- Extras-family M3 (boilerplate consolidation): the 57 full-template topic
  READMEs share identical Scripts/Outputs sections that drifted (m1/m4 fixed
  here). A durable fix would generate those sections from the
  `active_inference.extra_topics` registry (a small doc-gen step in
  `scripts/`); deferred because it is a code change, not a docs edit, and the
  two concrete drift symptoms are now fixed and validator-checked.
- ~~The manuscript directory remains an explicitly marked scaffold.~~ Resolved
  2026-11-12: the manuscript has a project-local render path, declared
  variables, and a validated figure registry. The prose itself is still a
  scaffold — it remains an evidence map, not a publication draft.
- No `CITATION.cff`, `SECURITY.md`, or `CONTRIBUTING.md` files were added
  because equivalent citation, contribution, and license guidance already
  exists in `README.md`, and no repository-specific security policy was
  present to document accurately.
