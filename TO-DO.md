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

Source correction: the pass-3 `../output/figures/example.png` placeholder
reference is superseded. `manuscript/SYNTAX.md` now demonstrates
`../output/figures/chapter_01/03_bayes_three_panel.png`. No placeholder artifact
is required. This correction does not claim a new link-walk run.

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

- **GNN–FEP shared-reader migration remains open.** Preserve the deliberate
  v5.0.3 pin and its parser digests until maintainers select a producer-qualified
  replacement. The reader migration, reviewed pin/digests, `.github/GNN.md`,
  gate behavior, and conformance evidence must move together. A newer tag's
  existence alone is not admission. Unavailable or unverified runtime evidence
  must continue to fail qualification.
- **Education publication remains content-blocked.** Chapter authors must
  supply authored learning objectives for chapters 2–14 and the corresponding
  unit/source mapping before an education bundle can quote them. Chapter 1's
  existing objective is retained. Do not synthesize objectives from chapter
  descriptions. Further glossary authorship is a content decision, not an
  invented schema minimum. Maintainers own integration against the selected
  courses contract/schema after those sources are admitted.
- Extras-family M3 uses `scripts/generate_extra_readme_sections.py` for all
  58 owned Scripts/Outputs pairs, not 57. The 13 bespoke READMEs remain
  excluded. Use `--write` only under documentation custody, then require a
  clean `--check`. Generated output declarations do not certify that teaching
  scripts or scientific artifacts have been executed.
- ~~The manuscript directory remains an explicitly marked scaffold.~~ Resolved
  2026-11-12: the manuscript has a project-local render path, declared
  variables, and a validated figure registry. The prose itself is still a
  scaffold — it remains an evidence map, not a publication draft.
- No `CITATION.cff`, `SECURITY.md`, or `CONTRIBUTING.md` files were added
  because equivalent citation, contribution, and license guidance already
  exists in `README.md`, and no repository-specific security policy was
  present to document accurately.

## Engineering contract corrections

- Predictive-coding convergence requires energy and recognition-gradient
  convergence; bounded-support VFE preserves infinite divergence where required.
- Message-passing traces are tuples of normalized messages, including
  heterogeneous cardinalities; the square chapter-12 caller stacks explicitly.
- GNN validation binds one selected document observation without weakening
  runtime provenance. Manuscript rendering requires the cross-reference resolver.
- HTTP execution admission, demo navigation, and operation-local immutable
  discovery share owner-qualified script identities.
- Focused controls cover tensor-axis/operator mistakes, invalid raw-data
  sidecars, and cross-orchestrator imports. Qualification is separate from
  scientific promotion, broad coverage claims, and hosted acceptance.
