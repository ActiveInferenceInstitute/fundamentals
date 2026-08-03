# Documentation deep-review log — 2026-08-02

Repository: `ActiveInferenceInstitute/fundamentals`
Branch at start: `main`
Starting commit: `645c5b4794e623364e2b97140d714975571c9320`
Working tree at start: clean
Documentation inventory: 274 Markdown files, including the root README, `docs/`, manuscript scaffold, per-directory `AGENTS.md` guidance, chapter/extras READMEs, and generated-output documentation.

## Phase 0 — preflight

- Fetched `origin` and verified `main` was already up to date.
- Confirmed the default branch from `origin/HEAD` is `main`.
- Confirmed the repository uses Python, `pyproject.toml`, `uv.lock`, pytest, and ruff-oriented guidance.
- Confirmed no pre-existing top-level TODO/review log file.

## Phase 1 — review results

- Minor findings: 7 actionable documentation corrections (four machine-local absolute PDF paths, one dead anchor, two stale API-reference paths); one intentional manuscript template placeholder remains deferred.
- Medium findings: 0. The repository already provides install, usage, architecture, notation, cookbook, reading-order, API-reference, chapter, topic, statistics, contribution, citation, and license documentation.
- Major findings: 0. The existing documentation organization is coherent and a new documentation site/toolchain would not be proportionate.
- Link/anchor scan covered all 274 Markdown files. After accounting for valid directory links, the only remaining path warnings are intentional manuscript-template references.
- Quantitative README claims checked against the repository: 71 extras topics, 57 with simulation and interactive wrappers, 23 with animation wrappers; all matched. Chapter inventories were also checked against their directories.

## Phase 2 — scope

- Created `TO-DO.md` with Minor / Medium / Major sections, completed items, and deferred work.

## Phase 3 — implementation

- Removed public-facing machine-local source-PDF paths and linked to the repository-relative source-spine ledger.
- Corrected the cookbook link to the actual `core.posterior` heading slug.
- Corrected stale `docs/estimators.md` and `docs/visualizations.md` references to the existing `docs/reference/` pages.

## Phase 4 — verification

Passing checks before the documentation edits:

- `uv run python scripts/validate_orchestrator_contracts.py` — 319 scripts, 6 existing duplicate-stem warnings.
- `uv run python scripts/validate_orchestrator_provenance.py`.
- `uv run python scripts/validate_book_topic_coverage.py` — 71 entries.
- `uv run python scripts/validate_source_spine.py --require-pdf`.
- `uv run python scripts/validate_raw_data_exports.py --root output/data` — 125 pairs.
- `uv run python scripts/validate_rendered_figures.py --root output/figures` — 125 artifacts.
- `uv run python scripts/validate_notebook_exports.py`.

The final documentation scan and git diff/status are run after the edits are committed.

## Comprehensive extension (same day) — full-family audit + implementation

The initial pass covered the top-level docs only. This extension reviewed ALL
274 Markdown files across families, using three parallel review-only
subagents (chapters family: 46 files; extras family: 72 files; docs
subfolders + manuscript: 54 files), plus a personal audit of the generated
output/ docs, tests/ docs, src/ subpackage READMEs, demo/ docs, and
scripts/ docs.

Findings implemented (per family):

- Chapters family: 6 Minor + 7 Medium findings — stale Lines columns
  (orchestrator_workflows-wrapped scripts), stale smoke-test node IDs,
  H1/`## Scripts` style drift, Rules-only ch11-14 AGENTS, skeletal ch08/11-14
  READMEs (expanded with concept-map links + executed programmatic snippets),
  docs/chapters contract relaxed to the two real page variants, missing
  forward pointer added to chapter_10 page.
- Extras family: 3 Medium + 4 Minor findings — root README Appendix A map
  aligned with the registry (6 rows corrected), 4 topic README book-section
  omissions fixed (+B.12/+12.7/+11.2.8/+12.4.1), appendix_math_fundamentals
  section range tightened, 142 artifact-path claims normalized to trailing
  slash, "Run any non-interactive script with --save" phrasing across 58
  topic READMEs. Boilerplate doc-gen (M3) deferred as a code change.
- Docs subfolders + manuscript: 7 Minor + 1 Medium — typo, audience count,
  duplicate source-PDF sentences, missing demos.md in docs map, missing
  pc_multivariate_linear_fixed_point API row (signature-verified), architecture
  diagram completions (notebooks.py, orchestrator_workflows.py), manuscript
  source-surfaces table de-duplicated (canonical in S01).
- Personal audit: output/figures README expected-file tables completed for
  chapters 1/11-14, tests/ README source↔test tables completed, tests/demo/
  README created, demo README API attributions corrected, output/ AGENTS
  chapter-spine ranges updated, chapters/AGENTS.md CHAPTER_DIRS removed.

Every added programmatic snippet was executed against the real API; every
line-count edit was verified with wc -l; every validators still passes
(orchestrator contracts/provenance, book-topic coverage, source spine, raw
data, rendered figures, notebook exports).
