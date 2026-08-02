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
