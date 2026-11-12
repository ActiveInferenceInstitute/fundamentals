# The repository gate

`./gate.sh` is the single entry point for this repository's checks. Each arm is
separately runnable, and each arm body is the command a CI job calls, so a job
never restates a command that runs elsewhere.

```bash
./gate.sh --list          # print the arms
./gate.sh fast            # what the CI test job runs
./gate.sh all             # every arm, in order
```

## Arms

| Arm | What it runs | CI job |
|---|---|---|
| `lint` | `uv run ruff check .` | `lint` |
| `fast` | `uv run python -m pytest -q tests/core tests/estimators tests/utils tests/visualizations tests/web tests/test_*.py` | `tests (py3.11)`, `tests (py3.14)` |
| `full` | `uv run python -m pytest -q` | local only |
| `docs` | `uv run python -m pytest -q tests/test_documentation_contracts.py tests/test_method_documentation_contracts.py` | local fast path (a strict subset of `fast`) |
| `contract` | `uv run python scripts/check_education_contract.py` | local only; migration blocked on authored content |
| `validators` | the orchestrator-contract, provenance, book-topic, source-spine, and rendered-figure validators | `repository validators` |
| `manuscript-check` | `uv run python scripts/render_manuscript.py --check` | `repository validators` |
| `manuscript` | `uv run python scripts/render_manuscript.py` | local only (needs `pandoc` and `xelatex`) |
| `gnn` | provision the pinned runtime, `validate_gnn_spec.py`, then `tests/test_gnn_spec.py` with that runtime | `gnn conformance (pinned)` |
| `all` | every arm above, in that order | — |

There is no `types` arm: `pyproject.toml` configures no type checker, and adding
one is its own reviewed migration rather than a side effect of this gate.

## CI shape

`.github/workflows/ci.yml` runs four jobs: `lint` (once, not once per
interpreter), `tests` on Python 3.11 and 3.14, `repository validators` (which also
runs `manuscript-check`), and `gnn conformance (pinned)`. Every job installs `uv`
and its interpreter through the shared composite step
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`,
installs the locked environment once with `uv sync --locked`, and then calls one
arm. The gate exports `MPLBACKEND=Agg`, so the workflow does not repeat it.

The `gnn` arm provisions the runtime for the commit named in `specs/gnn-pin.json`
under `$GATE_CACHE/gnn-runtime-<commit>` and refuses a runtime whose
`gnn.__version__` differs from the pinned tag. A pin bump therefore provisions a
fresh runtime instead of reusing a stale one, and CI keeps the cache on the runner
with `GATE_CACHE=${{ runner.temp }}/gate-cache`. Set `GNN_RUNTIME_PYTHON` to use
an existing interpreter instead of provisioning.

## Education migration — blocked on authored content

The [Education workflow contract](https://github.com/ActiveInferenceInstitute/courses/blob/f1e5af62dc2581b9d4c09746c8b7c4fa09dc5e1b/docs/workflows/education.md)
requires a chapter bundle and glossary transcription.
[`education-contract.json`](../education-contract.json) points to
[`specs/education/fundamentals.json`](../specs/education/fundamentals.json).
The draft transcribes chapter README titles verbatim and records each source path.
Unit IDs are member-prefixed to avoid cross-member collisions.

**Do not merge this migration yet.** The chapter READMEs describe coverage and
scripts but do not supply explicit learning objectives. Their `objectives`
arrays remain empty rather than recasting descriptions as authored objectives;
the unchanged schema rejects these arrays with `EC006`.
[`manuscript/98_symbols_glossary.md`](../manuscript/98_symbols_glossary.md)
says “This placeholder is reserved for project terminology and API symbols.”
It contains no definitions to transcribe, so `glossary` remains empty.
Schema acceptance of an empty glossary would not establish completion of the
requested migration. Prerequisites and edges are empty: chapter numbering and
coverage references are not authored hard prerequisites.

To unblock: authors must supply chapter learning objectives and definitions in
the required glossary source; then transcribe those statements verbatim into
the bundle and rerun the checks. No schema relaxation or substitute glossary
is part of this migration.

The schema and `vendor/education_contract/{schema,rules,registry}.py`
are byte-identical copies from `ActiveInferenceInstitute/courses` commit
`f1e5af62dc2581b9d4c09746c8b7c4fa09dc5e1b`. The local adapter passes an explicit
schema path to the vendored `check_member`; it does not reimplement the rules.
The local arm checks the pointer digest, schema, and declared bundles. Only a
comparison with the lead can establish that the vendored schema is still current.
These unchanged modules are by the Active Inference Institute, licensed under
CC BY 4.0; the upstream notice is retained at
[`vendor/education_contract/LICENSE`](../vendor/education_contract/LICENSE).
Upstream modules live outside the local library/chapter/script trees; their
source and documentation remain unchanged rather than adopting local style rules.

```bash
./gate.sh contract
# From this checkout, read the sibling lead without writing bytecode there:
python -B ../courses/software/scripts/check_education_contract.py \
  --schema ../courses/specs/education/schema.json specs/education/fundamentals.json
./gate.sh docs
./gate.sh fast
./gate.sh full
```

Local migration verification: the lead checker reported
`FAILED (14 issue(s))`, all `EC006` on empty objective lists.
The member `contract` arm reported `diverged -- 14 contract violation(s)`
and exited nonzero in 0.111 s measured around the command.
After documentation integration fixes, `docs` and `fast` passed.
The full arm was run before those fixes and failed: its summary recorded
190 subprocess-launch errors (`Errno 62: Too many levels of symbolic links`
at `.venv/bin/python`) and two documentation failures. It was not rerun
after the documentation corrections; full-suite verification remains open.


## Measured wall time

One run per arm on the development host (macOS, arm64), before and after this
gate landed. "Before" is the command as CI ran it, invoked directly; "after" is
the same command invoked through the arm. Warm `uv`, Ruff, and matplotlib caches
in both columns except where noted, and the runs were sequential — no two arms
overlapped. An arm that needs more than about ten minutes is not measured to
completion here; the `full` row records its one completed run of the raw command
and states plainly that the arm wrapper was only exercised with a narrowed
selection.

| Arm | Before | After | Note |
|---|---|---|---|
| `lint` | 0.05 s | 0.12 s | Ruff cache warm in both; the arm adds a shell wrapper |
| `fast` | 41.0 s | 41.0 s | 66.6 s on the first run, which builds the matplotlib font cache |
| `docs` | 1.02 s | 0.93 s | 19 tests |
| `validators` | 2.91 s | 2.81 s | 52 tracked artifacts |
| `manuscript-check` | — | 0.08 s | new arm; did not exist before |
| `manuscript` | — | 4.13 s | new arm; HTML and PDF in one run |
| `gnn` | 5.9 s | 2.3 s | 5.9 s was the validator (3.4 s) plus the pinned test (2.4 s) measured separately with a cold `gnn` import; provisioning adds 1.6 s on a cache miss |
| `full` | 650.0 s | not run | the raw command completed once (exit 0) and exceeds the ten-minute arm bound; the wrapper was exercised with `PYTEST_ADDOPTS="-q tests/core/test_pomdp.py" ./gate.sh full` → `gate: full ok (1.262s)` |

A repeat pass on the same host, taken immediately after a full-suite run,
measured `docs` 2.06 s, `validators` 5.75 s, `gnn` 4.44 s, `manuscript` 7.88 s,
and `fast` 110.3 s. The host is shared, so treat each value as roughly ±2× and
read the table as a record of what an arm costs, not as a speed comparison.

What these numbers support, and what they do not:

- Every arm runs the command that ran before, so no arm is faster because of the
  gate. The differences that appear are cache warmth plus the wrapper; the
  table is a record of what the gate costs, not a speedup claim.
- The work this change actually removes is at job level: lint runs once instead of
  in both matrix legs, each job's setup is one composite step instead of
  `setup-uv` plus `uv python install`, and the GNN provisioning exists once
  instead of in an inline workflow copy.
- GitHub Actions wall time for those jobs cannot be measured from here: the
  organization's Actions billing is locked, so CI has not run. This page does not
  claim a CI speedup it did not observe; the local arms are the gate.
- `docs` is a strict subset of `fast`. It exists so a documentation edit can be
  checked in about a second, and CI does not run it in addition to `fast`.

## Adding an arm

1. Add a function to `gate.sh` whose body is the literal command, and add the
   name to `ARMS`.
2. Point the CI job at `./gate.sh <arm>` instead of writing the command in the
   workflow.
3. Measure it, then add a row to the arm table and the measured-time table above.
4. Run the arm itself and `./gate.sh --list`; both must be green.
