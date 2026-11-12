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
