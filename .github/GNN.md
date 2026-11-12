# GNN conformance

`gnn conformance (pinned)` is defined in `.github/workflows/ci.yml` as the
`gnn-conformance` job and reads `specs/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`). The job installs that commit
into a Python 3.12 virtual environment and requires the installed
`gnn.__version__` to equal the pinned tag without its `v`.

The job runs `uv run python scripts/validate_gnn_spec.py` with
`--gnn-runtime "$RUNNER_TEMP/gnn-runtime/bin/python" --json`; it
digest-verifies the installed parser files against the pin and parses
`specs/weather_pomdp.gnn.md` through `scripts/gnn_parser_probe.py`. The pinned
`MarkdownGNNParser`, `SemanticValidator`, B-orientation check, and consistency
checker must agree with the dependency-free reader in
`active_inference.gnn_spec` on the sections, model name, variables, and
`A`/`D` matrices. The job then re-runs `tests/test_gnn_spec.py` under
`uv run pytest -q` with that runtime, so the pinned-parser case runs
instead of skipping and the declared matrices must still match
`chapters/chapter_09/example_9_1_state_inference.py`. It does not scan the
rest of the repository for `*.gnn.md`.
