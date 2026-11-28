# GNN conformance

`gnn conformance (pinned)` is the display name of the `gnn-conformance` job in
`.github/workflows/ci.yml`. It reads `specs/gnn-pin.json` — the canonical pin
file, schema `gnn-importer-pin/v1` — which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.2.0`
(`4a7cccce6d4d7cb483f00323ec5cdd79300f3464`). The job installs that commit
into a Python 3.12 virtual environment and requires the installed
`gnn.__version__` to equal the pinned tag without its `v`.

The job and the local arm are the same commands: the job installs the locked
environment, then runs `./gate.sh gnn`, which provisions that pinned runtime
under `$GATE_CACHE/gnn-runtime-<commit>`, runs the shared reader
`<runtime>/bin/python -m gnn.importers check --repo . --require-pointer --workflow .github/workflows/ci.yml --job gnn-conformance`
(pin discovery and validation, and this pointer), runs
`uv run python scripts/validate_gnn_spec.py --gnn-runtime <runtime> --verify-tag-remote --json`,
and re-runs `tests/test_gnn_spec.py` with `GNN_RUNTIME_PYTHON` set to the same
interpreter. Setup in every job goes through
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`,
which installs the pinned `uv` and the requested CPython.

The validator digest-verifies the installed parser files against the pin and
parses `specs/weather_pomdp.gnn.md` through `scripts/gnn_parser_probe.py`. The
pinned `MarkdownGNNParser`, `SemanticValidator`, B-orientation check, and
consistency checker must agree with the dependency-free reader in
`active_inference.gnn_spec` on the sections, model name, variables, and
`A`/`D` matrices. Re-running the test module under the same runtime means the
pinned-parser case executes instead of skipping, and the declared matrices must
still match `chapters/chapter_09/example_9_1_state_inference.py`. Neither the
job nor the validator scans the rest of the repository for `*.gnn.md`.

The adoption context, commands, and the limits of what each check proves are in
[`docs/gnn.md`](../docs/gnn.md).
