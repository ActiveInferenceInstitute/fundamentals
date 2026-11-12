#!/usr/bin/env bash
# The repository gate.
#
# One entry point; every arm is separately runnable and calls the same command
# that CI calls. CI jobs do not repeat any command the gate already defines:
# a job installs the locked environment and then invokes `./gate.sh <arm>`.
#
#   ./gate.sh lint             ruff over the whole tree
#   ./gate.sh fast             unit, contract, and documentation suites (CI default)
#   ./gate.sh full             the entire pytest suite, including the smoke tests
#   ./gate.sh docs             documentation and Markdown-link contracts only
#   ./gate.sh contract         Education pointer, schema, and bundle coherence
#   ./gate.sh validators       the repository validators (artifacts, coverage, spine)
#   ./gate.sh manuscript-check manuscript surfaces without Pandoc or TeX
#   ./gate.sh manuscript       render every format enabled in manuscript/config.yaml
#   ./gate.sh gnn              pinned GNN conformance plus the drift guard
#   ./gate.sh all              every arm above, in that order
#   ./gate.sh --list           print the arms
#
# There is no `types` arm: this repository configures no type checker, and
# adding one is its own reviewed migration.
#
# Environment knobs:
#   GATE_CACHE              cache directory for the pinned GNN runtime
#                           (default: ${XDG_CACHE_HOME:-$HOME/.cache}/fundamentals-gate)
#   GNN_RUNTIME_PYTHON      interpreter with the pinned GNN distribution installed;
#                           when unset the gate provisions the pin into GATE_CACHE
#
# Every arm prints `gate: <arm> ok (S.mmms)`; those numbers are the ones recorded
# in docs/gate.md. An arm that needs more than about ten minutes is not measured
# to completion on a development host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND="${MPLBACKEND:-Agg}"
GATE_CACHE="${GATE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/fundamentals-gate}"
GNN_PIN="specs/gnn-pin.json"

# --- The commands -----------------------------------------------------------
# Each arm body is the command CI runs; keep them literal so a reader can copy
# one out of this file and reproduce the job by hand.

arm_lint() {
    uv run ruff check .
}

arm_fast() {
    uv run python -m pytest -q \
        tests/core tests/estimators tests/utils tests/visualizations tests/web \
        tests/test_*.py
}

arm_full() {
    uv run python -m pytest -q
}

arm_docs() {
    uv run python -m pytest -q \
        tests/test_documentation_contracts.py tests/test_method_documentation_contracts.py
}

arm_contract() {
    uv run python scripts/check_education_contract.py
}

arm_validators() {
    uv run python scripts/validate_orchestrator_contracts.py
    uv run python scripts/validate_orchestrator_provenance.py
    uv run python scripts/validate_book_topic_coverage.py
    uv run python scripts/validate_source_spine.py
    uv run python scripts/validate_rendered_figures.py --root output/figures
}

arm_manuscript_check() {
    uv run python scripts/render_manuscript.py --check
}

arm_manuscript() {
    uv run python scripts/render_manuscript.py
}

# Return an interpreter that has the distribution pinned in
# specs/gnn-pin.json installed, provisioning it once per pinned commit.
provision_gnn_runtime() {
    if [[ -n "${GNN_RUNTIME_PYTHON:-}" ]]; then
        printf '%s\n' "$GNN_RUNTIME_PYTHON"
        return 0
    fi

    local repo commit tag
    read -r repo commit tag <<<"$(
        uv run python -c \
            'import json, sys; pin = json.load(open(sys.argv[1])); print(pin["repository"], pin["commit"], pin["tag"])' \
            "$GNN_PIN"
    )"

    local runtime="$GATE_CACHE/gnn-runtime-${commit:0:12}"
    if [[ ! -x "$runtime/bin/python" ]]; then
        mkdir -p "$GATE_CACHE"
        # The install follows the pin, so a pin bump provisions a fresh runtime
        # instead of reusing a stale one.
        uv venv --python 3.12 "$runtime" >/dev/null
        uv pip install --python "$runtime/bin/python" \
            "generalized-notation-notation @ git+${repo}@${commit}" >/dev/null
    fi

    local version expected
    version="$("$runtime/bin/python" -c 'import gnn; print(gnn.__version__)')"
    expected="${tag#v}"
    if [[ "$version" != "$expected" ]]; then
        echo "gate: pinned runtime reports gnn $version, pin names $expected" >&2
        echo "gate: remove $runtime and re-run" >&2
        return 1
    fi
    printf '%s\n' "$runtime/bin/python"
}

arm_gnn() {
    local runtime
    runtime="$(provision_gnn_runtime)"
    uv run python scripts/validate_gnn_spec.py --gnn-runtime "$runtime" --json
    # Without this the pinned-parser test skips instead of parsing.
    GNN_RUNTIME_PYTHON="$runtime" uv run pytest -q tests/test_gnn_spec.py
}

arm_all() {
    run_arm lint arm_lint
    run_arm fast arm_fast
    run_arm docs arm_docs
    run_arm contract arm_contract
    run_arm validators arm_validators
    run_arm manuscript-check arm_manuscript_check
    run_arm manuscript arm_manuscript
    run_arm gnn arm_gnn
    run_arm full arm_full
}

now_micros() {
    # Bash 5 exposes EPOCHREALTIME; bash 3.2 (macOS default) only has SECONDS.
    if [[ -n "${EPOCHREALTIME:-}" ]]; then
        printf '%s' "${EPOCHREALTIME//[.,]/}"
    else
        printf '%s000000' "$SECONDS"
    fi
}

run_arm() {
    local name="$1"
    shift
    local start elapsed
    start="$(now_micros)"
    "$@"
    elapsed=$(( $(now_micros) - start ))
    printf 'gate: %s ok (%d.%03ds)\n' "$name" "$((elapsed / 1000000))" "$((elapsed / 1000 % 1000))"
}

ARMS="lint fast full docs contract validators manuscript-check manuscript gnn all"

usage() {
    printf 'usage: ./gate.sh <%s>\n' "${ARMS// /|}"
    printf '       ./gate.sh --list\n'
}

if (($# != 1)); then
    usage >&2
    exit 2
fi

case "$1" in
    lint) run_arm lint arm_lint ;;
    fast) run_arm fast arm_fast ;;
    full) run_arm full arm_full ;;
    docs) run_arm docs arm_docs ;;
    contract) run_arm contract arm_contract ;;
    validators) run_arm validators arm_validators ;;
    manuscript-check) run_arm manuscript-check arm_manuscript_check ;;
    manuscript) run_arm manuscript arm_manuscript ;;
    gnn) run_arm gnn arm_gnn ;;
    all) arm_all ;;
    --list | -l) printf '%s\n' $ARMS ;;
    --help | -h) usage ;;
    *)
        echo "gate: unknown arm '$1'" >&2
        usage >&2
        exit 2
        ;;
esac
