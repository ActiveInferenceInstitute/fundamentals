#!/usr/bin/env bash
# Top-level chapter runner. Thin wrapper around the text menu in
# ``src/active_inference/menu/`` and the local web UI in
# ``src/active_inference/web/``. All logic lives in those modules; this
# script only resolves the right Python interpreter and routes between
# the two front ends.
#
# Usage:
#   ./run.sh                          # interactive text menu
#   ./run.sh --all                    # run every chapter (headless)
#   ./run.sh --chapter 2              # run chapter 2 only
#   ./run.sh --script example_2_2     # run one orchestrator by name
#   ./run.sh --list                   # print the menu and exit
#   ./run.sh --no-animations          # skip slow GIF renderers
#   ./run.sh --keep-going             # continue past failing scripts
#   ./run.sh --web                    # launch the local web UI in your browser
#   ./run.sh --web --no-browser       # web UI without auto-opening a browser
#   ./run.sh --web --port 8080        # custom port (uses ephemeral if taken)
#
# Environment knobs (override before invocation):
#   RUN_PY  — explicit python interpreter to use, e.g. RUN_PY=python3.12
#   USE_UV  — set to 0 to bypass `uv run` even if `uv` is on PATH

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Route to the web UI when the user passes --web (with or without trailing args).
TARGET_MODULE="active_inference.menu"
if (($# > 0)); then
    ARGS=("$@")
else
    unset ARGS
fi
for arg in "$@"; do
    if [[ "$arg" == "--web" ]]; then
        TARGET_MODULE="active_inference.web"
        # Strip the --web sentinel; everything else is forwarded to the web CLI.
        FILTERED=()
        for a in "${ARGS[@]+"${ARGS[@]}"}"; do
            [[ "$a" == "--web" ]] && continue
            FILTERED+=("$a")
        done
        if ((${#FILTERED[@]} > 0)); then
            ARGS=("${FILTERED[@]}")
        else
            unset ARGS
        fi
        break
    fi
done

# The text menu wants Agg (no display); the web server should leave the
# host's default backend alone so interactive scripts can pop a window.
if [[ "$TARGET_MODULE" == "active_inference.menu" ]]; then
    export MPLBACKEND="${MPLBACKEND:-Agg}"
fi

if [[ -n "${RUN_PY:-}" ]]; then
    exec "$RUN_PY" -m "$TARGET_MODULE" "${ARGS[@]+"${ARGS[@]}"}"
fi

if [[ "${USE_UV:-1}" != "0" ]] && command -v uv >/dev/null 2>&1; then
    exec uv run python -m "$TARGET_MODULE" "${ARGS[@]+"${ARGS[@]}"}"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 -m "$TARGET_MODULE" "${ARGS[@]+"${ARGS[@]}"}"
fi

if command -v python >/dev/null 2>&1; then
    exec python -m "$TARGET_MODULE" "${ARGS[@]+"${ARGS[@]}"}"
fi

echo "run.sh: no python interpreter found on PATH (and \`uv\` is also unavailable)." >&2
echo "Install Python 3.9+ or uv (https://docs.astral.sh/uv/) and re-run." >&2
exit 127
