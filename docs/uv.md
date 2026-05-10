# Using `uv` with this project

`uv` is the recommended package and environment manager for this
repository. It is faster than `pip`, gives us a deterministic
`uv.lock`, and the `uv run` wrapper makes it painless to invoke
commands inside the project's virtualenv without activating it
manually.

Install once: <https://docs.astral.sh/uv/getting-started/installation/>.

## TL;DR

```bash
git clone https://github.com/ActiveInferenceInstitute/fundamentals
cd fundamentals
uv sync                         # create .venv, install runtime + dev deps
uv run pytest                   # run the test suite
uv run ./run.sh                 # launch the chapter text menu
uv run ./run.sh --web           # launch the local web UI (see docs/web.md)
uv run python -m active_inference.menu --all   # batch-render every chapter
```

## Environment

| Command | What it does |
|---|---|
| `uv sync` | Materialize `.venv/` from `uv.lock`. Installs the runtime dependencies plus the `dev` group. |
| `uv sync --extra interactive` | Add the optional `ipywidgets` / `jupyter` extras. |
| `uv sync --no-dev` | Runtime dependencies only — useful in shrinking CI containers. |
| `uv lock` | Resolve `pyproject.toml` and rewrite `uv.lock`. Run this after adding a dependency. |
| `uv lock --upgrade` | Resolve again, allowing newer versions. |

`.venv/` is git-ignored; `uv.lock` is checked in so contributors and CI
see the same environment.

## Running commands

`uv run <command>` invokes the command using the project's interpreter,
auto-creating / syncing the venv as needed. Equivalent to activating
`.venv/` and running the command, but more ergonomic for scripted use.

```bash
uv run python -c "import active_inference; print(active_inference.__version__)"
uv run pytest tests/core
uv run ruff check src tests
uv run python scripts/run_all_figures.py --chapters 1 --clean
```

The top-level [`../run.sh`](../run.sh) prefers `uv run`, falls back to
`python3`, then `python`. Set `USE_UV=0` to bypass the `uv run` branch
even when `uv` is on `PATH` (e.g. on a CI runner that prefers a
pre-baked image's `python`).

## Adding a dependency

1. Edit `pyproject.toml`: add the package to `[project].dependencies`
   (runtime), `[project.optional-dependencies].interactive` (notebooks),
   or `[dependency-groups].dev` (developer tooling).
2. Run `uv lock` to update `uv.lock`.
3. Run `uv sync` to install it locally.
4. Commit both `pyproject.toml` and `uv.lock`.

## Plain-pip compatibility

`pyproject.toml` is a standard PEP 621 project, so `pip install -e
".[dev]"` still works. `requirements.txt` is kept as a minimal
runtime-only file for users who prefer that workflow; the source of
truth, however, is `pyproject.toml`.

## Troubleshooting

- *`uv sync` resolves but the import fails:* make sure you ran the
  command from the repo root (so it picks up the local `pyproject.toml`)
  and that you didn't activate a different venv first.
- *`uv.lock` shows up dirty in `git status` after `uv sync`:* you
  probably edited `pyproject.toml` without running `uv lock`. Commit the
  regenerated lock alongside the `pyproject.toml` change.
- *`run.sh` complains about no Python:* install Python 3.9+ or `uv`. The
  script intentionally avoids guessing other interpreters.
