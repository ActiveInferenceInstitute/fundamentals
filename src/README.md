# `src/` — source tree root

The `src/` layout is used so that tests run against the *installed* package
rather than the working tree by accident. The single subfolder
`active_inference/` is the importable Python package.

```
src/
└── active_inference/   ← import as `active_inference`
    ├── core/             ← distributions, inference, diagnostics, compose, posterior, validators, types, lgs
    ├── estimators/       ← MLE, MAP, gradient descent, linear regression, EM
    ├── utils/            ← grids, logging, paths
    ├── visualizations/   ← plotting, animations, interactive, diagnostics, style
    ├── menu/             ← stdlib text menu used by run.sh
    └── web/              ← stdlib local web UI used by run.sh --web
```

## Why `src/`?

- Forces tests to import from the installed location, catching missing
  packages or stale `__init__` files.
- Cleanly separates package code from project metadata
  (`pyproject.toml`, `tests/`, `chapters/`).
- Standard layout used by most modern Python projects.

## Editable install

```bash
uv sync                  # recommended — creates .venv from uv.lock
# or
pip install -e ".[dev]"  # plain-pip fallback
```

After that, `from active_inference import ...` works from anywhere on the
machine, while edits in `src/active_inference/` are picked up immediately.

## See also

- [`active_inference/README.md`](active_inference/README.md) — package
  contents and public API.
- [`active_inference/AGENTS.md`](active_inference/AGENTS.md) — design
  rules for contributors.
