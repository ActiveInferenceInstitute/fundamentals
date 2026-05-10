# `src/` — agent guide

This folder contains exactly one thing: the importable Python package
`active_inference`. Do not add other top-level packages here without
discussing it first — multi-package layouts are out of scope for this
companion.

## Boundaries

- `src/active_inference/` is the **only** importable package.
- All public API decisions live in `src/active_inference/__init__.py`.
- All design rules live in
  [`src/active_inference/AGENTS.md`](active_inference/AGENTS.md).
- Tests live under the sibling `tests/` directory and mirror this layout.

## Build / install

```bash
uv sync                  # recommended — uses uv.lock
# or
pip install -e ".[dev]"  # plain-pip fallback
```

After install, `from active_inference import ...` is the import surface;
nothing should import from `src.active_inference.<...>` directly.
