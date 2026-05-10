# `docs/reference/` — agent guide

API reference. One file per subpackage under `src/active_inference/`.
Update *every* time the public surface (`__all__`) of a subpackage
changes.

## Hard contract

Each reference page must contain, in order:

1. **H1 title** naming the subpackage.
2. **Overview paragraph** — what the subpackage is for.
3. **Module table** mapping `.py` filenames to roles.
4. **Public API table** listing every symbol in `__all__` with its
   signature or one-line role description.
5. **Conventions** — invariants enforced by the subpackage.
6. **See also** — relative links into `topics/` and `statistics/`.

## When to edit

| Change | Edit |
|---|---|
| New public function in `core.distributions` | `core.md` API table. |
| New class in `estimators.linear_regression` | `estimators.md` API table. |
| New plotting helper | `visualizations.md` API table. |
| New `core.diagnostics` symbol | `core.md` (and likely a new `statistics/` page). |

## Verification

Before merging, run::

    python -c "import active_inference as ai; print(sorted(ai.__all__))"

and diff against the API table in `core.md` / `estimators.md` /
`utils.md` / `visualizations.md`. Any mismatch must be reconciled.

## Avoid

- Tutorial prose — that belongs in `topics/`.
- Mathematical derivations — that belongs in `statistics/`.
- Deprecated symbols — remove the row when the symbol is removed from
  `__all__`.
