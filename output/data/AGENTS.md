# `output/data/` — agent guide

Ephemeral, regenerable storage for numerical artifacts.

## Hard rules

1. **Everything here is regenerable.** If you ever feel tempted to keep
   a hand-edited file under `output/data/`, that file belongs in
   `chapters/` (as a script that produces it) or `docs/` (as text).
2. **Never check in generated artifacts.** `.gitignore` excludes
   everything here except `.gitkeep`.
3. **Filenames mirror the producing orchestrator** so an `ls` reveals
   which scripts have run.

## Suggested formats

| Format | When |
|---|---|
| `.npz` | Multi-array NumPy outputs (loadings + LL trace + posterior means). |
| `.json` | Metadata, configuration snapshots. |
| `.csv` | Results that need to open in tools other than Python. |
| `.pkl` | **Avoid** — not portable across versions. |
