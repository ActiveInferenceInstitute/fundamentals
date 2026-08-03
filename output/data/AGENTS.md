# `output/data/` — agent guide

Ephemeral, regenerable storage for raw numerical artifacts.

## Hard rules

1. **Every non-interactive `--save` chapter or extras run exports raw data.**
   The contract is one compressed `NPZ` file plus one `JSON` manifest under
   `output/data/chapter_NN/` or `output/data/extras/<topic>/`.
2. **Use the shared helpers.** Prefer `save_or_show` / `save_animation` for
   figure-derived data, `save_chapter_data` for bespoke chapter arrays, and
   `save_extra_data` for bespoke extras arrays that are not recoverable from
   Matplotlib artists.
3. **Never check in generated arrays/manifests.** `.gitignore` excludes
   generated `.npz` and `.json` files; README/AGENTS files are the only
   hand-maintained content in chapter data folders.
4. **Validate after full renders.** Run
   `python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14`
   for the chapter spine and
   `python scripts/validate_raw_data_exports.py --root output/data` after
   regenerating the extras curriculum.

## Manifest expectations

The JSON manifest records script name, chapter or extras topic, CLI args, seed
when present, linked figure paths, array names, shapes, dtypes, and summary
statistics. Arrays must be finite, numeric, non-empty, and non-object so they
can reconstruct the educational analysis without importing orchestrator scripts.
