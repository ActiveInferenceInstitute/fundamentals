# `output/data/` — Raw Numerical Exports

This directory stores reproducible raw data for chapter and extras artifacts.
Every non-interactive chapter or extras script that runs with `--save` writes at
least one paired export:

- `chapter_NN/<stem>.npz` — compressed `NPZ` arrays, numeric only.
- `chapter_NN/<stem>.json` — `JSON` metadata with script name, chapter,
  CLI args, seed when present, figure paths, array names/shapes/dtypes, and
  per-array summary statistics.
- `extras/<topic>/<stem>.npz` and `.json` — the same contract for extras
  topic orchestrators.

The shared figure helpers export plotted reconstruction data automatically.
Scripts that need bespoke arrays can call `save_chapter_data(chapter, stem,
arrays, metadata, figures=...)` or `save_extra_data(topic, stem, arrays,
metadata, figures=...)`.

## Validation

```bash
python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14
python scripts/validate_raw_data_exports.py --root output/data
python scripts/validate_book_topic_coverage.py
python scripts/validate_book_topic_coverage.py --require-rendered
python scripts/validate_source_spine.py --require-pdf
```

The validator rejects missing JSON/NPZ partners, empty arrays, object arrays,
non-finite values, and shape/dtype manifest drift.

## Regenerate

```bash
python scripts/run_all_figures.py --clean --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14
python scripts/run_all_figures.py --no-chapters --extras
python -m active_inference.menu --extras
```

Generated `.npz` and `.json` files are ignored by git. README/AGENTS files are
hand-maintained so the raw-data contract remains visible.
