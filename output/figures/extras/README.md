# `output/figures/extras/` - Generated Extras Figures

This directory contains regenerable PNG and GIF artifacts for cross-cutting
extras topics. Static and simulation scripts write PNGs; animation scripts
write GIFs. Interactive extras wrappers are GUI launchers and are tested through
their reusable constructors rather than saved here.

The complete live topic registry is documented in
[`../../../extras/README.md`](../../../extras/README.md). This output directory
is intentionally layout-focused because generated topic subfolders appear only
after their corresponding extras scripts have been rendered.

## Layout

| Path | Produced by | Raw sidecars |
|---|---|---|
| `output/figures/extras/<topic>/visualize_<topic>.png` | `extras/<topic>/visualize_<topic>.py --save` | `output/data/extras/<topic>/visualize_<topic>.npz` + `.json` |
| `output/figures/extras/<topic>/simulate_<topic>.png` | `extras/<topic>/simulate_<topic>.py --save` when the topic declares a simulation mode | `output/data/extras/<topic>/simulate_<topic>.npz` + `.json` |
| `output/figures/extras/<topic>/animation_<topic>.gif` | `extras/<topic>/animation_<topic>.py --save` when the topic declares an animation mode | `output/data/extras/<topic>/animation_<topic>.npz` + `.json` |

Validate the rendered extras contract with:

```bash
uv run python scripts/validate_book_topic_coverage.py --require-rendered
```

Generated media are ignored by git. Topic README files are hand-maintained.
