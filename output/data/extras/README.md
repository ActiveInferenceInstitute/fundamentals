# `output/data/extras/` - Extras Raw Data

This directory stores paired NPZ and JSON sidecars for extras topic figures,
simulations, and animations. Each topic writes to
`output/data/extras/<topic>/` when a non-interactive script runs with `--save`.
Interactive extras wrappers do not write sidecars; their shared slider
constructor is covered by visualization tests.

The complete live topic registry is documented in
[`../../../extras/README.md`](../../../extras/README.md). This output directory
is intentionally layout-focused because generated topic subfolders appear only
after their corresponding extras scripts have been rendered.
Linked figure and animation artifacts live under
`output/figures/extras/<topic>/`.

## Layout

| Path | Produced by | Contents |
|---|---|---|
| `output/data/extras/<topic>/visualize_<topic>.npz` + `.json` | `extras/<topic>/visualize_<topic>.py --save` | Static figure arrays and manifest metadata. |
| `output/data/extras/<topic>/simulate_<topic>.npz` + `.json` | `extras/<topic>/simulate_<topic>.py --save` when the topic declares a simulation mode | Parameter-sweep or trajectory arrays and manifest metadata. |
| `output/data/extras/<topic>/animation_<topic>.npz` + `.json` | `extras/<topic>/animation_<topic>.py --save` when the topic declares an animation mode | Animation frame data and manifest metadata. |

Extras manifests include the registry source API references used to build the
demo, making each saved artifact traceable to tested `active_inference`
methods.

Validate all currently rendered sidecars with:

```bash
uv run python scripts/validate_raw_data_exports.py --root output/data
uv run python scripts/validate_book_topic_coverage.py --require-rendered
```
