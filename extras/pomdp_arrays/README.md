# POMDP Arrays

D, A, B, C, and E arrays as discrete generative-model components.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9, 10
- Sections: 9.1, B.10

## Scripts

- `visualize_pomdp_arrays.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_pomdp_arrays.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_pomdp_arrays.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/pomdp_arrays` and raw-data sidecars under `output/data/extras/pomdp_arrays`.

```bash
uv run python extras/pomdp_arrays/visualize_pomdp_arrays.py --save
uv run python extras/pomdp_arrays/simulate_pomdp_arrays.py --save
uv run python extras/pomdp_arrays/interactive_pomdp_arrays.py
```
