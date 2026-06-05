# Order And Surprisal

Order as low-surprisal occupancy over viable states.

## Book Mapping

- Family: Foundations
- Chapters: 1, 14
- Sections: 1.2, 14.3

## Scripts

- `visualize_order_and_surprisal.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_order_and_surprisal.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_order_and_surprisal.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/order_and_surprisal` and raw-data sidecars under `output/data/extras/order_and_surprisal`.

```bash
uv run python extras/order_and_surprisal/visualize_order_and_surprisal.py --save
uv run python extras/order_and_surprisal/simulate_order_and_surprisal.py --save
uv run python extras/order_and_surprisal/interactive_order_and_surprisal.py
```
