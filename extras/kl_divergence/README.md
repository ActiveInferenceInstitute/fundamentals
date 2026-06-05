# KL Divergence

Asymmetric divergence and posterior approximation loss.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.1, C.10.5

## Scripts

- `visualize_kl_divergence.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_kl_divergence.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_kl_divergence.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/kl_divergence` and raw-data sidecars under `output/data/extras/kl_divergence`.

```bash
uv run python extras/kl_divergence/visualize_kl_divergence.py --save
uv run python extras/kl_divergence/simulate_kl_divergence.py --save
uv run python extras/kl_divergence/interactive_kl_divergence.py
```
