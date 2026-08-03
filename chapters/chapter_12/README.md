# Chapter 12 — Factor graphs and message passing

Chapter 12 reframes active-inference updates as local factor-graph messages.
The concept map is
[`docs/chapters/chapter_12.md`](../../docs/chapters/chapter_12.md).

## Scripts

| Script | Mirrors | What it shows |
|---|---:|---|
| `example_12_1_factor_graph_messages.py` | §12.1-§12.3 | Forward messages, backward smoothing, and smoothed categorical beliefs. |
| `example_12_2_belief_propagation_smoothing.py` | §12.2-§12.3 | Belief-propagation forward/backward messages and smoothing diagnostics. |
| `example_12_3_vmp_marginal_messages.py` | §12.4 | VMP updates and marginal message passing in categorical factors. |
| `example_12_4_hybrid_message_bridge.py` | §12.4-§12.6 | VMP-style factor messages and a simple hybrid continuous/discrete belief bridge. |
| `example_12_5_active_factor_learning_attention.py` | §12.5-§12.8 | Active-inference factor messages plus learning/attention precision weights. |

## Running

```bash
uv run python chapters/chapter_12/example_12_1_factor_graph_messages.py --save
uv run python chapters/chapter_12/example_12_2_belief_propagation_smoothing.py --save
uv run python chapters/chapter_12/example_12_3_vmp_marginal_messages.py --save
uv run python chapters/chapter_12/example_12_4_hybrid_message_bridge.py --save
uv run python chapters/chapter_12/example_12_5_active_factor_learning_attention.py --save
```

`--save` writes figures to `output/figures/chapter_12/` and raw NPZ+JSON
sidecars to `output/data/chapter_12/`.

## Programmatic usage

```python
from active_inference import (
    FactorGraphChain,
    backward_smoothing_messages,
    forward_backward_smoothing,
    sum_product_chain,
)
import numpy as np

prior = np.array([0.5, 0.5])
transitions = np.array([[0.8, 0.2], [0.2, 0.8]])
likelihoods = np.array([[0.9, 0.1], [0.1, 0.9]])
smoothed = forward_backward_smoothing(prior, transitions, likelihoods)
print(smoothed.shape)
```

The factor-graph helpers live in `core.factor_graph`; the scripts wrap them
with the book's §12.1–§12.8 teaching examples.
