# Chapter 12 - Factor graphs and message passing

Chapter 12 reframes active-inference updates as local factor-graph messages.

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
