# `chapters/chapter_12/` — agent guide

Chapter 12 scripts are thin wrappers for factor-graph and message-passing
helpers in `active_inference`.

## Rules

- Keep factor algebra in `src/active_inference/core/factor_graph.py`.
- Use `--save` plus `save_chapter_data(12, ...)` for every non-interactive script.
- Do not import from sibling chapter scripts.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_12_1_factor_graph_messages.py`](example_12_1_factor_graph_messages.py) | ~80 | Forward messages, backward smoothing, and smoothed categorical beliefs. |
| [`example_12_2_belief_propagation_smoothing.py`](example_12_2_belief_propagation_smoothing.py) | ~50 | Belief-propagation forward/backward messages and smoothing diagnostics. |
| [`example_12_3_vmp_marginal_messages.py`](example_12_3_vmp_marginal_messages.py) | ~50 | VMP updates and marginal message passing in categorical factors. |
| [`example_12_4_hybrid_message_bridge.py`](example_12_4_hybrid_message_bridge.py) | ~70 | VMP-style factor messages and a simple hybrid continuous/discrete belief bridge. |
| [`example_12_5_active_factor_learning_attention.py`](example_12_5_active_factor_learning_attention.py) | ~60 | Active-inference factor messages plus learning/attention precision weights. |

## Library Usage

```python
from active_inference import (
    FactorGraphChain,
    backward_smoothing_messages,
    forward_backward_smoothing,
    sum_product_chain,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered chapter script with
`--save` in the single parametrized `test_chapter_script_runs_and_exports_raw_data`.
Unit tests for the factor-graph helpers live in `tests/core/test_factor_graph.py`.
