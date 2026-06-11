# Chapter 12 — concept map

Chapter 12 recasts categorical inference as message passing on factor graphs.
The companion keeps the implementation deliberately small: normalized
sum-product messages, forward-backward smoothing, a VMP categorical update, and
a simple bridge between continuous features and discrete beliefs.

> **Implemented:** PDF-grounded Chapter 12 companion coverage with
> factor-graph containers,
> belief propagation, backward smoothing, VMP updates, hybrid bridges, thin
> orchestrators, and paired NPZ/JSON exports.

## Script inventory

| File | Role |
|---|---|
| `example_12_1_factor_graph_messages.py` | Compare forward messages and forward-backward smoothed beliefs in a categorical chain. |
| `example_12_2_belief_propagation_smoothing.py` | Show forward messages, backward smoothing messages, and smoothed beliefs together. |
| `example_12_3_vmp_marginal_messages.py` | Compare VMP-style local updates with marginal message-passing beliefs. |
| `example_12_4_hybrid_message_bridge.py` | Demonstrate VMP-style categorical messages and a hybrid continuous/discrete belief bridge. |
| `example_12_5_active_factor_learning_attention.py` | Couple factor messages to active-inference-style preferences and learning-attention weights. |

## Library surface

`FactorGraphChain` records a categorical chain, while
`sum_product_chain`, `backward_smoothing_messages`, and
`forward_backward_smoothing` expose the message-passing path. The
`variational_message_update` helper gives a compact VMP-style categorical
update, `marginal_message_passing` exposes approximate marginals,
`active_inference_factor_messages` couples preferences to messages,
`learning_attention_message` precision-weights categorical evidence, and
`hybrid_model_bridge` turns a continuous feature vector plus a categorical
belief into a joint hybrid representation.

## Where the book takes this next

Chapter 13 uses the same inference and planning primitives in application-level
settings: robotics navigation, control, and social inference.
