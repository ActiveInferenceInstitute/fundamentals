# Tree Policy Search

Tree-based optimization and receding policy search.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.4

## Scripts

- `visualize_tree_policy_search.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_tree_policy_search.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_tree_policy_search.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_tree_policy_search.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/tree_policy_search` and raw-data sidecars under `output/data/extras/tree_policy_search`.

```bash
uv run python extras/tree_policy_search/visualize_tree_policy_search.py --save
uv run python extras/tree_policy_search/simulate_tree_policy_search.py --save
uv run python extras/tree_policy_search/interactive_tree_policy_search.py
uv run python extras/tree_policy_search/animation_tree_policy_search.py --save
```
