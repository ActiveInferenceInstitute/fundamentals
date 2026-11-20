# Tree Policy Search

Tree-based optimization and receding policy search.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.8, 11.4

## Scripts

- `visualize_tree_policy_search.py` - Static visualization using shared library APIs.
- `simulate_tree_policy_search.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_tree_policy_search.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_tree_policy_search.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/tree_policy_search/` and `output/data/extras/tree_policy_search/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_tree_policy_search.py --save` | `visualize_tree_policy_search.png` | `visualize_tree_policy_search.npz`, `visualize_tree_policy_search.json` |
| `simulate_tree_policy_search.py --save` | `simulate_tree_policy_search.png` | `simulate_tree_policy_search.npz`, `simulate_tree_policy_search.json` |
| `animation_tree_policy_search.py --save` | `animation_tree_policy_search.gif` | `animation_tree_policy_search.npz`, `animation_tree_policy_search.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/tree_policy_search/visualize_tree_policy_search.py --save
uv run python extras/tree_policy_search/simulate_tree_policy_search.py --save
uv run python extras/tree_policy_search/interactive_tree_policy_search.py
uv run python extras/tree_policy_search/animation_tree_policy_search.py --save
```

