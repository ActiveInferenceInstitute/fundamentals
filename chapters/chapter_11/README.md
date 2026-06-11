# Chapter 11 - Part III planning extensions

Chapter 11 collects active-inference planning extensions: alternate free-energy
forms, sophisticated lookahead, state preferences, parameter uncertainty, and
structure learning.

| Script | Mirrors | What it shows |
|---|---:|---|
| `example_11_1_free_energy_variants.py` | §11.1 / Appendix D | EFE, FEF, GFE, and Renyi-style certainty-equivalent teaching curves over policies. |
| `example_11_2_sophisticated_planning.py` | §11.2 | Bounded tree policy search, future-belief entropy, time-dependent preferences, forgetting, and structure posterior diagnostics. |
| `example_11_3_preference_habit_learning.py` | §11.2.4 | Preference pseudocount learning and habit-prior construction. |
| `example_11_4_hybrid_tree_structure.py` | §11.3-§11.5 | Hybrid model evidence, path-policy scores, and structure posteriors. |

## Running

```bash
uv run python chapters/chapter_11/example_11_1_free_energy_variants.py --save
uv run python chapters/chapter_11/example_11_2_sophisticated_planning.py --save
uv run python chapters/chapter_11/example_11_3_preference_habit_learning.py --save
uv run python chapters/chapter_11/example_11_4_hybrid_tree_structure.py --save
```
