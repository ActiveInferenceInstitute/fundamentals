# `output/figures/` — agent guide

Ephemeral, regenerable storage for figures (PNGs) and animations (GIFs).

## Hard rules

1. **Everything here is regenerable** by `scripts/run_all_figures.py`.
   Anything not regenerable does not belong in this folder.
2. **Never hand-edit a file here.** If a figure needs cosmetic tweaks,
   change the helper in `src/active_inference/visualizations/` or the
   orchestrator in `chapters/`.
3. **Never check in generated artifacts.** `.gitignore` excludes every
   PNG/GIF here; per-chapter `.gitkeep` files keep the directories
   present.

## Subfolder layout

```
output/figures/
├── chapter_01/   ← from chapters/chapter_01/0*.py --save
├── chapter_02/   ← from chapters/chapter_02/example_*.py and animation_*.py
└── chapter_03/   ← from chapters/chapter_03/example_*.py and animation_*.py
```

## Filename contract

Each output uses the prefix of its producing script:

| Script | Output |
|---|---|
| `chapters/chapter_03/example_3_5_bayesian_linear_regression.py` | `output/figures/chapter_03/example_3_5_blr_*.png` |
| `chapters/chapter_03/animation_em_convergence.py` | `output/figures/chapter_03/animation_em_convergence.gif` |

Following this convention means a directory listing alone reveals which
orchestrators have been run and which still need to be generated.
