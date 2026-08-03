# `output/figures/` — agent guide

Ephemeral, regenerable storage for figures (PNGs) and animations (GIFs).

## Hard rules

1. **Generated media here is regenerable** by `scripts/run_all_figures.py`.
   Anything not regenerable belongs in `docs/` or `chapters/`, not as media here.
2. **Never hand-edit generated media.** If a figure needs cosmetic tweaks,
   change the helper in `src/active_inference/visualizations/` or the
   orchestrator in `chapters/`.
3. **Keep docs hand-maintained.** README/AGENTS files explain the artifact
   contract and are intentionally not cleaned by `--clean`.
4. **Do not add new generated media to history without intent.** `.gitignore`
   excludes new PNG/GIF media recursively; historical sample artifacts may
   remain tracked.

## Subfolder layout

```
output/figures/
├── chapter_01/   ← from chapters/chapter_01/0*.py --save
├── chapter_02/   ← from chapters/chapter_02/example_*.py and animation_*.py
├── chapter_03/   ← from chapters/chapter_03/example_*.py and animation_*.py
├── chapter_04/   ← from chapters/chapter_04/*.py
├── chapter_05/   ← from chapters/chapter_05/*.py
├── chapter_06/   ← from chapters/chapter_06/*.py
├── chapter_07/   ← from chapters/chapter_07/*.py
├── chapter_08/   ← from chapters/chapter_08/*.py
├── chapter_09/   ← from chapters/chapter_09/*.py
├── chapter_10/   ← from chapters/chapter_10/*.py
├── chapter_11/   ← from chapters/chapter_11/*.py
├── chapter_12/   ← from chapters/chapter_12/*.py
├── chapter_13/   ← from chapters/chapter_13/*.py
├── chapter_14/   ← from chapters/chapter_14/*.py
├── demo/         ← from demo/<slug>/visualize_<slug>.py
└── extras/       ← from extras/<topic>/visualize_<topic>.py
```

## Filename contract

Each output uses the prefix of its producing script:

| Script | Output |
|---|---|
| `chapters/chapter_03/example_3_5_bayesian_linear_regression.py` | `output/figures/chapter_03/example_3_5_blr_*.png` |
| `chapters/chapter_03/animation_em_convergence.py` | `output/figures/chapter_03/animation_em_convergence.gif` |

Following this convention means a directory listing alone reveals which
orchestrators have been run and which still need to be generated.
