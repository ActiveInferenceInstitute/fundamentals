# `output/data/` — serialized numerical results

Reserved for `.npz`, `.json`, and `.csv` files produced by chapter
orchestrators that want to persist non-figure outputs (e.g., posterior
samples, learning curves, EM trajectories).

Currently empty — orchestrators write their numbers into figures rather
than separate files. The directory is kept under version control via a
`.gitkeep` so it always exists for scripts that want it.

## Conventions

- File names mirror the orchestrator that produced them
  (`example_3_7_em_trajectory.npz`, etc.).
- `.npz` for arrays, `.json` for metadata, `.csv` only when a result
  needs to be opened in non-Python tools.
- Generated content is gitignored (`.gitignore` excludes `output/data/*`
  except `.gitkeep`).

## Regenerate

```bash
python scripts/run_all_figures.py --clean
```
