# Chapter 1 — The Hypothesis-Testing Brain

Chapter 1 is conceptual: it motivates the view of an agent as a Bayesian
inference engine that does not have direct access to the external world and
must reconstruct it from sensory observations.

This folder contains short, runnable orchestrators that turn each conceptual
diagram from the chapter into something you can poke. Every script imports
helpers from `active_inference` rather than re-deriving anything inline.

## Learning objective

- Explain how Bayes' theorem combines a prior and a likelihood to infer hidden
  states from observations.

Practice this objective with `03_bayes_intuition.py`, which shows the update
step by step.

## Scripts

| Script | What it shows |
|--------|---------------|
| `01_box_scenario.py`      | The "agent in a box" thought experiment as a stream of noisy sensor readings. |
| `02_three_perspectives.py`| Side-by-side simulation of the *scientific*, *hypothesis-testing*, and *statistical* views of modeling. |
| `03_bayes_intuition.py`   | Bayes' theorem step-by-step on a single-state, single-observation toy. |
| `04_inverse_problem.py`   | When the generating function is non-injective the posterior splits into multiple modes. |
| `05_belief_from_stream.py`| **Animation** (GIF): the box-scenario sensor stream arriving beside the belief sharpening toward `x*`, with the `σ·√N` concentration statistic. Final mode is cross-checked against the MLE. |
| `interactive_inverse_problem.py`| **Interactive** (GUI / web-launchable): sliders for the observation `y` and likelihood variance `σ_y²` drive the bi-modal posterior of the non-injective generator; readout reports both modes, their separation, and the posterior entropy. |

## Running

```bash
# from the repo root
uv sync                          # creates .venv + installs everything
uv run python chapters/chapter_01/01_box_scenario.py --save

# or via the top-level menu:
./run.sh --chapter 1
```

Pass `--save` to dump figures into `output/figures/chapter_01/`. Without it the
scripts open an interactive window.
