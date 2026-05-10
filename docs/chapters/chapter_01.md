# Chapter 1 — concept map

Chapter 1 of the book sets up the conceptual frame: an agent embedded in an
environment to which it has no direct access. It must reconstruct what is
happening "out there" purely from a stream of noisy sensor readings.

## What the orchestrators show

`chapters/chapter_01/01_box_scenario.py` simulates the canonical thought
experiment — a sealed agent receives a stream of noisy observations from a
fixed external state. Even with the generator and noise variance fully known
to us (the modellers), the agent only ever sees the histogram on the right
panel of the figure: it must *invert* that histogram to recover the
unobserved hidden state.

`chapters/chapter_01/02_three_perspectives.py` lays the same simulation under
three viewpoints side by side:

* the **scientific modeller** who proposes a hypothesis and refits as data
  arrives,
* the **hypothesis-testing brain** view, where the system makes a prediction
  and learns from the prediction error,
* the **statistical** view, where the agent maintains a posterior over its
  hidden state.

`chapters/chapter_01/03_bayes_intuition.py` walks through Bayes' theorem one
factor at a time. It is the natural lead-in to Chapter 2.

`chapters/chapter_01/04_inverse_problem.py` shows the failure mode: when the
generator is non-injective (here `y = beta0 + beta1 x²`), two states explain
the same observation and the posterior is bi-modal. This is the inverse
problem in its simplest form.

## Going further

* Try editing `--y-obs` and `--seed` on `01_box_scenario.py` to see how the
  histogram concentrates around the noise-free generator.
* In `04_inverse_problem.py`, replace the uniform prior with a Gaussian
  prior centred at a positive value and re-run; the bi-modality vanishes.
  This is the cheapest possible "hierarchical model" — Chapter 8 of the
  book formalizes this idea.
