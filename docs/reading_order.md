# Reading order

These docs cover four audiences. Pick the path that matches what you're
trying to do.

> Reading the book alongside others? The
> [Active Inference Institute](https://activeinference.institute/) runs
> rolling cohorts of the textbook group, free and open to everyone —
> register at
> [textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/).

## Path A — *I'm following the book chapter by chapter*

1. [`chapters/chapter_01.md`](chapters/chapter_01.md) → run
   [`chapters/chapter_01/01_box_scenario.py`](../chapters/chapter_01/01_box_scenario.py).
2. [`chapters/chapter_02.md`](chapters/chapter_02.md) → run Examples 2.1–2.10.
3. [`chapters/chapter_03.md`](chapters/chapter_03.md) → run Examples 3.1–3.7
   plus the diagnostic visualizers.
4. [`chapters/chapter_04.md`](chapters/chapter_04.md) → run the variational
   examples (coordinate search, fixed-form VI, mean-field CAVI) and the
   `animation_vfe_descent.py` GIF.
5. [`chapters/chapter_05.md`](chapters/chapter_05.md) → run the predictive-coding
   examples (recognition dynamics, multivariate, hierarchical) and the two
   animation orchestrators.
6. [`chapters/chapter_06.md`](chapters/chapter_06.md) → run generalized filtering
   for perception: online tracking, multivariate filtering, generalized
   coordinates, correlated embedding-order precision, and Example 6.7 vector
   generalized filtering.
7. [`chapters/chapter_07.md`](chapters/chapter_07.md) → run active generalized
   filtering: start with set-point regulation, then inspect the §7.5
   multivariate action-perception loop and animation.
8. [`chapters/chapter_08.md`](chapters/chapter_08.md) → run learning/attention and
   hierarchy/message-passing examples.
9. [`chapters/chapter_09.md`](chapters/chapter_09.md) → run the categorical POMDP
   examples: state inference, dynamic filtering, discrete VFE, Grid World, and EFE.
10. [`chapters/chapter_10.md`](chapters/chapter_10.md) → run POMDP learning,
    habit/precision, factorial bandit, and hierarchical examples.
11. [`chapters/chapter_11.md`](chapters/chapter_11.md) → run Part III planning
    extensions: free-energy variants, sophisticated inference, policy-tree
    search, preferences, forgetting, and structure learning.
12. [`chapters/chapter_12.md`](chapters/chapter_12.md) → run factor-graph
    message passing, smoothing, VMP, and hybrid bridge examples.
13. [`chapters/chapter_13.md`](chapters/chapter_13.md) → run robotics
    navigation and social-inference application demos.
14. [`chapters/chapter_14.md`](chapters/chapter_14.md) → run ergodic-density,
    entropy-bound, Bayesian-mechanics, and Markov-blanket demos.
15. When a concept is unfamiliar, jump to the matching `topics/` page
   and come back.

## Path B — *I want a deep dive on a concept*

Start at [`topics/`](topics/) and pick:

| If you're trying to learn… | Read |
|---|---|
| What Bayesian inference is and why it works on a grid here | [`topics/bayesian_inference.md`](topics/bayesian_inference.md) |
| How the process / model split is realized in code | [`topics/generative_models.md`](topics/generative_models.md) |
| The MLE / MAP / BLR / EM lineage | [`topics/learning_and_inference.md`](topics/learning_and_inference.md) |
| Why Cholesky-based solves matter for MVN math | [`topics/multivariate_gaussians.md`](topics/multivariate_gaussians.md) |
| The stability bound on gradient descent | [`topics/gradient_descent.md`](topics/gradient_descent.md) |
| When inversion goes bi-modal | [`topics/inverse_problem.md`](topics/inverse_problem.md) |
| The big-picture active-inference / FEP / Bayesian-mechanics triangle | [`topics/active_inference.md`](topics/active_inference.md) → [`topics/free_energy_principle.md`](topics/free_energy_principle.md) → [`topics/bayesian_mechanics.md`](topics/bayesian_mechanics.md) |

## Path C — *I want to use the library in my own code*

1. [`cookbook.md`](cookbook.md) — copy-paste recipes for the 10 most-used
   workflows.
2. [`reference/core.md`](reference/core.md), [`reference/estimators.md`](reference/estimators.md),
   [`reference/utils.md`](reference/utils.md), [`reference/visualizations.md`](reference/visualizations.md)
   — every public symbol with its signature and purpose.
3. [`architecture.md`](architecture.md) — the layered design, so you know
   which subpackage owns which concern.
4. [`notation.md`](notation.md) — book symbol ↔ Python identifier table.

## Path D — *I want to extend the library*

1. [`AGENTS.md`](AGENTS.md) and the per-folder `AGENTS.md` files
   ([`chapters/AGENTS.md`](chapters/AGENTS.md), [`topics/AGENTS.md`](topics/AGENTS.md),
   [`statistics/AGENTS.md`](statistics/AGENTS.md), [`reference/AGENTS.md`](reference/AGENTS.md))
   spell out the *file contracts* every new page must satisfy.
2. The src tree's `AGENTS.md` files (under
   `src/active_inference/<sub>/`) tell you when adding a new file is
   justified and what minimum review checklist applies.
3. The tests tree mirrors the src tree one-for-one — see
   [`../tests/AGENTS.md`](../tests/AGENTS.md).

## Path E — *I want to verify a statistical claim the codebase makes*

Each page in [`statistics/`](statistics/) follows the same shape:
**definition → closed form (when one exists) → API → tests that pin it
down → pitfalls**. The "tests that pin it down" pointers are real test
classes you can run with::

    pytest tests/core/test_diagnostics.py::TestKL

Combine that with [`reference/core.md`](reference/core.md) (the API
table) for full traceability from claim to formula to implementation to
test.

## Quick orientation

If you only ever read one page, make it [`cookbook.md`](cookbook.md);
it surfaces the most common entry points and links into every other
section. If you read a second, make it
[`architecture.md`](architecture.md); it explains how the four
subpackages depend on each other and why the chapter scripts stay
short.
