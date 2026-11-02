# GNN Specification: Fundamentals Weather POMDP

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1.1

## ModelName

Fundamentals Weather POMDP (Example 9.1)

## ModelAnnotation

Discrete perception model of the weather agent in *Fundamentals of Active
Inference* (Namjoshi), Chapter 9 §9.1, Example 9.1; implemented in this
repository by `chapters/chapter_09/example_9_1_state_inference.py`.

- Four hidden states (rainy, cloudy, sunny, snowy) and three observations
  (water, hot, bright); `A` is the 3x4 column-stochastic likelihood
  `A[o, s] = P(observation o | hidden state s)`.
- `D` is the uniform prior over the four hidden states.
- Perception is exact and static: for a one-hot observation the posterior is
  `q(s) = sigma(log A^T o + log D)` (book Eq. 13); observing "hot" reproduces
  the book's Eq. 15 posterior `[0.18, 0.40, 0.36, 0.06]`.
- This document declares the model. The implementation's copy of the same
  literals lives in the chapter orchestrator, and
  `tests/test_gnn_spec.py` fails when either side is edited alone.

## StateSpaceBlock

A[3,4,type=float]      # Likelihood P(observation | hidden state); columns are states
D[4,type=float]        # Prior over hidden states
s[4,1,type=float]      # Posterior over hidden states
o[3,1,type=int]        # Observation (one-hot)

## Connections

D>s
s-A
A-o
o>s

## InitialParameterization

# Columns are hidden states (rainy, cloudy, sunny, snowy); rows are observations
# (water, hot, bright); every column sums to 1 over observations.
A={
  (0.80, 0.33, 0.05, 0.40),
  (0.15, 0.33, 0.30, 0.05),
  (0.05, 0.34, 0.65, 0.55)
}

# Uniform prior over the four hidden states.
D={(0.25, 0.25, 0.25, 0.25)}

## Equations

Likelihood (every column sums to 1 over observations):
  A[o, s] = P(o | s)

Exact static perception for a one-hot observation o (book Eq. 13):
  q(s) = sigma(log A^T o + log D) = softmax(log A[:, o] + log D)

Book Eq. 15, observing "hot" (the second row of A):
  q = (0.18, 0.40, 0.36, 0.06)

## Time

Time=t
Static
Discrete

## ActInfOntologyAnnotation

A=LikelihoodMatrix
D=PriorOverHiddenStates
s=HiddenState
o=Observation

## ModelParameters

num_hidden_states: 4
num_obs: 3
state_labels: rainy, cloudy, sunny, snowy
observation_labels: water, hot, bright
example_observation: hot
expected_posterior: 0.18, 0.40, 0.36, 0.06

## Footer

Fundamentals Weather POMDP (Example 9.1) — the discrete perception model of
`chapters/chapter_09/example_9_1_state_inference.py`, declared here as GNN.
Validated with the parser revision pinned in `specs/gnn-pin.json`.

## Signature

Hand-authored declaration; bound to the implementation by
`tests/test_gnn_spec.py`. Not cryptographically signed.
