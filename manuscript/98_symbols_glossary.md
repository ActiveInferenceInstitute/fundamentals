# Symbols and Glossary {#sec:symbols_glossary}

## Generative model

A generative model is the agent's internal representation of the environment,
expressed as the joint distribution `p(x, y) = p(y | x) p(x)` over hidden states
`x` and observations `y`.

Source: the module description and `GenerativeModel` interface in
[`src/active_inference/core/generative_model.py`](../src/active_inference/core/generative_model.py).
The interface exposes the log-likelihood and log-prior; concrete models supply
these densities. See also the
[generative-model walkthrough](../docs/topics/generative_models.md).
