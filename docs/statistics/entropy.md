# Entropy

Differential entropy ``H[p]`` measures the average surprise of samples
drawn from ``p``, in nats. Unlike discrete entropy it is **not** bounded
below by zero — a tightly concentrated continuous density has *negative*
differential entropy. The codebase reports entropy as one of the
quick-look statistics on every grid posterior.

## Definition

For a continuous density on the real line::

    H[p] = − ∫ p(x) log p(x) dx

The integrand is taken to be zero wherever ``p(x) = 0``.

## Closed form for Gaussians

Univariate::

    H[N(μ, σ²)] = ½ log(2π e σ²)

Multivariate (dimension `d`)::

    H[N(μ, Σ)] = ½ log( (2π e)^d |Σ| )

## API

| Function | Signature | Notes |
|---|---|---|
| `grid_entropy` | `(p, x_grid) -> float` | Trapezoid rule. |
| `gaussian_entropy_univariate` | `(sigma2) -> float` | Closed form. |
| `gaussian_entropy_mvn` | `(cov) -> float` | Cholesky-based log-det; never inverts. |
| `InferenceResult.entropy` | `() -> float` | Convenience on a grid posterior. |

All live in `active_inference.core.diagnostics`.

## Tests that pin it down

`tests/core/test_diagnostics.py::TestEntropy` covers:

- Closed-form Gaussian entropy matches `grid_entropy` on a dense grid.
- Entropy increases monotonically with variance.
- Negative variance raises `ValueError`.
- The 1-D MVN reduces to the univariate formula.
- A near-Dirac density has near-zero (or negative) differential entropy.

## Pitfalls

- **Differential entropy can be negative.** A Gaussian with
  ``σ² < 1 / (2π e)`` has `H[p] < 0`. This is expected, not a bug.
- The grid result depends on how dense the grid is; for tight
  distributions, widen `x_grid` rather than narrowing the grid spacing.
- The Gaussian closed forms exit in nats; multiply by `1/log(2)` for
  bits.

## See also

- [`divergences.md`](divergences.md) — KL is the "extra" entropy paid
  for using `q` instead of `p`.
- [`../topics/bayesian_inference.md`](../topics/bayesian_inference.md)
  — entropy as a uncertainty diagnostic on the posterior.
- [`../reference/core.md`](../reference/core.md) — full API.
