# Divergences

The Kullback–Leibler divergence ``KL(p ‖ q)`` measures how much
``p`` differs from ``q`` in nats. It is non-negative, asymmetric, and
zero iff ``p ≡ q`` almost everywhere on the support of ``p``.

## Definition

For continuous densities on a shared support::

    KL(p ‖ q) = ∫ p(x) [log p(x) − log q(x)] dx

The integrand is taken to be ``0`` wherever ``p(x) = 0``, and the
divergence is ``+∞`` if ``q(x) = 0`` somewhere ``p(x) > 0`` (KL is
undefined there).

## Closed form for Gaussians

Univariate::

    KL(N(μ_p, σ²_p) ‖ N(μ_q, σ²_q))
        = ½ [ log(σ²_q / σ²_p) + (σ²_p + (μ_p − μ_q)²) / σ²_q − 1 ]

Multivariate::

    KL(N_p ‖ N_q) = ½ [ tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p)
                       − d + log |Σ_q| − log |Σ_p| ]

## API

| Function | Signature | Notes |
|---|---|---|
| `grid_kl_divergence` | `(p, q, x_grid) -> float` | Trapezoid rule on a shared 1-D grid. Returns `+inf` when `q == 0` while `p > 0`. |
| `gaussian_kl_univariate` | `(mu_p, sigma2_p, mu_q, sigma2_q) -> float` | Closed-form scalar. |
| `gaussian_kl_mvn` | `(mu_p, cov_p, mu_q, cov_q) -> float` | Closed-form vector + matrix. |
| `InferenceResult.kl_from_prior` | `() -> float` | Convenience: `KL(posterior ‖ prior)` from a grid result. |

All three live in `active_inference.core.diagnostics` and are re-exported
from the top-level package.

## Tests that pin it down

`tests/core/test_diagnostics.py::TestKL` covers:

- Self-KL is zero (`grid_kl_divergence(p, p, x)`).
- KL is non-negative for distinct shifted Gaussians.
- The trapezoid grid result matches the closed-form Gaussian KL to
  ``rtol=1e-3``.
- KL returns `+inf` when ``q`` vanishes where ``p`` does not.
- MVN self-KL is zero for any positive-definite covariance.

## Pitfalls

- KL is **asymmetric**: ``KL(p ‖ q) ≠ KL(q ‖ p)``. Use the
  argument order intentionally.
- The grid version requires both densities to live on the *same* grid.
  Resample one if necessary before calling.
- Negative results are a numerical artefact (rounding); the math
  guarantees non-negativity.

## See also

- [`entropy.md`](entropy.md) — KL is `H(p, q) − H(p)` (cross-entropy
  minus entropy).
- [`../topics/bayesian_inference.md`](../topics/bayesian_inference.md)
  — `KL(post ‖ prior)` quantifies the data's effect.
- [`../reference/core.md`](../reference/core.md) — full API.
