# Chapter 2 — Hidden State Estimation

Chapter 2 turns the conceptual ideas of Chapter 1 into the canonical
linear-Gaussian Bayesian inference recipe and walks through nine numbered
examples that explore variations of it.

Every script here is a **thin orchestrator**: it imports configurable building
blocks from `active_inference` and arranges them into a single figure or
short numerical study. Each script accepts standard CLI flags:

```bash
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py --save
# or via the top-level menu:
./run.sh --chapter 2
```

## Scripts

| Script | Mirrors | What it adds |
|--------|---------|--------------|
| `example_2_1_linear_deterministic.py` | Example 2.1 | Bayesian inversion of a noiseless linear sensor. |
| `example_2_2_linear_probabilistic.py` | Example 2.2 | Standard Gaussian likelihood × Gaussian prior. |
| `example_2_3_precision.py`            | Example 2.3 | Sweep prior vs likelihood precision; plot the trade-off. |
| `example_2_4_nonlinear_deterministic.py` | Example 2.4 | Quadratic generator → bi-modal posterior. |
| `example_2_5_nonlinear_probabilistic.py` | Example 2.5 | Nonlinear generator with Gaussian noise. |
| `example_2_6_imperfect_model.py`      | Example 2.6 | Mismatch between generative process and model. |
| `example_2_7_multiple_samples.py`     | Example 2.7 | Sequential vs batch inference over N i.i.d. samples. |
| `example_2_8_mle_analytic.py`         | Example 2.8 | Closed-form MLE compared to grid-Bayesian mode. |
| `example_2_9_map_analytic.py`         | Example 2.9 | Closed-form MAP compared to grid-Bayesian mode. |
| `example_2_10_gradient_descent.py`    | §2.5.2     | Iterative MLE / MAP via gradient descent. |
| `visualize_generative_model.py`       | §2.4       | Heatmap and 3D view of `p(x, y)`. |
| `interactive_explorer.py`             | bonus      | Slider-driven exploration of the canonical model. |
| `animation_sequential.py`             | bonus      | Animated posterior tightening as N grows (GIF). |
| `animation_gradient_descent.py`       | bonus      | Animated iterate rolling down the NLL (GIF). |

## Programmatic usage

```python
from active_inference import (
    LinearGaussianModel,
    LinearGaussianProcess,
    GridBayesianInference,
    make_grid,
)

process = LinearGaussianProcess(beta0=3, beta1=2, sigma2_y=0.25)
y = process.sample(x_star=2.0, n=1)[0]

model = LinearGaussianModel(
    beta0=3, beta1=2, sigma2_y=0.25, m_x=4, s2_x=0.25
)
result = GridBayesianInference(model, make_grid(0, 5, 500)).infer(y)
print(result.posterior_mode, result.credible_interval(0.95))
```
