"""Extra topic: Bayes' equation as prior-likelihood normalization."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    default_figure_dir,
    ensure_dir,
    make_grid,
    save_extra_data,
)
from active_inference.extra_topics import extra_topic_spec
from active_inference.visualizations import COLORS, save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--y-obs", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    """Render or display the Bayes equation topic figure."""
    args = parse_args()
    x_grid = make_grid(0.0, 5.0, 600)
    model = LinearGaussianModel(
        beta0=3.0,
        beta1=2.0,
        sigma2_y=0.25,
        m_x=4.0,
        s2_x=0.25,
        prior_kind="gaussian",
    )
    result = GridBayesianInference(model, x_grid).infer(args.y_obs)
    unnormalized = result.prior * result.likelihood
    evidence = float(np.trapezoid(unnormalized, x_grid))
    normalized = unnormalized / evidence

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x_grid, result.prior, label="prior p(x)", lw=2, color=COLORS["prior"])
    axes[0].plot(x_grid, result.likelihood / np.max(result.likelihood),
                 label="likelihood p(y|x), peak=1", lw=2, color=COLORS["likelihood"])
    axes[0].plot(
        x_grid, result.posterior, label="posterior p(x|y)", lw=2,
        color=COLORS["posterior"],
    )
    axes[0].set_xlabel("hidden state x")
    axes[0].set_ylabel("density / scaled credibility")
    axes[0].set_title("Bayes update ingredients")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(x_grid, unnormalized, label="likelihood * prior", lw=2,
                 color=COLORS["likelihood"])
    axes[1].plot(x_grid, normalized, label="normalized posterior", lw=2,
                 color=COLORS["posterior"])
    axes[1].fill_between(x_grid, normalized, alpha=0.2, color=COLORS["posterior"])
    axes[1].set_xlabel("hidden state x")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"evidence normalizes area: p(y) = {evidence:.4f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Bayes equation: posterior = likelihood * prior / evidence")

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "bayes_equation")
        figure = save_or_show(fig, out / "visualize_bayes_equation.png")
        save_extra_data(
            "bayes_equation",
            "visualize_bayes_equation",
            arrays={
                "x_grid": x_grid,
                "prior": result.prior,
                "likelihood": result.likelihood,
                "unnormalized": unnormalized,
                "posterior": result.posterior,
                "normalized": normalized,
                "evidence": np.array([evidence]),
                "log_evidence": np.array([result.log_evidence]),
            },
            metadata={
                "script": "visualize_bayes_equation.py",
                "y_obs": args.y_obs,
                "source_apis": list(extra_topic_spec("bayes_equation").source_apis),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
