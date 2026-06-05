"""Extra topic: variational free-energy decompositions and bound gap."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    default_figure_dir,
    ensure_dir,
    save_extra_data,
    variational_free_energy,
    vfe_thermodynamic_state,
)
from active_inference.extra_topics import extra_topic_spec
from active_inference.visualizations import COLORS, save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--y", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    """Render or display the variational free-energy topic figure."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model, x_grid).infer(args.y)
    mus = np.linspace(4.5, exact.posterior_mean, 36)
    variances = np.linspace(1.5, exact.posterior_variance, 36)
    components = [
        variational_free_energy(GaussianBelief(float(mu), float(var)), model, args.y, x_grid)
        for mu, var in zip(mus, variances)
    ]
    states = [
        vfe_thermodynamic_state(GaussianBelief(float(mu), float(var)), model, args.y, x_grid)
        for mu, var in zip(mus, variances)
    ]
    free_energy = np.array([c.free_energy for c in components])
    surprisal = np.array([c.surprisal for c in components])
    bound_gap = np.array([c.bound_gap for c in components])
    complexity = np.array([c.complexity for c in components])
    accuracy = np.array([c.accuracy for c in components])
    internal_energy = np.array([s.energy for s in states])
    entropy = np.array([s.entropy for s in states])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    iteration = np.arange(free_energy.size)
    axes[0].plot(iteration, free_energy, lw=2, label="F", color=COLORS["posterior"])
    axes[0].plot(
        iteration, surprisal, lw=2, ls="--", label="-log p(y)",
        color=COLORS["likelihood"],
    )
    axes[0].fill_between(
        iteration,
        surprisal,
        free_energy,
        alpha=0.2,
        color=COLORS["posterior"],
        label="bound gap",
    )
    axes[0].set_xlabel("belief interpolation step")
    axes[0].set_ylabel("nats")
    axes[0].set_title("VFE upper-bounds surprisal")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(iteration, complexity, lw=2, label="complexity", color=COLORS["prior"])
    axes[1].plot(
        iteration, -accuracy, lw=2, label="-accuracy", color=COLORS["likelihood"]
    )
    axes[1].plot(iteration, internal_energy - entropy, lw=2, ls=":",
                 label="U - S", color=COLORS["posterior"])
    axes[1].set_xlabel("belief interpolation step")
    axes[1].set_ylabel("nats")
    axes[1].set_title("Equivalent decompositions")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Variational free energy as a thermodynamic bridge")

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "variational_free_energy")
        figure = save_or_show(fig, out / "visualize_variational_free_energy.png")
        save_extra_data(
            "variational_free_energy",
            "visualize_variational_free_energy",
            arrays={
                "iteration": iteration,
                "mu": mus,
                "variance": variances,
                "free_energy": free_energy,
                "surprisal": surprisal,
                "bound_gap": bound_gap,
                "complexity": complexity,
                "accuracy": accuracy,
                "internal_energy": internal_energy,
                "entropy": entropy,
            },
            metadata={
                "script": "visualize_variational_free_energy.py",
                "y": args.y,
                "source_apis": list(
                    extra_topic_spec("variational_free_energy").source_apis
                ),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
