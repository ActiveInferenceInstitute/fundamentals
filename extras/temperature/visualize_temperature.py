"""Extra topic: temperature-scaled canonical probabilities and free energy."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    boltzmann_entropy,
    canonical_probabilities,
    default_figure_dir,
    ensure_dir,
    expected_energy,
    helmholtz_free_energy,
    save_extra_data,
)
from active_inference.extra_topics import extra_topic_spec
from active_inference.visualizations import COLORS, save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Render or display the temperature topic figure."""
    args = parse_args()
    energies = np.array([0.0, 1.0, 2.5, 4.0])
    temperatures = np.linspace(0.25, 5.0, 140)
    probabilities = np.vstack([
        canonical_probabilities(energies, temperature=float(t))
        for t in temperatures
    ])
    internal_energy = np.array([
        expected_energy(probabilities[i], energies)
        for i in range(probabilities.shape[0])
    ])
    entropy = np.array([boltzmann_entropy(row) for row in probabilities])
    free_energy = np.array([
        helmholtz_free_energy(internal_energy[i], entropy[i], temperatures[i])
        for i in range(temperatures.size)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for level in range(energies.size):
        axes[0].plot(temperatures, probabilities[:, level], lw=2,
                     color=list(COLORS.values())[level % len(COLORS)],
                     label=f"E={energies[level]:.1f}")
    axes[0].set_xlabel("temperature T")
    axes[0].set_ylabel("canonical probability")
    axes[0].set_title("Temperature flattens state probabilities")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(temperatures, internal_energy, label="U", lw=2, color=COLORS["prior"])
    axes[1].plot(
        temperatures,
        temperatures * entropy,
        label="T S",
        lw=2,
        color=COLORS["likelihood"],
    )
    axes[1].plot(
        temperatures,
        free_energy,
        label="A = U - T S",
        lw=2,
        color=COLORS["posterior"],
    )
    axes[1].set_xlabel("temperature T")
    axes[1].set_ylabel("nats / energy units")
    axes[1].set_title("Entropy-energy tradeoff")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Temperature as an entropy multiplier")

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "temperature")
        figure = save_or_show(fig, out / "visualize_temperature.png")
        save_extra_data(
            "temperature",
            "visualize_temperature",
            arrays={
                "energies": energies,
                "temperature": temperatures,
                "probabilities": probabilities,
                "internal_energy": internal_energy,
                "entropy": entropy,
                "helmholtz_free_energy": free_energy,
            },
            metadata={
                "script": "visualize_temperature.py",
                "source_apis": list(extra_topic_spec("temperature").source_apis),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
