"""Extra topic: enthalpy and Gibbs free energy as explicit analogy terms."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    enthalpy,
    gibbs_free_energy,
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
    """Render or display the enthalpy topic figure."""
    args = parse_args()
    energy = 2.0
    entropy = 1.2
    temperature = 1.0
    pressures = np.linspace(0.0, 3.0, 120)
    volumes = np.array([0.0, 0.5, 1.0, 1.5])
    enthalpies = np.array([
        [enthalpy(energy, pressure=float(p), volume=float(v)) for p in pressures]
        for v in volumes
    ])
    gibbs_values = np.array([
        [
            gibbs_free_energy(
                energy,
                entropy,
                temperature,
                pressure=float(p),
                volume=float(v),
            )
            for p in pressures
        ]
        for v in volumes
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, volume in enumerate(volumes):
        color = list(COLORS.values())[i % len(COLORS)]
        axes[0].plot(pressures, enthalpies[i], lw=2, color=color, label=f"V={volume:.1f}")
        axes[1].plot(pressures, gibbs_values[i], lw=2, color=color, label=f"V={volume:.1f}")
    axes[0].set_xlabel("pressure p")
    axes[0].set_ylabel("H")
    axes[0].set_title("Enthalpy H = U + pV")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("pressure p")
    axes[1].set_ylabel("G")
    axes[1].set_title("Gibbs free energy G = H - T S")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Pressure-volume terms are caller-supplied analogy terms")

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "enthalpy")
        figure = save_or_show(fig, out / "visualize_enthalpy.png")
        save_extra_data(
            "enthalpy",
            "visualize_enthalpy",
            arrays={
                "pressure": pressures,
                "volume": volumes,
                "enthalpy": enthalpies,
                "gibbs_free_energy": gibbs_values,
                "energy": np.array([energy]),
                "entropy": np.array([entropy]),
                "temperature": np.array([temperature]),
            },
            metadata={
                "script": "visualize_enthalpy.py",
                "source_apis": list(extra_topic_spec("enthalpy").source_apis),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
