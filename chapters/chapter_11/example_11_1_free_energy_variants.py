"""Example 11.1 - Part III free-energy variants over policy scores."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    free_energy_variant_table,
    renyi_bound,
    renyi_limit_energy,
    save_chapter_data,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic policy-indexed free-energy variant arrays."""
    policies = np.arange(6, dtype=float)
    risk = np.array([1.8, 1.25, 0.82, 0.72, 1.1, 1.55])
    ambiguity = np.array([0.25, 0.42, 0.66, 0.81, 0.48, 0.3])
    epistemic = np.array([0.15, 0.35, 0.72, 0.55, 0.28, 0.12])
    table = free_energy_variant_table(risk, ambiguity, epistemic)
    alphas = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    probs = np.ones_like(risk) / risk.size
    energies = risk + ambiguity
    renyi = np.array([
        renyi_limit_energy(probs, energies) if np.isclose(alpha, 1.0)
        else renyi_bound(probs, energies, alpha)
        for alpha in alphas
    ])
    return {
        "policies": policies,
        "risk": risk,
        "ambiguity": ambiguity,
        "epistemic_value": epistemic,
        "expected_free_energy": table["expected_free_energy"],
        "free_energy_of_future": table["free_energy_of_future"],
        "generalized_free_energy": table["generalized_free_energy"],
        "renyi_alpha": alphas,
        "renyi_bound": renyi,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 11 free-energy variants figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["policies"], arrays["expected_free_energy"], marker="o", label="EFE")
    axes[0].plot(arrays["policies"], arrays["free_energy_of_future"], marker="s", label="FEF")
    axes[0].plot(arrays["policies"], arrays["generalized_free_energy"], marker="^", label="GFE")
    axes[0].set_xlabel("policy")
    axes[0].set_ylabel("score")
    axes[0].set_title("Policy free-energy variants")
    axes[0].legend()
    axes[1].plot(arrays["renyi_alpha"], arrays["renyi_bound"], marker="o", color="black")
    axes[1].axvline(1.0, color="0.65", linestyle="--", label="expected-energy limit")
    axes[1].set_xlabel("Renyi alpha")
    axes[1].set_ylabel("certainty equivalent")
    axes[1].set_title("Renyi-style bound")
    axes[1].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_11")
        figure = fig_dir / "example_11_1_free_energy_variants.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            11,
            "example_11_1_free_energy_variants",
            arrays,
            metadata={"script": "example_11_1_free_energy_variants.py"},
            figures=[figure],
        )
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
