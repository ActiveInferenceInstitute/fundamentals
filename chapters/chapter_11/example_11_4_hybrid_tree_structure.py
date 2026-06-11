"""Example 11.4 - hybrid models, tree search, and structure learning."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    hybrid_model_bridge,
    save_chapter_data,
    simulate_sophisticated_planning,
    simulate_structure_learning,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build arrays for §11.3-§11.5."""
    search, beliefs, entropies = simulate_sophisticated_planning(n_states=6, horizon=4)
    structures = simulate_structure_learning([1.0, 1.7, 2.0, 1.6], complexity=0.18)
    hybrid = hybrid_model_bridge([0.2, 0.7, 1.1], [0.2, 0.5, 0.3])
    return {
        "policy_index": np.arange(search.posterior.size, dtype=float),
        "policy_posterior": search.posterior,
        "policy_efe": search.expected_free_energies,
        "beliefs": beliefs,
        "belief_entropy": entropies,
        "structure_index": np.arange(structures.posterior.size, dtype=float),
        "structure_posterior": structures.posterior,
        "hybrid_joint": hybrid,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the hybrid/tree/structure figure."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(arrays["policy_index"], arrays["policy_posterior"])
    axes[0].set_title("Tree-search posterior")
    axes[0].set_xlabel("policy")
    axes[1].imshow(arrays["hybrid_joint"], aspect="auto", origin="lower")
    axes[1].set_title("Hybrid joint bridge")
    axes[1].set_xlabel("continuous feature")
    axes[1].set_ylabel("discrete state")
    axes[2].bar(arrays["structure_index"], arrays["structure_posterior"], color="tab:green")
    axes[2].set_title("Structure posterior")
    axes[2].set_xlabel("candidate")
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_11")
        figure = fig_dir / "example_11_4_hybrid_tree_structure.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(11, "example_11_4_hybrid_tree_structure", arrays, {"script": "example_11_4_hybrid_tree_structure.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 11 hybrid/tree example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
