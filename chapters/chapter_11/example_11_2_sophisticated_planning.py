"""Example 11.2 - sophisticated planning, preferences, forgetting, and structure learning."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, save_chapter_data
from active_inference import (
    simulate_parameter_forgetting,
    simulate_sophisticated_planning,
    simulate_state_preference_schedule,
    simulate_structure_learning,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic planning-extension diagnostics."""
    search, beliefs, entropies = simulate_sophisticated_planning(n_states=5, horizon=3)
    prefs = simulate_state_preference_schedule(
        np.array([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3], [0.1, 0.4, 2.1]])
    )
    forgetting = simulate_parameter_forgetting(np.array([1.0, 3.0, 8.0, 18.0]), rate=0.25)
    structures = simulate_structure_learning(np.array([1.2, 2.1, 1.8, 1.4]), complexity=0.12)
    return {
        "policy_index": np.arange(search.posterior.size, dtype=float),
        "policy_posterior": search.posterior,
        "policy_efe": search.expected_free_energies,
        "best_policy": search.best_policy.astype(float),
        "belief_step": np.arange(beliefs.shape[0], dtype=float),
        "future_beliefs": beliefs,
        "belief_entropy": entropies,
        "preference_schedule": prefs.schedule,
        "preference_targets": prefs.target_states.astype(float),
        "counts_before": forgetting.before,
        "counts_after": forgetting.after,
        "structure_evidence": structures.evidence,
        "structure_posterior": structures.posterior,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 11 planning-extension figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].bar(arrays["policy_index"], arrays["policy_posterior"])
    axes[0, 0].set_title("Tree policy posterior")
    axes[0, 0].set_xlabel("policy")
    axes[0, 0].set_ylabel("Q(pi)")
    axes[0, 1].plot(arrays["belief_step"], arrays["belief_entropy"], marker="o")
    axes[0, 1].set_title("Future belief entropy")
    axes[0, 1].set_xlabel("lookahead step")
    axes[0, 1].set_ylabel("H[s]")
    axes[1, 0].plot(arrays["counts_before"], marker="o", label="before")
    axes[1, 0].plot(arrays["counts_after"], marker="s", label="after forgetting")
    axes[1, 0].set_title("Parameter forgetting")
    axes[1, 0].set_xlabel("parameter")
    axes[1, 0].set_ylabel("pseudocount")
    axes[1, 0].legend()
    axes[1, 1].bar(np.arange(arrays["structure_posterior"].size), arrays["structure_posterior"])
    axes[1, 1].set_title("Structure posterior")
    axes[1, 1].set_xlabel("candidate")
    axes[1, 1].set_ylabel("probability")
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_11")
        figure = fig_dir / "example_11_2_sophisticated_planning.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            11,
            "example_11_2_sophisticated_planning",
            arrays,
            metadata={"script": "example_11_2_sophisticated_planning.py"},
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
