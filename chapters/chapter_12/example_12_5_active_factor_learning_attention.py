"""Example 12.5 - active-inference factor messages with learning/attention."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import active_inference_factor_messages, default_figure_dir, ensure_dir, learning_attention_message, save_chapter_data


def build_arrays() -> dict[str, np.ndarray]:
    """Build arrays for §12.5-§12.7."""
    likelihood = np.array([[0.9, 0.15], [0.1, 0.85]])
    transition = np.array([[0.8, 0.25], [0.2, 0.75]])
    messages = active_inference_factor_messages(likelihood, transition, [0.65, 0.35])
    errors = np.array([0.4, -0.2, 0.1])
    attention = learning_attention_message(errors, np.log(3.0))
    return {
        "message_index": np.arange(2, dtype=float),
        "prior": messages["prior"],
        "prediction": messages["prediction"],
        "observation": messages["observation"],
        "posterior": messages["posterior"],
        "errors": errors,
        "attention_weighted_errors": attention,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render active factor messages and attention weighting."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key in ("prior", "prediction", "posterior"):
        axes[0].plot(arrays["message_index"], arrays[key], marker="o", label=key)
    axes[0].set_title("Active-inference messages")
    axes[0].legend()
    axes[1].plot(arrays["errors"], marker="o", label="error")
    axes[1].plot(arrays["attention_weighted_errors"], marker="s", label="precision weighted")
    axes[1].set_title("Learning/attention message")
    axes[1].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_12")
        figure = fig_dir / "example_12_5_active_factor_learning_attention.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(12, "example_12_5_active_factor_learning_attention", arrays, {"script": "example_12_5_active_factor_learning_attention.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 12 active-message example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
