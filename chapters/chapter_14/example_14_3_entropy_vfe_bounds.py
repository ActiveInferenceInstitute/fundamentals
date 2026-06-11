"""Example 14.3 - entropy and VFE-bound bridge."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, entropy_vfe_bound_curve, phase1_fep_bridge, save_chapter_data


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic entropy/VFE-bound arrays for §14.3."""
    time = np.linspace(0.0, 1.0, 120)
    entropy = 0.5 + 0.4 * (1.0 - np.exp(-4.0 * time))
    entropy_curve, bound = entropy_vfe_bound_curve(entropy, margin=0.35)
    bridge = phase1_fep_bridge(float(entropy_curve[-1]), float(bound[-1]), viability=0.82)
    return {"time": time, "entropy": entropy_curve, "vfe_bound": bound, "phase1_bridge": bridge}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render entropy bounds and the Phase-I bridge summary."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["entropy"], label="entropy")
    axes[0].plot(arrays["time"], arrays["vfe_bound"], label="VFE-like bound")
    axes[0].set_title("Entropy upper bound")
    axes[0].legend()
    axes[1].bar([0, 1, 2], arrays["phase1_bridge"], color="tab:red")
    axes[1].set_xticks([0, 1, 2], ["H", "F", "1-viability"])
    axes[1].set_title("Phase-I FEP bridge")
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_14")
        figure = fig_dir / "example_14_3_entropy_vfe_bounds.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(14, "example_14_3_entropy_vfe_bounds", arrays, {"script": "example_14_3_entropy_vfe_bounds.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 14 entropy-bound example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
